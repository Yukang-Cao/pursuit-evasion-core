// Signed-speed relative capture-tube solver, adapted from
// research/dp_capture/hji.cpp. Standard ENO/Lax-Friedrichs spatial update and
// RK1/RK2 time stepping. Storage axes: [vP, vE, psi, y, x]. Public relative
// state: [x,y,psi,vP,vE].
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

struct Config {
  int nx = 41, np = 24, nvp = 7, nve = 7, threads = 4, order = 2;
  double extent = 4, T = 2, outdt = .1, cfl = .8, r = .3;
  double vpmin = -1, vpmax = 2, vemin = -.5, vemax = 1, ap = 2, ae = 1, kp = 1,
         ke = 2;
  bool history = true;
  std::string out;
};
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
static std::string arg(int n, char **v, const std::string &key,
                       const std::string &fallback) {
  for (int i = 1; i + 1 < n; i++)
    if (v[i] == key)
      return v[i + 1];
  return fallback;
}
inline void slopes(const float *s, size_t j, size_t stride, int pos, int n,
                   double h, bool periodic, int order, double &dm, double &dp) {
  auto at = [&](int off) -> double {
    int q = pos + off;
    if (periodic)
      q = (q % n + n) % n;
    return s[std::ptrdiff_t(j) + std::ptrdiff_t(q - pos) * stride];
  };
  const double v = s[j];
  if (!periodic && pos == 0) {
    dm = dp = (at(1) - v) / h;
    return;
  }
  if (!periodic && pos == n - 1) {
    dm = dp = (v - at(-1)) / h;
    return;
  }
  const double vm = at(-1), vp = at(1);
  dm = (v - vm) / h;
  dp = (vp - v) / h;
  if (order == 1)
    return;
  const double sc = (vp - 2 * v + vm) / (h * h),
               sl = (periodic || pos >= 2) ? (v - 2 * vm + at(-2)) / (h * h)
                                           : sc;
  const double sr =
      (periodic || pos + 2 < n) ? (at(2) - 2 * vp + v) / (h * h) : sc;
  dm += .5 * h * (std::abs(sl) < std::abs(sc) ? sl : sc);
  dp -= .5 * h * (std::abs(sc) < std::abs(sr) ? sc : sr);
}
int run(const Config &c) {
  const auto started = Clock::now();
  const double pi = std::acos(-1.);
  const double dx = 2 * c.extent / (c.nx - 1), dphi = 2 * pi / c.np;
  const double dvp = (c.vpmax - c.vpmin) / (c.nvp - 1),
               dve = (c.vemax - c.vemin) / (c.nve - 1);
  const size_t sy = c.nx, sp = size_t(c.nx) * c.nx, se = sp * c.np,
               sv = se * c.nve, N = sv * c.nvp;
  const double VP = std::max(std::abs(c.vpmin), std::abs(c.vpmax)),
               VE = std::max(std::abs(c.vemin), std::abs(c.vemax));
  const double sum = (VP + VE + VP * c.kp * c.extent) / dx +
                     (VE + VP * c.kp * c.extent) / dx +
                     (VP * c.kp + VE * c.ke) / dphi + c.ap / dvp + c.ae / dve;
  const double limit = c.cfl / sum;
  std::vector<float> a(N), b(N),
      arrival(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<double> sn(c.np), cs(c.np);
  for (int k = 0; k < c.np; k++) {
    const double q = -pi + k * dphi;
    sn[k] = std::sin(q);
    cs[k] = std::cos(q);
  }
  omp_set_num_threads(c.threads);
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < N / c.nx; block++) {
    const double y = -c.extent + (block % c.nx) * dx;
    for (int ix = 0; ix < c.nx; ix++) {
      size_t j = block * c.nx + ix;
      a[j] = float(std::hypot(-c.extent + ix * dx, y) - c.r);
      if (a[j] <= 0)
        arrival[j] = 0;
    }
  }
  const double allocation_seconds = elapsed(started);
  std::ofstream history;
  if (c.history) {
    history.open(c.out + "/fields.f32", std::ios::binary);
    if (!history)
      throw std::runtime_error("Cannot open history");
  }
  double write_seconds = 0, update_seconds = 0, scan_seconds = 0, t = 0,
         maxincrease = 0;
  int steps = 0, frames = 0;
  auto save = [&]() {
    const auto before = Clock::now();
    if (c.history) {
      history.write(reinterpret_cast<const char *>(a.data()),
                    N * sizeof(float));
      if (!history)
        throw std::runtime_error("History write failed");
    }
    write_seconds += elapsed(before);
    frames++;
  };
  save();
  std::ofstream trace(c.out + "/progress.jsonl");
  auto advance = [&](const float *src, float *dst, double dt, bool blend) {
    double increase = 0;
#pragma omp parallel for schedule(static) reduction(max : increase)
    for (size_t block = 0; block < N / c.nx; block++) {
      const int iy = block % c.nx, ip = (block / c.nx) % c.np,
                ie = (block / (size_t(c.nx) * c.np)) % c.nve,
                iv = block / (size_t(c.nx) * c.np * c.nve);
      const double y = -c.extent + iy * dx, vp = c.vpmin + iv * dvp,
                   ve = c.vemin + ie * dve;
      const double wp = std::abs(vp) * c.kp, we = std::abs(ve) * c.ke,
                   bx = ve * cs[ip] - vp, by = ve * sn[ip];
      for (int ix = 0; ix < c.nx; ix++) {
        const size_t j = block * c.nx + ix;
        const double x = -c.extent + ix * dx;
        double xm, xp, ym, yp, pm, pp, vm, vplus, em, ep;
        slopes(src, j, 1, ix, c.nx, dx, false, c.order, xm, xp);
        slopes(src, j, sy, iy, c.nx, dx, false, c.order, ym, yp);
        slopes(src, j, sp, ip, c.np, dphi, true, c.order, pm, pp);
        slopes(src, j, sv, iv, c.nvp, dvp, false, c.order, vm, vplus);
        slopes(src, j, se, ie, c.nve, dve, false, c.order, em, ep);
        const double gx = .5 * (xm + xp), gy = .5 * (ym + yp),
                     gp = .5 * (pm + pp), gv = .5 * (vm + vplus),
                     ge = .5 * (em + ep);
        const double plow = iv == 0 ? 0 : -c.ap,
                     phigh = iv == c.nvp - 1 ? 0 : c.ap,
                     elow = ie == 0 ? 0 : -c.ae,
                     ehigh = ie == c.nve - 1 ? 0 : c.ae;
        double H = bx * gx + by * gy - wp * std::abs(y * gx - x * gy - gp) +
                   we * std::abs(gp) + std::min(plow * gv, phigh * gv) +
                   std::max(elow * ge, ehigh * ge);
        H += .5 *
             ((std::abs(bx) + wp * std::abs(y)) * (xp - xm) +
              (std::abs(by) + wp * std::abs(x)) * (yp - ym) +
              (wp + we) * (pp - pm) + c.ap * (vplus - vm) + c.ae * (ep - em));
        const double old = src[j], previous = blend ? dst[j] : old;
        double value = old + dt * std::min(0., H);
        if (blend)
          value = .5 * (previous + value);
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite field update");
        dst[j] = float(value);
        increase = std::max(increase, double(dst[j]) - previous);
        if ((blend || c.order == 1) && previous > 0 && dst[j] <= 0 &&
            std::isnan(arrival[j]))
          arrival[j] = float(t + dt * previous / (previous - double(dst[j])));
      }
    }
    return increase;
  };
  const int outputs = int(std::ceil(c.T / c.outdt - 1e-10));
  for (int out = 1; out <= outputs; out++) {
    const double goal = std::min(c.T, out * c.outdt);
    while (t < goal - 1e-12) {
      const double dt = std::min(limit, goal - t);
      auto before = Clock::now();
      double inc;
      if (c.order == 1) {
        inc = advance(a.data(), b.data(), dt, false);
        a.swap(b);
      } else {
        advance(a.data(), b.data(), dt, false);
        inc = advance(b.data(), a.data(), dt, true);
      }
      update_seconds += elapsed(before);
      maxincrease = std::max(maxincrease, inc);
      t += dt;
      steps++;
    }
    save();
    if (out % 5 == 0 || out == outputs) {
      auto before = Clock::now();
      size_t resolved = 0;
      for (float x : arrival)
        resolved += std::isfinite(x);
      scan_seconds += elapsed(before);
      trace << std::setprecision(12) << "{\"time\":" << t
            << ",\"steps\":" << steps << ",\"resolved\":" << resolved
            << ",\"elapsed_seconds\":" << elapsed(started) << "}\n";
      trace.flush();
      std::cout << "budget=" << t << " steps=" << steps
                << " resolved=" << resolved << " elapsed=" << elapsed(started)
                << std::endl;
    }
  }
  if (history.is_open())
    history.close();
  auto before = Clock::now();
  for (const auto &item : std::vector<std::pair<std::string, const float *>>{
           {"final.f32", a.data()}, {"arrival.f32", arrival.data()}}) {
    std::ofstream f(c.out + "/" + item.first, std::ios::binary);
    f.write(reinterpret_cast<const char *>(item.second), N * sizeof(float));
    if (!f)
      throw std::runtime_error("Final field write failed");
  }
  const double final_write = elapsed(before);
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::ofstream meta(c.out + "/timing.json");
  meta << std::setprecision(14) << "{\"states\":" << N
       << ",\"working_array_bytes\":" << 12 * N
       << ",\"peak_rss_kib\":" << usage.ru_maxrss
       << ",\"cfl_dt_limit\":" << limit << ",\"internal_steps\":" << steps
       << ",\"frames\":" << frames
       << ",\"allocation_seconds\":" << allocation_seconds
       << ",\"update_seconds\":" << update_seconds
       << ",\"history_write_seconds\":" << write_seconds
       << ",\"final_write_seconds\":" << final_write
       << ",\"summary_scan_seconds\":" << scan_seconds
       << ",\"elapsed_seconds\":" << elapsed(started)
       << ",\"maximum_increase\":" << maxincrease << "}\n";
  return 0;
}
int main(int n, char **v) {
  try {
    Config c;
    c.nx = std::stoi(arg(n, v, "--positions", "41"));
    c.np = std::stoi(arg(n, v, "--headings", "24"));
    c.nvp = std::stoi(arg(n, v, "--p-speeds", "7"));
    c.nve = std::stoi(arg(n, v, "--e-speeds", "7"));
    c.extent = std::stod(arg(n, v, "--extent", "4"));
    c.T = std::stod(arg(n, v, "--horizon", "2"));
    c.outdt = std::stod(arg(n, v, "--output-dt", ".1"));
    c.cfl = std::stod(arg(n, v, "--cfl", ".8"));
    c.r = std::stod(arg(n, v, "--radius", ".3"));
    c.vpmin = std::stod(arg(n, v, "--p-min", "-1"));
    c.vpmax = std::stod(arg(n, v, "--p-max", "2"));
    c.vemin = std::stod(arg(n, v, "--e-min", "-.5"));
    c.vemax = std::stod(arg(n, v, "--e-max", "1"));
    c.ap = std::stod(arg(n, v, "--p-acceleration", "2"));
    c.ae = std::stod(arg(n, v, "--e-acceleration", "1"));
    c.kp = std::stod(arg(n, v, "--p-curvature", "1"));
    c.ke = std::stod(arg(n, v, "--e-curvature", "2"));
    c.threads = std::stoi(arg(n, v, "--threads", "4"));
    c.order = std::stoi(arg(n, v, "--order", "2"));
    c.history = arg(n, v, "--history", "1") == "1";
    c.out = arg(n, v, "--output", "");
    if (c.nx < 3 || c.np < 4 || c.nvp < 3 || c.nve < 3 || c.threads < 1 ||
        (c.order != 1 && c.order != 2) || c.extent <= c.r || c.r <= 0 ||
        c.T <= 0 || c.outdt <= 0 || c.cfl <= 0 || c.cfl > 1 || c.ap <= 0 ||
        c.ae <= 0 || c.kp <= 0 || c.ke <= 0 || c.vpmin >= c.vpmax ||
        c.vemin >= c.vemax || c.out.empty())
      throw std::runtime_error("Invalid solver options");
    return run(c);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
