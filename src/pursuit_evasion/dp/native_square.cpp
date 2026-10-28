// Six-dimensional square reach-avoid game. See docs/model.md for event
// semantics. Float32 fields, float32 local arithmetic, ENO1/ENO2, and TVD
// RK1/RK2.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <sys/resource.h>
#include <vector>
using Real = float;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
std::string arg(int argc, char **argv, const std::string &k,
                const std::string &fallback) {
  for (int i = 1; i + 1 < argc; i++)
    if (argv[i] == "--" + k)
      return argv[i + 1];
  return fallback;
}

inline void slopes(const Real *v, size_t j, size_t stride, int pos, int n,
                   Real h, bool periodic, int order, Real &left, Real &right) {
  auto at = [&](int offset) -> Real {
    int q = pos + offset;
    if (periodic) {
      q = (q % n + n) % n;
      return v[std::ptrdiff_t(j) + std::ptrdiff_t(q - pos) * stride];
    }
    if (q < 0) {
      size_t edge = j - size_t(pos) * stride;
      Real a = v[edge], b = v[edge + stride];
      return a + (-q) * std::abs(b - a) * ((a > 0) - (a < 0));
    }
    if (q >= n) {
      size_t edge = j + size_t(n - 1 - pos) * stride;
      Real a = v[edge], b = v[edge - stride];
      return a + (q - n + 1) * std::abs(a - b) * ((a > 0) - (a < 0));
    }
    return v[std::ptrdiff_t(j) + std::ptrdiff_t(offset) * stride];
  };
  Real f = v[j], fm = at(-1), fp = at(1);
  left = (f - fm) / h;
  right = (fp - f) / h;
  if (order == 2) {
    Real dl = (f - 2 * fm + at(-2)) / (h * h), dc = (fp - 2 * f + fm) / (h * h),
         dr = (at(2) - 2 * fp + f) / (h * h);
    Real a = std::abs(dl) <= std::abs(dc) ? dl : dc,
         b = std::abs(dc) <= std::abs(dr) ? dc : dr;
    left += .5f * h * a;
    right -= .5f * h * b;
  }
}

int main(int argc, char **argv) {
  const int n = std::stoi(arg(argc, argv, "positions", "17")),
            nh = std::stoi(arg(argc, argv, "headings", "16"));
  const int threads = std::stoi(arg(argc, argv, "threads", "8")),
            order = std::stoi(arg(argc, argv, "order", "2"));
  const Real extent = std::stof(arg(argc, argv, "extent", "3.5")),
             L = std::stof(arg(argc, argv, "arena", "3")),
             r = std::stof(arg(argc, argv, "radius", ".3"));
  const Real vp = std::stof(arg(argc, argv, "vp", "1")),
             ve = std::stof(arg(argc, argv, "ve", ".3")),
             wp = std::stof(arg(argc, argv, "wp", "1")),
             we = std::stof(arg(argc, argv, "we", "2"));
  const double horizon = std::stod(arg(argc, argv, "horizon", "2")),
               outdt = std::stod(arg(argc, argv, "output-dt", ".1")),
               cfl = std::stod(arg(argc, argv, "cfl", ".8"));
  const bool history = arg(argc, argv, "history", "1") == "1";
  const std::string out = arg(argc, argv, "output", "");
  if (out.empty() || n < 5 || nh < 4 || extent <= L || order < 1 || order > 2) {
    std::cerr << "Invalid grid/configuration/output\n";
    return 2;
  }
  std::filesystem::create_directories(out);
  omp_set_num_threads(threads);
  const double pi = std::acos(-1.);
  const Real dx = 2 * extent / (n - 1), dh = Real(2 * pi / nh);
  const std::array<int, 6> shape = {n, n, nh, n, n, nh};
  std::array<size_t, 6> stride;
  stride[5] = 1;
  for (int k = 4; k >= 0; k--)
    stride[k] = stride[k + 1] * shape[k + 1];
  const size_t N = stride[0] * n;
  const double maxdt =
      cfl / (2 * (vp + ve) / double(dx) + (wp + we) / double(dh));
  auto start = Clock::now();
  std::vector<Real> a(N), b(N),
      arrival(N, std::numeric_limits<Real>::quiet_NaN());
  std::vector<Real> x(n), cs(nh), sn(nh);
  for (int i = 0; i < n; i++)
    x[i] = -extent + i * dx;
  for (int i = 0; i < nh; i++) {
    cs[i] = std::cos(-pi + i * 2 * pi / nh);
    sn[i] = std::sin(-pi + i * 2 * pi / nh);
  }
  auto decode = [&](size_t j) {
    std::array<int, 6> q;
    for (int k = 0; k < 6; k++)
      q[k] = (j / stride[k]) % shape[k];
    return q;
  };
  auto surfaces = [&](const std::array<int, 6> &q, Real &target, Real &avoid) {
    target = std::min(std::hypot(x[q[0]] - x[q[3]], x[q[1]] - x[q[4]]) - r,
                      L - std::max(std::abs(x[q[3]]), std::abs(x[q[4]])));
    avoid = std::max(std::abs(x[q[0]]), std::abs(x[q[1]])) - L + 1e-6f;
  };
#pragma omp parallel for schedule(static)
  for (size_t j = 0; j < N; j++) {
    Real target, avoid;
    surfaces(decode(j), target, avoid);
    a[j] = std::max(target, avoid);
    if (a[j] <= 0)
      arrival[j] = 0;
  }
  double init_seconds = seconds(start), update_seconds = 0, write_seconds = 0;
  int steps = 0, frames = 0;
  std::ofstream fields;
  if (history)
    fields.open(out + "/fields.f32", std::ios::binary);
  auto save = [&]() {
    auto t = Clock::now();
    if (history) {
      fields.write(reinterpret_cast<const char *>(a.data()), N * sizeof(Real));
      if (!fields)
        throw std::runtime_error("field write failed");
    }
    write_seconds += seconds(t);
    frames++;
  };
  save();
  auto advance = [&](const Real *src, Real *dst, Real dt, const Real *base,
                     double tau) {
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; j++) {
      const auto q = decode(j);
      const std::array<Real, 6> f = {vp * cs[q[2]], vp * sn[q[2]], 0,
                                     ve * cs[q[5]], ve * sn[q[5]], 0};
      std::array<Real, 6> g;
      Real diss = 0, H = 0;
      for (int k = 0; k < 6; k++) {
        Real lm, rp;
        slopes(src, j, stride[k], q[k], shape[k], (k == 2 || k == 5) ? dh : dx,
               k == 2 || k == 5, order, lm, rp);
        g[k] = (lm + rp) * .5f;
        Real alpha = k == 2 ? wp : k == 5 ? we : std::abs(f[k]);
        diss += alpha * (rp - lm) * .5f;
        H += f[k] * g[k];
      }
      H += -wp * std::abs(g[2]) + we * std::abs(g[5]);
      Real value = src[j] + dt * (H + diss);
      if (base) {
        value = .5f * (base[j] + value);
      }
      if (order == 1 || base) {
        Real previous = base ? base[j] : src[j];
        const Real avoid =
            std::max(std::abs(x[q[0]]), std::abs(x[q[1]])) - L + 1e-6f;
        value = std::max(std::min(value, previous), avoid);
        if (previous > 0 && value <= 0 && std::isnan(arrival[j]))
          arrival[j] = Real(tau) + dt * previous / (previous - value);
      }
      dst[j] = value;
    }
  };
  double tau = 0;
  for (int frame = 1; frame <= int(std::ceil(horizon / outdt - 1e-10));
       frame++) {
    double goal = std::min(horizon, frame * outdt);
    while (tau < goal - 1e-10) {
      double dt = std::min(maxdt, goal - tau);
      auto t = Clock::now();
      if (order == 1) {
        advance(a.data(), b.data(), Real(dt), nullptr, tau);
        a.swap(b);
      } else {
        advance(a.data(), b.data(), Real(dt), nullptr, tau);
        advance(b.data(), a.data(), Real(dt), a.data(), tau);
      }
      update_seconds += seconds(t);
      tau += dt;
      steps++;
    }
    save();
    std::cout << "budget " << tau << " steps " << steps << " updates_s "
              << update_seconds << std::endl;
  }
  auto t = Clock::now();
  std::ofstream final(out + "/final.f32", std::ios::binary);
  final.write(reinterpret_cast<const char *>(a.data()), N * 4);
  final.close();
  std::ofstream first(out + "/arrival.f32", std::ios::binary);
  first.write(reinterpret_cast<const char *>(arrival.data()), N * 4);
  first.close();
  write_seconds += seconds(t);
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::ofstream meta(out + "/native_timing.json");
  meta << std::setprecision(12) << "{\"states\":" << N
       << ",\"internal_steps\":" << steps << ",\"frames\":" << frames
       << ",\"update_seconds\":" << update_seconds
       << ",\"initialization_seconds\":" << init_seconds
       << ",\"write_seconds\":" << write_seconds
       << ",\"process_seconds\":" << seconds(start)
       << ",\"core_array_bytes\":" << N * 12
       << ",\"peak_rss_bytes\":" << usage.ru_maxrss * 1024ULL
       << ",\"max_internal_dt\":" << maxdt
       << ",\"arithmetic\":\"float32 fields and float32 local arithmetic\"}\n";
  return 0;
}
