// Eight-dimensional reversible car game in a fixed square.
// State: xP,yP,thetaP,vP,xE,yE,thetaE,vE. Controls: acceleration,curvature.
// Native ENO1/ENO2 + local LF dissipation + RK1/RK2, float32 fields.
// Boundary target is capture OR E fault; P fault is avoid. See model contract.
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
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>
using Real = float;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
std::string arg(int argc, char **argv, const std::string &k,
                const std::string &def) {
  for (int i = 1; i + 1 < argc; i++)
    if (argv[i] == "--" + k)
      return argv[i + 1];
  return def;
}
inline void slopes(const Real *v, size_t j, size_t stride, int pos, int n,
                   Real h, bool periodic, bool linear_boundary, int order,
                   Real &left, Real &right) {
  // Speed bounds are physical state boundaries. Use inward one-sided slopes,
  // not signed-distance extrapolation intended for spatial grid truncation.
  if (linear_boundary && pos == 0) {
    left = right = (v[j + stride] - v[j]) / h;
    return;
  }
  if (linear_boundary && pos == n - 1) {
    left = right = (v[j] - v[j - stride]) / h;
    return;
  }
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
    Real dc = (fp - 2 * f + fm) / (h * h);
    Real dl = linear_boundary && pos < 2 ? dc : (f - 2 * fm + at(-2)) / (h * h);
    Real dr =
        linear_boundary && pos + 2 >= n ? dc : (at(2) - 2 * fp + f) / (h * h);
    left += .5f * h * (std::abs(dl) <= std::abs(dc) ? dl : dc);
    right -= .5f * h * (std::abs(dc) <= std::abs(dr) ? dc : dr);
  }
}

int run(int argc, char **argv) {
  const int n = std::stoi(arg(argc, argv, "positions", "9")),
            nh = std::stoi(arg(argc, argv, "headings", "8")),
            np = std::stoi(arg(argc, argv, "p-speeds", "7")),
            ne = std::stoi(arg(argc, argv, "e-speeds", "7")),
            threads = std::stoi(arg(argc, argv, "threads", "8")),
            order = std::stoi(arg(argc, argv, "order", "2"));
  const Real extent = std::stof(arg(argc, argv, "extent", "3.5")),
             L = std::stof(arg(argc, argv, "arena", "3")),
             radius = std::stof(arg(argc, argv, "radius", ".3")),
             pmin = std::stof(arg(argc, argv, "p-min", "-1")),
             pmax = std::stof(arg(argc, argv, "p-max", "2")),
             emin = std::stof(arg(argc, argv, "e-min", "-.5")),
             emax = std::stof(arg(argc, argv, "e-max", "1")),
             ap = std::stof(arg(argc, argv, "p-accel", "2")),
             ae = std::stof(arg(argc, argv, "e-accel", "1")),
             kp = std::stof(arg(argc, argv, "p-curvature", "1")),
             ke = std::stof(arg(argc, argv, "e-curvature", "2"));
  const double T = std::stod(arg(argc, argv, "horizon", ".5")),
               outdt = std::stod(arg(argc, argv, "output-dt", ".25")),
               cfl = std::stod(arg(argc, argv, "cfl", ".8")),
               start_time = std::stod(arg(argc, argv, "start-time", "0"));
  const bool history = arg(argc, argv, "history", "0") == "1";
  const bool linear_speed_boundary =
      arg(argc, argv, "linear-speed-boundary", "1") == "1";
  const std::string out = arg(argc, argv, "output", "");
  const std::string initial_field = arg(argc, argv, "initial-field", ""),
                    initial_arrival = arg(argc, argv, "initial-arrival", "");
  const size_t initial_offset =
      std::stoull(arg(argc, argv, "initial-offset", "0"));
  const bool write_final = arg(argc, argv, "write-final", "1") == "1",
             write_arrival = arg(argc, argv, "write-arrival", "1") == "1";
  const size_t probe_count = std::stoull(arg(argc, argv, "probe-count", "0"));
  if (out.empty() || n < 5 || nh < 4 || np < 3 || ne < 3 || extent <= L ||
      order < 1 || order > 2 || T <= 0 || outdt <= 0 || cfl <= 0 || cfl > 1 ||
      pmin >= pmax || emin >= emax || ap <= 0 || ae <= 0 || kp <= 0 ||
      ke <= 0 || threads < 1 || start_time < 0 || start_time >= T ||
      (start_time > 0 && initial_field.empty()) || (!history && !write_final))
    throw std::runtime_error("Invalid arguments");
  std::filesystem::create_directories(out);
  omp_set_num_threads(threads);
  const double pi = std::acos(-1.);
  const Real dx = 2 * extent / (n - 1), dh = Real(2 * pi / nh),
             dvp = (pmax - pmin) / (np - 1), dve = (emax - emin) / (ne - 1);
  const Real pabs = std::max(std::abs(pmin), std::abs(pmax)),
             eabs = std::max(std::abs(emin), std::abs(emax));
  const std::array<int, 8> shape = {n, n, nh, np, n, n, nh, ne};
  const std::array<Real, 8> spacing = {dx, dx, dh, dvp, dx, dx, dh, dve};
  std::array<size_t, 8> stride;
  stride[7] = 1;
  for (int k = 6; k >= 0; k--) {
    if (stride[k + 1] >
        std::numeric_limits<size_t>::max() / size_t(shape[k + 1]))
      throw std::runtime_error("Size overflow");
    stride[k] = stride[k + 1] * shape[k + 1];
  }
  const size_t N = stride[0] * n, inner = size_t(nh) * ne;
  const double maxdt =
      cfl / (2 * double(pabs + eabs) / dx + double(pabs * kp + eabs * ke) / dh +
             double(ap) / dvp + double(ae) / dve);
  auto start = Clock::now();
  std::vector<Real> a(N), b(N),
      arrival(N, std::numeric_limits<Real>::quiet_NaN());
  std::vector<Real> x(n), cs(nh), sn(nh), pv(np), ev(ne);
  for (int i = 0; i < n; i++)
    x[i] = -extent + i * dx;
  for (int i = 0; i < nh; i++) {
    cs[i] = std::cos(-pi + i * 2 * pi / nh);
    sn[i] = std::sin(-pi + i * 2 * pi / nh);
  }
  for (int i = 0; i < np; i++)
    pv[i] = pmin + i * dvp;
  for (int i = 0; i < ne; i++)
    ev[i] = emin + i * dve;
  auto decode = [&](size_t j) {
    std::array<int, 8> q;
    for (int k = 0; k < 8; k++)
      q[k] = (j / stride[k]) % shape[k];
    return q;
  };
  auto surfaces = [&](const std::array<int, 8> &q, Real &target, Real &avoid) {
    target = std::min(std::hypot(x[q[0]] - x[q[4]], x[q[1]] - x[q[5]]) - radius,
                      L - std::max(std::abs(x[q[4]]), std::abs(x[q[5]])));
    avoid = std::max(std::abs(x[q[0]]), std::abs(x[q[1]])) - L + 1e-6f;
  };
  if (initial_field.empty()) {
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < N / inner; block++) {
      const size_t begin = block * inner;
      Real target, avoid;
      surfaces(decode(begin), target, avoid);
      Real initial = std::max(target, avoid);
      for (size_t k = 0; k < inner; k++) {
        a[begin + k] = initial;
        if (initial <= 0)
          arrival[begin + k] = 0;
      }
    }
  } else {
    std::ifstream input(initial_field, std::ios::binary);
    input.seekg(initial_offset);
    input.read(reinterpret_cast<char *>(a.data()), N * sizeof(Real));
    if (!input)
      throw std::runtime_error("Cannot load complete continuation field");
    if (!initial_arrival.empty()) {
      std::ifstream first(initial_arrival, std::ios::binary);
      first.read(reinterpret_cast<char *>(arrival.data()), N * sizeof(Real));
      if (!first)
        throw std::runtime_error("Cannot load continuation arrival diagnostic");
    } else {
#pragma omp parallel for schedule(static)
      for (size_t j = 0; j < N; j++)
        if (a[j] <= 0)
          arrival[j] = Real(start_time);
    }
  }
  double initialization_seconds = seconds(start), update_seconds = 0,
         write_seconds = 0, probe_seconds = 0;
  size_t steps = 0;
  int frames = 0;
  std::ofstream fields;
  if (history) {
    fields.open(out + "/fields.f32", std::ios::binary);
    if (!fields)
      throw std::runtime_error("Cannot open fields");
  }
  std::vector<uint64_t> probe_indices(probe_count * 256);
  std::vector<double> probe_weights(probe_count * 256),
      probe_values(probe_count);
  std::ofstream probes;
  if (probe_count) {
    std::ifstream fi(arg(argc, argv, "probe-indices", ""), std::ios::binary),
        fw(arg(argc, argv, "probe-weights", ""), std::ios::binary);
    fi.read(reinterpret_cast<char *>(probe_indices.data()),
            probe_indices.size() * sizeof(uint64_t));
    fw.read(reinterpret_cast<char *>(probe_weights.data()),
            probe_weights.size() * sizeof(double));
    if (!fi || !fw || fi.peek() != EOF || fw.peek() != EOF)
      throw std::runtime_error("Bad probes");
    for (size_t q = 0; q < probe_count; q++) {
      double sum = 0;
      for (size_t k = 0; k < 256; k++) {
        size_t j = q * 256 + k;
        if (probe_indices[j] >= N || !std::isfinite(probe_weights[j]) ||
            probe_weights[j] < 0)
          throw std::runtime_error("Bad probe stencil");
        sum += probe_weights[j];
      }
      if (std::abs(sum - 1) > 1e-8)
        throw std::runtime_error("Bad probe weights");
    }
    probes.open(out + "/probe_fields.f64", std::ios::binary);
    if (!probes)
      throw std::runtime_error("Cannot open probe output");
  }
  auto save = [&]() {
    auto t = Clock::now();
    if (history) {
      fields.write(reinterpret_cast<const char *>(a.data()), N * sizeof(Real));
      if (!fields)
        throw std::runtime_error("Field write failed");
    }
    write_seconds += seconds(t);
    if (probe_count) {
      auto pt = Clock::now();
      for (size_t q = 0; q < probe_count; q++) {
        double v = 0;
        for (size_t k = 0; k < 256; k++) {
          size_t j = q * 256 + k;
          v += double(a[probe_indices[j]]) * probe_weights[j];
        }
        probe_values[q] = v;
      }
      probes.write(reinterpret_cast<const char *>(probe_values.data()),
                   probe_values.size() * sizeof(double));
      if (!probes)
        throw std::runtime_error("Probe write failed");
      probe_seconds += seconds(pt);
    }
    frames++;
  };
  // A resumed history references the parent's final frame instead of copying
  // it.
  if (start_time == 0)
    save();
  auto advance = [&](const Real *src, Real *dst, Real dt, const Real *base,
                     double tau) {
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < N / inner; block++) {
      const size_t begin = block * inner;
      auto q = decode(begin);
      const Real vp = pv[q[3]], palo = q[3] == 0 ? 0 : -ap,
                 pahi = q[3] == np - 1 ? 0 : ap;
      const Real fpx = vp * cs[q[2]], fpy = vp * sn[q[2]],
                 wp = std::abs(vp) * kp;
      const Real avoid =
          std::max(std::abs(x[q[0]]), std::abs(x[q[1]])) - L + 1e-6f;
      for (int eh = 0; eh < nh; eh++) {
        q[6] = eh;
        for (int iv = 0; iv < ne; iv++) {
          q[7] = iv;
          size_t j = begin + size_t(eh) * ne + iv;
          const Real ve = ev[iv], we = std::abs(ve) * ke,
                     ealo = iv == 0 ? 0 : -ae, eahi = iv == ne - 1 ? 0 : ae;
          const std::array<Real, 8> f = {fpx,         fpy,         0, 0,
                                         ve * cs[eh], ve * sn[eh], 0, 0};
          std::array<Real, 8> g;
          Real H = 0, diss = 0;
          for (int k = 0; k < 8; k++) {
            Real lm, rp;
            slopes(src, j, stride[k], q[k], shape[k], spacing[k],
                   k == 2 || k == 6,
                   linear_speed_boundary && (k == 3 || k == 7), order, lm, rp);
            g[k] = .5f * (lm + rp);
            const Real alpha = k == 2   ? wp
                               : k == 3 ? ap
                               : k == 6 ? we
                               : k == 7 ? ae
                                        : std::abs(f[k]);
            H += f[k] * g[k];
            diss += .5f * alpha * (rp - lm);
          }
          H += -wp * std::abs(g[2]) + we * std::abs(g[6]) +
               std::min(palo * g[3], pahi * g[3]) +
               std::max(ealo * g[7], eahi * g[7]);
          Real value = src[j] + dt * (H + diss);
          if (base)
            value = .5f * (base[j] + value);
          if (order == 1 || base) {
            Real previous = base ? base[j] : src[j];
            value = std::max(std::min(value, previous), avoid);
            if (previous > 0 && value <= 0 && std::isnan(arrival[j]))
              arrival[j] = Real(tau) + dt * previous / (previous - value);
          }
          dst[j] = value;
        }
      }
    }
  };
  double tau = start_time;
  for (int frame = 1; frame <= int(std::ceil((T - start_time) / outdt - 1e-10));
       frame++) {
    const double goal = std::min(T, start_time + frame * outdt);
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
    std::cout << std::setprecision(10) << "budget " << tau << " steps " << steps
              << " updates_s " << update_seconds << std::endl;
  }
  auto validation_start = Clock::now();
  size_t nonfinite = 0;
#pragma omp parallel for schedule(static) reduction(+ : nonfinite)
  for (size_t j = 0; j < N; j++)
    nonfinite += !std::isfinite(a[j]);
  const double validation_seconds = seconds(validation_start);
  if (nonfinite)
    throw std::runtime_error("Nonfinite values in the final signed field");
  auto t = Clock::now();
  for (auto item : std::array<std::pair<std::string, const Real *>, 2>{
           {{"final.f32", a.data()}, {"arrival.f32", arrival.data()}}}) {
    if ((item.first == "final.f32" && !write_final) ||
        (item.first == "arrival.f32" && !write_arrival))
      continue;
    std::ofstream f(out + "/" + item.first, std::ios::binary);
    f.write(reinterpret_cast<const char *>(item.second), N * 4);
    if (!f)
      throw std::runtime_error("Final write failed");
  }
  write_seconds += seconds(t);
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::ofstream meta(out + "/native_timing.json");
  meta << std::setprecision(12) << "{\"states\":" << N
       << ",\"internal_steps\":" << steps << ",\"frames\":" << frames
       << ",\"update_seconds\":" << update_seconds
       << ",\"initialization_seconds\":" << initialization_seconds
       << ",\"write_seconds\":" << write_seconds
       << ",\"validation_seconds\":" << validation_seconds
       << ",\"probe_seconds\":" << probe_seconds
       << ",\"process_seconds\":" << seconds(start)
       << ",\"core_array_bytes\":" << N * 12
       << ",\"peak_rss_bytes\":" << usage.ru_maxrss * 1024ULL
       << ",\"max_internal_dt\":" << maxdt
       << ",\"arithmetic\":\"float32 fields and local arithmetic\"}\n";
  return 0;
}
int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
}
