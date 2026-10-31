#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
struct Model {
    double low, high, a, k, L, W, dt;
};
static inline void step(const double *s, double a, double k, double t, const Model &m,
                        double *out) {
    double target = a >= 0 ? m.high : m.low;
    double hit = std::abs(a) > 1e-14 ? (target - s[3]) / a : INFINITY;
    double active = std::min(t, std::max(0., hit));
    double v = std::max(m.low, std::min(m.high, s[3] + a * t));
    double d = s[3] * active + .5 * a * active * active + v * (t - active),
           turn = k * d;
    double z = turn * .5, sc = std::abs(z) > 1e-12 ? std::sin(z) / z : 1.;
    out[0] = s[0] + d * sc * std::cos(s[2] + z);
    out[1] = s[1] + d * sc * std::sin(s[2] + z);
    out[2] = std::remainder(s[2] + turn, 2 * M_PI);
    out[3] = v;
}
static inline double clearance(const double *s, const double *obs, int nc, double half,
                               const Model &m) {
    double c = std::cos(s[2]), q = std::sin(s[2]);
    double ex = .5 * (m.L * std::abs(c) + m.W * std::abs(q)),
           ey = .5 * (m.L * std::abs(q) + m.W * std::abs(c));
    double result = std::min(half - std::abs(s[0]) - ex, half - std::abs(s[1]) - ey),
           rad = .5 * std::hypot(m.L, m.W);
    for (int j = 0; j < nc; j++) {
        double dx = obs[3 * j] - s[0], dy = obs[3 * j + 1] - s[1], r = obs[3 * j + 2];
        double bound = rad + r + std::max(0., result);
        if (dx * dx + dy * dy > bound * bound)
            continue;
        double x = std::abs(dx * c + dy * q) - m.L * .5,
               y = std::abs(-dx * q + dy * c) - m.W * .5;
        double value = std::hypot(std::max(0., x), std::max(0., y)) +
                       std::min(std::max(x, y), 0.) - r;
        result = std::min(result, value);
    }
    return result;
}
static bool interval(const double *s, double a, double k, double lo, double hi,
                     double cl, double cr, double lip, const double *obs, int nc,
                     double half, const Model &m, int depth) {
    if (std::min(cl, cr) <= 1e-7)
        return false;
    if (std::min(cl, cr) > lip * (hi - lo) * .5 + 1e-7)
        return true;
    if (depth >= 18 || hi - lo < 1e-8)
        return false;
    double mid = .5 * (lo + hi), z[4];
    step(s, a, k, mid, m, z);
    double cm = clearance(z, obs, nc, half, m);
    return interval(s, a, k, lo, mid, cl, cm, lip, obs, nc, half, m, depth + 1) &&
           interval(s, a, k, mid, hi, cm, cr, lip, obs, nc, half, m, depth + 1);
}
static bool motion_safe(const double *s, double a, double k, double t,
                        const double *obs, int nc, double half, const Model &m) {
    double out[4];
    step(s, a, k, t, m, out);
    double lip = std::max(std::abs(s[3]), std::abs(out[3])) *
                 (1 + .5 * std::hypot(m.L, m.W) * std::abs(k));
    return interval(s, a, k, 0, t, clearance(s, obs, nc, half, m),
                    clearance(out, obs, nc, half, m), lip, obs, nc, half, m, 0);
}
static bool backup_safe(const double *s, const double *obs, int nc, double half,
                        const Model &m) {
    if (clearance(s, obs, nc, half, m) <= 1e-7)
        return false;
    if (std::abs(s[3]) < 1e-10)
        return true;
    // Braking uses held commands at the declared control period. The final
    // command uses a=-v/dt, so it stops exactly at the next update instead of
    // reversing or assuming an unavailable mid-period command switch.
    double speed = std::abs(s[3]);
    int periods = std::max(1, (int)std::ceil(speed / (m.a * m.dt) - 1e-10));
    double full = (periods - 1) * m.dt, last = speed - m.a * full;
    double distance = speed * full - .5 * m.a * full * full + .5 * last * m.dt;
    // Collision depends only on the swept constant-curvature arc. This
    // surrogate deceleration traces exactly the same arc up to that distance.
    double t = 2 * distance / speed, a = -s[3] / t;
    for (int q = 0; q < 3; q++) {
        double k = q == 0 ? 0 : q == 1 ? -m.k : m.k;
        if (motion_safe(s, a, k, t, obs, nc, half, m))
            return true;
    }
    return false;
}
static Model read_model(const double *param) {
    return Model{param[0], param[1], param[2], param[3], param[4], param[5], param[6]};
}
extern "C" void batch_clearance(const double *states, long n, const double *obs, int nc,
                                double half, const double *param, double *out,
                                int threads) {
    Model m = read_model(param);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < n; i++)
        out[i] = clearance(states + 4 * i, obs, nc, half, m);
}
extern "C" void batch_backups(const double *states, long n, const double *obs, int nc,
                              double half, const double *param, uint8_t *out,
                              int threads) {
    Model m = read_model(param);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 128)
    for (long i = 0; i < n; i++)
        out[i] = backup_safe(states + 4 * i, obs, nc, half, m);
}
extern "C" void batch_actions(const double *states, long n, const double *controls,
                              int na, double dt, const double *obs, int nc, double half,
                              const double *param, uint8_t *ok, double *next,
                              int threads) {
    Model m = read_model(param);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 128)
    for (long i = 0; i < n; i++)
        for (int u = 0; u < na; u++) {
            const double *s = states + 4 * i;
            double a = controls[2 * u], k = controls[2 * u + 1];
            double *out = next + 4 * (i * na + u);
            step(s, a, k, dt, m, out);
            ok[i * na + u] = motion_safe(s, a, k, dt, obs, nc, half, m) &&
                             backup_safe(out, obs, nc, half, m);
        }
}
static inline double contact_value(const double *p, const double *e, const Model &pm,
                                   const Model &em) {
    double dx = e[0] - p[0], dy = e[1] - p[1], dc = e[2] - p[2],
           c = std::abs(std::cos(dc)), s = std::abs(std::sin(dc));
    double cp = std::cos(p[2]), sp = std::sin(p[2]), ce = std::cos(e[2]),
           se = std::sin(e[2]);
    double x = std::abs(dx * cp + dy * sp) - .5 * (pm.L + em.L * c + em.W * s),
           y = std::abs(-dx * sp + dy * cp) - .5 * (pm.W + em.L * s + em.W * c);
    double u = std::abs(dx * ce + dy * se) - .5 * (em.L + pm.L * c + pm.W * s),
           v = std::abs(-dx * se + dy * ce) - .5 * (em.W + pm.L * s + pm.W * c);
    return std::max(std::max(x, y), std::max(u, v));
}
static double contact_at(const double *s, const double *pu, const double *eu, double t,
                         const Model &pm, const Model &em) {
    double p[4], e[4];
    step(s, pu[0], pu[1], t, pm, p);
    step(s + 4, eu[0], eu[1], t, em, e);
    return contact_value(p, e, pm, em);
}
static double contact_search(const double *s, const double *pu, const double *eu,
                             double lo, double hi, double fl, double fr, double lip,
                             const Model &pm, const Model &em, int depth) {
    if (fl <= 1e-9)
        return lo;
    if (std::min(fl, fr) > lip * (hi - lo) * .5 + 1e-9)
        return INFINITY;
    if (depth >= 25 || hi - lo < 1e-9)
        return fr <= 1e-9 ? hi : INFINITY;
    double mid = (lo + hi) * .5, fm = contact_at(s, pu, eu, mid, pm, em);
    double left = contact_search(s, pu, eu, lo, mid, fl, fm, lip, pm, em, depth + 1);
    if (std::isfinite(left))
        return left;
    return contact_search(s, pu, eu, mid, hi, fm, fr, lip, pm, em, depth + 1);
}
extern "C" void pair_capture_times(const double *s, const double *pu, int np,
                                   const double *eu, int ne, double dt,
                                   const double *pp, const double *ep, double *out,
                                   int threads) {
    Model pm = read_model(pp), em = read_model(ep);
    double initial = contact_value(s, s + 4, pm, em),
           D = std::hypot(s[0] - s[4], s[1] - s[5]);
    double radius = .5 * (std::hypot(pm.L, pm.W) + std::hypot(em.L, em.W));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < np; i++)
        for (int j = 0; j < ne; j++) {
            double p[4], e[4];
            step(s, pu[2 * i], pu[2 * i + 1], dt, pm, p);
            step(s + 4, eu[2 * j], eu[2 * j + 1], dt, em, e);
            double vp = std::max(std::abs(s[3]), std::abs(p[3])),
                   ve = std::max(std::abs(s[7]), std::abs(e[3]));
            if (D > radius + (vp + ve) * dt) {
                out[i * ne + j] = INFINITY;
                continue;
            }
            double wp = vp * std::abs(pu[2 * i + 1]), we = ve * std::abs(eu[2 * j + 1]);
            double lip = vp + ve + (D + (vp + ve) * dt) * std::max(wp, we) +
                         .5 * std::max(pm.L + pm.W, em.L + em.W) * (wp + we);
            out[i * ne + j] =
                contact_search(s, pu + 2 * i, eu + 2 * j, 0, dt, initial,
                               contact_value(p, e, pm, em), lip, pm, em, 0);
        }
}
extern "C" void lidar_ranges(const double *pose, const double *obs, int nc, double half,
                             int beams, double range, double *out) {
    for (int i = 0; i < beams; i++) {
        double theta = pose[2] + 2 * M_PI * i / beams, dx = std::cos(theta),
               dy = std::sin(theta), best = range;
        if (dx > 1e-14)
            best = std::min(best, (half - pose[0]) / dx);
        if (dx < -1e-14)
            best = std::min(best, (-half - pose[0]) / dx);
        if (dy > 1e-14)
            best = std::min(best, (half - pose[1]) / dy);
        if (dy < -1e-14)
            best = std::min(best, (-half - pose[1]) / dy);
        for (int j = 0; j < nc; j++) {
            double x = obs[3 * j] - pose[0], y = obs[3 * j + 1] - pose[1],
                   along = x * dx + y * dy,
                   disc = obs[3 * j + 2] * obs[3 * j + 2] -
                          (x * x + y * y - along * along);
            if (disc >= 0) {
                double hit = along - std::sqrt(disc);
                if (hit >= 0)
                    best = std::min(best, hit);
            }
        }
        out[i] = std::max(0., best);
    }
}
static double interpolated_clearance(const double *a, const double *b, double t,
                                     double angle, const double *obs, int nc,
                                     double half, const Model &m) {
    double s[4] = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * angle,
                   0};
    return clearance(s, obs, nc, half, m);
}
static bool config_interval(const double *a, const double *b, double lo, double hi,
                            double cl, double cr, double angle, double lip,
                            const double *obs, int nc, double half, const Model &m,
                            int depth) {
    if (std::min(cl, cr) <= 1e-7)
        return false;
    if (std::min(cl, cr) > lip * (hi - lo) * .5 + 1e-7)
        return true;
    if (depth >= 20)
        return false;
    double mid = (lo + hi) * .5,
           cm = interpolated_clearance(a, b, mid, angle, obs, nc, half, m);
    return config_interval(a, b, lo, mid, cl, cm, angle, lip, obs, nc, half, m,
                           depth + 1) &&
           config_interval(a, b, mid, hi, cm, cr, angle, lip, obs, nc, half, m,
                           depth + 1);
}
extern "C" void configuration_connections(const double *from, const double *to, long n,
                                          const double *obs, int nc, double half,
                                          const double *param, uint8_t *out,
                                          int threads) {
    Model m = read_model(param);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < n; i++) {
        const double *a = from + 4 * i, *b = to + 4 * i;
        double angle = std::remainder(b[2] - a[2], 2 * M_PI);
        double lip = std::hypot(b[0] - a[0], b[1] - a[1]) +
                     .5 * std::hypot(m.L, m.W) * std::abs(angle);
        out[i] = config_interval(a, b, 0, 1, clearance(a, obs, nc, half, m),
                                 clearance(b, obs, nc, half, m), angle, lip, obs, nc,
                                 half, m, 0);
    }
}
extern "C" void matched_steps(const double *states, const double *controls, long n,
                              double dt, const double *obs, int nc, double half,
                              const double *param, uint8_t *safe, double *next,
                              int threads) {
    Model m = read_model(param);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < n; i++) {
        const double *s = states + 4 * i, *u = controls + 2 * i;
        double *out = next + 4 * i;
        step(s, u[0], u[1], dt, m, out);
        safe[i] = motion_safe(s, u[0], u[1], dt, obs, nc, half, m);
    }
}
