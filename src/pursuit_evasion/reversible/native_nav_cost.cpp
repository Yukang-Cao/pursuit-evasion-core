// Free-space implementation of the existing early-goal MPPI navigation cost.
// This kernel changes evaluation speed, not the objective or sampling/update.
#include <algorithm>
#include <cmath>

extern "C" void pe_free_costs(const double* pose, const double* controls,
                              const double* goal, double* costs, int count,
                              int steps, double dt, double vmin, double vmax,
                              double tolerance2, double weight, double terminal) {
    constexpr double pi = 3.1415926535897932384626433832795;
    for (int i = 0; i < count; ++i) {
        double x = pose[0], y = pose[1], heading = pose[2], speed = pose[3];
        double cost = 0, distance2 = 0;
        bool reached = false;
        for (int t = 0; t <= steps; ++t) {
            const double dx = x - goal[0], dy = y - goal[1];
            distance2 = dx * dx + dy * dy;
            if (distance2 <= tolerance2) {
                reached = true;
                break;
            }
            cost += weight * distance2;
            if (t == steps) break;

            const double acceleration = controls[(i * steps + t) * 2];
            const double curvature = controls[(i * steps + t) * 2 + 1];
            const double hit = std::abs(acceleration) > 1e-14
                ? ((acceleration >= 0 ? vmax : vmin) - speed) / acceleration
                : INFINITY;
            const double accelerating_time = std::min(dt, std::max(0., hit));
            const double next_speed = std::clamp(speed + acceleration * dt, vmin, vmax);
            const double distance = speed * accelerating_time
                + .5 * acceleration * accelerating_time * accelerating_time
                + next_speed * (dt - accelerating_time);
            const double turn = curvature * distance, half_turn = turn / 2;
            const double chord = distance * (std::abs(half_turn) < 1e-12
                ? 1. : std::sin(half_turn) / half_turn);
            x += chord * std::cos(heading + half_turn);
            y += chord * std::sin(heading + half_turn);
            heading = std::fmod(heading + turn + pi, 2 * pi);
            if (heading < 0) heading += 2 * pi;
            heading -= pi;
            speed = next_speed;
        }
        costs[i] = cost + (reached ? 0. : terminal * distance2);
    }
}
