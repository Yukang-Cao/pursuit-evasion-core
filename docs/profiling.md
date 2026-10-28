# Profiling that separates preparation from deployment

Offline numerical records distinguish:

- dependency/setup cost (an engineering observation, not DP runtime);
- cold compilation and initialization;
- numerical kernel/integration work;
- host/device transfers and field writes;
- peak host RSS and reported accelerator allocations;
- shape, physical spacings, horizon, internal update count and precision.

Online records distinguish:

- table loading/mapping from first-touch and repeated query cost;
- query-only latency from the complete P controller call;
- P and E controller calls in a development simulator;
- median, p95, p99, maximum and count above the configured command period.

Always synchronize accelerator work before stopping its timer. Run final timing
comparisons serially without competing solver jobs. Label pilot timings obtained
under concurrent development load. Keep source versions and exact parameters.

## Robot constraint

The intended online target is Jetson Orin at at least 10 Hz. That is 100 ms for
the complete cycle; a 20 Hz cycle is 50 ms. State acquisition, table queries,
search, transfers and command publication all consume the budget. A fast laptop
kernel or an average below the deadline does not establish onboard performance.

Episode replays currently advance logical simulation time independently of
computation latency. Their metadata says so. Deadline-aware execution and the
exact Orin variant, RAM and power mode remain required deployment checks.

Dense table history is not automatically resident in RAM: the online reader
memory-maps it. This can incur cold page faults and storage I/O. Report both the
on-disk footprint and actual RSS/query latency; do not equate virtual address
space with resident memory or pretend storage access is free.

