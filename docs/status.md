# Implementation status

Implemented: six-dimensional fixed-speed car model, explicit square boundary
fault convention, exact command arcs, first-event detection, portable signed-
front queries, and native/JAX/OptimizedDP numerical adapters. Small solves have
run; cross-solver/refinement validation is in progress.

In progress: larger matched solver checks and a local DP controller; baseline 1.

Planned: matched UCT baseline, behavior cloning and PPO. No learned controller
or Jetson real-time result is claimed yet.

