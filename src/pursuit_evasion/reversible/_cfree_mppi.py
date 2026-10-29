"""Small NumPy adaptation of C-Free-Uniform's MPPI update and navigation cost.

Copyright (c) 2025 Yukang Cao. MIT license; see C_FREE_UNIFORM_LICENSE.txt.

Origins (the original routines are retained as optional test references):
  controllers/mppi_pytorch_controller.py::_run_mppi_optimization
  controllers/torch_planner_base.py::_calculate_trajectory_costs

The softmax/raw-noise update and sum-of-distance + terminal-distance objective
are preserved. The one intentional event change is that a goal reached AFTER a
collision cannot erase that earlier collision. See docs/reversible-mppi.md.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NavigationCosts:
    costs: np.ndarray
    goal_reached: np.ndarray
    blocked: np.ndarray
    goal_step: np.ndarray
    blocked_step: np.ndarray


def navigation_costs(
    trajectories,
    target,
    blocked_at_step,
    *,
    goal_tolerance=0.15,
    distance_weight=1.0,
    terminal_weight=5.0,
    obstacle_penalty=2000.0,
):
    """Evaluate (time, sample, state) arrays using the upstream arrival mask.

    The local navigation task terminates at its first goal or blocked event.
    A blocked event wins a same-sample tie. These supplied planning events are
    distinct from exact physical capture/boundary events in the game simulator.
    """
    paths = np.asarray(trajectories, dtype=float)
    goal = np.asarray(target, dtype=float)
    blocked_at_step = np.asarray(blocked_at_step, dtype=bool)
    if paths.ndim != 3 or paths.shape[-1] < 2 or goal.shape != (2,):
        raise ValueError("Expected trajectories[time,sample,state] and target[2].")
    if blocked_at_step.shape != paths.shape[:2]:
        raise ValueError("Planning-event mask must match time and sample axes.")
    if not np.isfinite(paths).all() or not np.isfinite(goal).all():
        raise ValueError("Navigation inputs must be finite.")
    distance2 = np.sum((paths[..., :2] - goal) ** 2, axis=-1)
    reached_at_step = distance2 <= goal_tolerance**2

    # Cumulative masks, as in TorchPlannerBase. Collision precedence fixes the
    # original independent masks allowing a later goal to erase an earlier hit.
    blocked_before = np.maximum.accumulate(blocked_at_step, axis=0)
    goal_mask = np.maximum.accumulate(reached_at_step & ~blocked_before, axis=0)
    blocked_mask = np.maximum.accumulate(blocked_at_step & ~goal_mask, axis=0)
    terminated = goal_mask | blocked_mask
    running = np.where(terminated, 0.0, distance_weight * distance2)
    obstacle = np.where(blocked_mask, obstacle_penalty, 0.0)
    costs = np.sum(running + obstacle, axis=0)
    successful = goal_mask[-1]
    blocked = blocked_mask[-1]
    costs += np.where(~(successful | blocked), terminal_weight * distance2[-1], 0.0)
    sentinel = paths.shape[0]
    goal_step = np.where(successful, np.argmax(goal_mask, axis=0), sentinel)
    blocked_step = np.where(blocked, np.argmax(blocked_mask, axis=0), sentinel)
    return NavigationCosts(costs, successful, blocked, goal_step, blocked_step)


def raw_noise_update(nominal, noise, costs, lower, upper, temperature):
    """Upstream softmax/raw-noise nominal update, followed by action clipping.

    Shape convention is retained: nominal[time,control], noise[time,sample,control].
    Positive infinite cost denotes an invalid sample. At least one finite cost
    is required. This routine does not assert the averaged trajectory is valid.
    """
    nominal = np.asarray(nominal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    if noise.ndim != 3 or nominal.shape != (noise.shape[0], noise.shape[2]):
        raise ValueError("Nominal/noise shape mismatch.")
    if costs.shape != (noise.shape[1],) or np.isnan(costs).any():
        raise ValueError("Expected one non-NaN cost per sampled sequence.")
    finite = np.isfinite(costs)
    if not finite.any() or np.isneginf(costs).any():
        raise ValueError("At least one finite cost and no negative infinity required.")
    minimum = costs[finite].min()
    weights = np.exp(np.maximum(-(costs - minimum) / temperature, -745.0))
    weights[~finite] = 0.0
    weights /= weights.sum()
    updated = nominal + np.sum(weights[None, :, None] * noise, axis=1)
    return np.clip(updated, lower, upper), weights
