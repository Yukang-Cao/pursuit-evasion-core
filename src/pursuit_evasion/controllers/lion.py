"""A radial lion-guided target, followed by a vehicle tracker.

This is a declared vehicle adaptation of the geometric alignment idea. The
original alternating-move polygon theorem is not claimed for this controller.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class LionTarget:
    step_length: float = 1.0
    update_period: float = 0.25

    def reset(self, state):
        self.anchor = np.asarray(state[:2], dtype=float).copy()
        self.target = None
        self.next_update = -np.inf
        self.last = {}

    def target_at(self, state, time):
        if self.target is not None and time < self.next_update - 1e-10:
            return self.target.copy()
        p = np.asarray(state[:2])
        e = np.asarray(state[3:5])
        ray = e - self.anchor
        length = float(np.linalg.norm(ray))
        if length < 1e-12:
            goal = e.copy()
            mode = "evader_at_anchor"
            cross_error = 0.0
        else:
            d = ray / length
            projection = float(np.dot(p - self.anchor, d))
            cross_error = float(np.linalg.norm(p - (self.anchor + projection * d)))
            if np.linalg.norm(e - p) <= self.step_length:
                goal = e.copy()
                mode = "target_evader_position"
            elif cross_error <= self.step_length:
                radial = projection + np.sqrt(
                    max(0.0, self.step_length**2 - cross_error**2)
                )
                radial = np.clip(radial, 0.0, length)
                goal = self.anchor + radial * d
                mode = "radial_intersection"
            else:
                # Vehicle tracking no longer satisfies the ideal ray invariant.
                # Move the geometric target toward the nearest point on the ray.
                nearest = self.anchor + np.clip(projection, 0.0, length) * d
                delta = nearest - p
                goal = p + delta * min(
                    1.0, self.step_length / max(np.linalg.norm(delta), 1e-12)
                )
                mode = "alignment_recovery"
        self.target = goal
        self.next_update = time + self.update_period
        self.last = dict(
            anchor=self.anchor.tolist(),
            target=goal.tolist(),
            mode=mode,
            cross_track_to_anchor_ray=cross_error,
            target_updated_at=float(time),
        )
        return goal.copy()
