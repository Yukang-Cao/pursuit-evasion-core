"""Optional learned field queries from a map, with no numerical table dependency."""

from pathlib import Path
import numpy as np
from .planning_frame import PlanningFrame
from .map_game_field import first_crossing


class NeuralMapField:
    """Cache one map encoding in a fixed frame; query both future vehicle states.

    Physics and footprint sizes must match the checkpoint. This pilot assumes a
    fully known map; cropping unknown exterior is not a validated game boundary.
    Importing this module does not import Torch until a model is instantiated.
    """

    def __init__(self, checkpoint, config, geometry, *, device="cpu", frame=None):
        import torch
        from .map_value import MapValue

        self.torch = torch
        self.device = device
        self.config = config
        self.geometry = geometry
        self.frame = PlanningFrame() if frame is None else frame
        saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        if config.point_vehicles:
            raise ValueError(
                "Neural footprint-game queries require point_vehicles=False"
            )
        for key in ["pursuer", "evader", "capture_radius", "control_dt"]:
            if config.to_dict()[key] != saved["physics"][key]:
                raise ValueError("Checkpoint physics mismatch")
        for role in ["pursuer", "evader"]:
            if geometry.to_dict()[role] != saved["footprints"][role]:
                raise ValueError("Checkpoint footprint mismatch")
        self.horizon = float(saved["max_budget"])
        self.times = np.linspace(0, self.horizon, 5)
        self.model = (
            MapValue(
                saved["use_map"], extent=saved.get("extent", 3), hidden=saved["hidden"]
            )
            .to(device)
            .eval()
        )
        self.model.load_state_dict(saved["state_dict"])
        image = self.frame.sdf_image(
            geometry, self.model.extent, saved.get("image_size", 64)
        )
        with torch.no_grad():
            self.features = self.model.encode(
                torch.tensor(image[None, None], device=device)
            )

    def field(self, states, budget=None):
        torch = self.torch
        budget = self.horizon if budget is None else float(budget)
        if not np.isfinite(budget) or not 0 <= budget <= self.horizon:
            raise ValueError("Budget is outside the trained range")
        states = np.asarray(states, float)
        local = self.frame.to_local(states)
        shape = states.shape[:-1]
        flat = states.reshape(-1, 8)
        local = local.reshape(-1, 8)
        valid = (abs(local[:, [0, 1, 4, 5]]) <= self.model.extent).all(axis=1)
        for k, bounds in [(3, self.config.pursuer), (7, self.config.evader)]:
            valid &= (flat[:, k] >= bounds.speed_min) & (flat[:, k] <= bounds.speed_max)
        phi, _ = self.geometry.surfaces(flat, self.config.capture_radius)
        geo = np.stack(
            [
                phi,
                self.geometry.clearance(flat[:, :4]),
                self.geometry.clearance(flat[:, 4:], "evader"),
            ],
            -1,
        ).astype(np.float32)
        result = np.full(len(flat), np.nan)
        ids = np.flatnonzero(valid)
        with torch.no_grad():
            for begin in range(0, len(ids), 8192):
                q = ids[begin : begin + 8192]
                s = torch.tensor(
                    local[None, q], dtype=torch.float32, device=self.device
                )
                g = torch.tensor(geo[None, q], device=self.device)
                t = torch.full(
                    s.shape[:2], budget, dtype=torch.float32, device=self.device
                )
                result[q] = self.model.query(self.features, s, t, g)[0].cpu().numpy()
        return result.reshape(shape)

    def time_value(self, states):
        """Estimated time to P success, including the declared E-collision rule."""
        shape = np.asarray(states).shape[:-1]
        fields = np.stack([self.field(states, t).reshape(-1) for t in self.times])
        return first_crossing(self.times, fields).reshape(shape)
