"""Optional learned capture fields from refreshed P-centered observations."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..reversible.map_value import MapEncoder, bilinear_features
from ..reversible.footprint_map import Footprint
from .geometry import rectangle_separation
from .perception import observed_clearances


class ContactValueNet(nn.Module):
    """One state query predicts a monotone curve over remaining-time budgets.

    More available time cannot worsen a capture reachability value. The head
    predicts nonnegative decreases from the known zero-time contact geometry.
    This is a supervised numerical-value model, not an RL policy.
    """

    def __init__(self, use_map=True, extent=6.0, hidden=256, times=None, channels=16):
        super().__init__()
        self.use_map = use_map
        self.extent = float(extent)
        self.hidden = hidden
        times = np.arange(9) * 0.25 if times is None else np.asarray(times)
        self.register_buffer("times", torch.tensor(times, dtype=torch.float32))
        self.encoder = MapEncoder(channels) if use_map else None
        if use_map:
            self.encoder.a.net[0] = nn.Conv2d(2, 16, 3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(13 + (3 * channels if use_map else 0), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(times) - 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def encode(self, image):
        if not self.use_map:
            return None
        return self.encoder(torch.cat([image[:, :1] / self.extent, image[:, 1:2]], 1))

    def query(self, features, states, geometry):
        s = states
        x = torch.stack(
            [
                s[..., 0] / self.extent,
                s[..., 1] / self.extent,
                torch.sin(s[..., 2]),
                torch.cos(s[..., 2]),
                s[..., 3] / 2,
                s[..., 4] / self.extent,
                s[..., 5] / self.extent,
                torch.sin(s[..., 6]),
                torch.cos(s[..., 6]),
                s[..., 7],
                geometry[..., 0] / self.extent,
                geometry[..., 1] / self.extent,
                geometry[..., 2] / self.extent,
            ],
            -1,
        )
        if self.use_map:
            p = bilinear_features(features, s[..., :2], self.extent)
            e = bilinear_features(features, s[..., 4:6], self.extent)
            g = features.mean((-1, -2))[:, None].expand(-1, s.shape[1], -1)
            x = torch.cat([x, p, e, g], -1)
        decreases = F.softplus(self.head(x)) * torch.diff(self.times)
        initial = geometry[..., :1]
        return torch.cat([initial, initial - torch.cumsum(decreases, -1)], -1)

    def forward(self, image, states, geometry):
        return self.query(self.encode(image), states, geometry)


class ContactNeuralField:
    name = "learned local DP"

    def __init__(self, checkpoint, config, *, device="cuda"):
        saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        if saved["physics"] != config.to_dict():
            raise ValueError("Checkpoint/game physics mismatch")
        self.device = device
        self.config = config
        self.times = np.asarray(saved["times"])
        self.horizon = float(self.times[-1])
        self.exit_radius = float(saved["exit_radius"])
        self.p_body = Footprint(**saved["footprints"]["pursuer"])
        self.e_body = Footprint(**saved["footprints"]["evader"])
        self.model = (
            ContactValueNet(
                saved["use_map"],
                saved["extent"],
                saved["hidden"],
                saved["times"],
                channels=saved.get("channels", 16),
            )
            .to(device)
            .eval()
        )
        self.model.load_state_dict(saved["state_dict"])
        self.frame = None
        self.features = None
        self.image = None

    @torch.no_grad()
    def prepare(self, image, frame):
        self.image = np.asarray(image, np.float32)
        self.frame = frame
        self.features = self.model.encode(
            torch.tensor(self.image[None], device=self.device)
        )

    @torch.no_grad()
    def fields(self, states):
        if self.frame is None:
            raise RuntimeError("Prepare the current observation first")
        s = np.asarray(states, float)
        shape = s.shape[:-1]
        flat = s.reshape(-1, 8)
        local = self.frame.to_local(flat)
        phi = rectangle_separation(flat[:, :4], flat[:, 4:], self.p_body, self.e_body)
        pc, ec = observed_clearances(
            self.image, local, self.model.extent, self.p_body, self.e_body
        )
        geometry = np.stack([phi, pc, ec], -1).astype(np.float32)
        valid = (abs(local[:, [0, 1, 4, 5]]) <= self.model.extent).all(1) & (
            np.linalg.norm(flat[:, :2] - flat[:, 4:6], axis=1) <= self.exit_radius
        )
        outputs = []
        for start in range(0, len(flat), 8192):
            z = torch.tensor(
                local[None, start : start + 8192],
                dtype=torch.float32,
                device=self.device,
            )
            g = torch.tensor(geometry[None, start : start + 8192], device=self.device)
            outputs.append(self.model.query(self.features, z, g)[0].cpu().numpy())
        values = np.concatenate(outputs).T
        values[:, ~valid] = self.exit_radius
        values[0, valid] = phi[valid]
        return values, valid, shape

    def time_value(self, states):
        from ..reversible.map_game_field import first_crossing

        values, valid, shape = self.fields(states)
        return first_crossing(self.times, values, valid).reshape(shape)
