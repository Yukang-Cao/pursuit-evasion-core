"""Optional PyTorch map-conditioned game-value query model.

The map is encoded once in a fixed planning frame. BOTH queried car poses
remain in that frame. This is an 8-state query, not an 8-dimensional output grid.
The scalar target is a signed finite-budget game value, not capture seconds.
"""

import torch
from torch import nn
import torch.nn.functional as F


def bilinear_features(features, xy, extent):
    """B,C,H,W map + B,Q,2 world query -> B,Q,C, without map duplication.

    Ordinary gather/multiply operations also permit derivatives of query
    gradients with respect to network parameters (useful for future PDE losses).
    """
    b, c, h, w = features.shape
    gx = ((xy[..., 0] / extent + 1) * 0.5 * (w - 1)).clamp(0, w - 1)
    gy = ((xy[..., 1] / extent + 1) * 0.5 * (h - 1)).clamp(0, h - 1)
    ix = gx.floor().long().clamp(0, w - 2)
    iy = gy.floor().long().clamp(0, h - 2)
    wx = gx - ix
    wy = gy - iy
    flat = features.flatten(2)
    result = 0
    for dx, dy, weight in [
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ]:
        index = ((iy + dy) * w + ix + dx)[:, None].expand(-1, c, -1)
        result = result + flat.gather(2, index) * weight[:, None]
    return result.transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class MapEncoder(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.a = ConvBlock(1, 16)
        self.b = ConvBlock(16, 32)
        self.c = ConvBlock(32, 64)
        self.d = ConvBlock(64, 64)
        self.upc = ConvBlock(128, 32)
        self.upb = ConvBlock(64, 16)
        self.upa = ConvBlock(32, channels)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(F.avg_pool2d(c, 2))
        c2 = self.upc(
            torch.cat(
                [
                    F.interpolate(
                        d, size=c.shape[-2:], mode="bilinear", align_corners=True
                    ),
                    c,
                ],
                1,
            )
        )
        b2 = self.upb(
            torch.cat(
                [
                    F.interpolate(
                        c2, size=b.shape[-2:], mode="bilinear", align_corners=True
                    ),
                    b,
                ],
                1,
            )
        )
        return self.upa(
            torch.cat(
                [
                    F.interpolate(
                        b2, size=a.shape[-2:], mode="bilinear", align_corners=True
                    ),
                    a,
                ],
                1,
            )
        )


class MapValue(nn.Module):
    """CFU-style dense features + interpolation + MLP, with a scalar value head.

    Both comparison models receive current P/E footprint clearance and the
    exact initial payoff. Only use_map=True receives the surrounding map image.
    The no-image model is therefore a state-and-clearance baseline, not map-blind.
    """

    def __init__(self, use_map=True, extent=3.0, channels=16, hidden=256):
        super().__init__()
        self.use_map = use_map
        self.extent = float(extent)
        self.encoder = MapEncoder(channels) if use_map else None
        self.head = nn.Sequential(
            nn.Linear(14 + (3 * channels if use_map else 0), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, sdf):
        return self.encoder(sdf / self.extent) if self.use_map else None

    def query(self, features, states, times, geometry):
        """states B,Q,8: xP,yP,headingP,vP,xE,yE,headingE,vE.

        geometry B,Q,3: initial payoff, P clearance, E clearance.
        At time zero prediction equals the supplied initial payoff exactly.
        """
        s = states
        scalar = torch.stack(
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
                times / 2,
                geometry[..., 0] / self.extent,
                geometry[..., 1] / self.extent,
                geometry[..., 2] / self.extent,
            ],
            -1,
        )
        if self.use_map:
            fp = bilinear_features(features, s[..., :2], self.extent)
            fe = bilinear_features(features, s[..., 4:6], self.extent)
            global_map = features.mean((-2, -1))[:, None].expand(-1, s.shape[1], -1)
            scalar = torch.cat([scalar, fp, fe, global_map], -1)
        correction = self.head(scalar).squeeze(-1)
        return geometry[..., 0] + times * correction

    def forward(self, sdf, states, times, geometry):
        return self.query(self.encode(sdf), states, times, geometry)
