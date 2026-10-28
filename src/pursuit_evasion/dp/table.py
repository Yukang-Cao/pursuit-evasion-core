"""Portable signed-front queries. No compiler, GPU, or solver package is imported.

Do not interpolate arrival times across capture-time jumps. Instead interpolate
the signed spatial field at each budget and locate its first zero crossing.
"""

from itertools import product
import json
from pathlib import Path

import numpy as np

from ..model import GameConfig
from .grid import GridSpec


class LocalTable:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if self.meta["status"] != "complete" or not self.meta["history_saved"]:
            raise ValueError(
                "A completed solve with retained signed fields is required."
            )
        if (
            self.meta.get("tube_projection")
            != "min over every internal update, then max with avoid"
        ):
            raise ValueError(
                "This query requires the declared monotone-by-budget field convention."
            )
        g = self.meta["grid"]
        self.grid = GridSpec(g["positions"], g["headings"], g["extent"])
        self.config = GameConfig.from_dict(self.meta["config"])
        self.times = np.array(self.meta["times"])
        self.fields = np.memmap(
            self.folder / "fields.f32",
            dtype=np.float32,
            mode="r",
            shape=(len(self.times), *self.grid.shape),
        )
        if self.fields.nbytes != (self.folder / "fields.f32").stat().st_size:
            raise ValueError("Field length does not match metadata.")
        self._flat = self.fields.reshape(len(self.times), -1)
        self._strides = np.cumprod((1,) + self.grid.shape[:0:-1])[::-1]
        self._bits = np.array(list(product([0, 1], repeat=6)), dtype=np.int64)

    def stencil(self, states):
        states = np.asarray(states, dtype=float)
        if states.shape[-1] != 6:
            raise ValueError("State order must be [xP,yP,thetaP,xE,yE,thetaE].")
        shape = states.shape[:-1]
        s = states.reshape(-1, 6)
        valid = np.isfinite(s).all(1)
        s = np.where(np.isfinite(s), s, 0.0)
        indices = []
        weights = []
        for k, (axis, n, dx) in enumerate(
            zip(self.grid.axes, self.grid.shape, self.grid.spacings)
        ):
            q = s[:, k]
            if k in (2, 5):
                f = ((q + np.pi) % (2 * np.pi)) / dx
                j = np.floor(f).astype(int) % n
                w = f - np.floor(f)
            else:
                valid &= (q >= axis[0] - 1e-10) & (q <= axis[-1] + 1e-10)
                f = (np.clip(q, axis[0], axis[-1]) - axis[0]) / dx
                j = np.minimum(np.floor(f).astype(int), n - 2)
                w = f - j
            indices.append(j)
            weights.append(w)
        ii = np.zeros((len(s), 64), np.int64)
        ww = np.ones((len(s), 64))
        for k, (idx, w, n) in enumerate(zip(indices, weights, self.grid.shape)):
            coord = idx[:, None] + self._bits[None, :, k]
            if k in (2, 5):
                coord %= n
            ii += coord * self._strides[k]
            ww *= np.where(self._bits[None, :, k], w[:, None], 1 - w[:, None])
        return ii, ww, valid, shape

    def field(self, states, budget):
        ii, ww, valid, shape = self.stencil(states)
        tau = np.broadcast_to(np.asarray(budget), shape).reshape(-1)
        valid &= np.isfinite(tau) & (tau >= 0) & (tau <= self.times[-1] + 1e-10)
        k = np.clip(
            np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2
        )
        ratio = (tau - self.times[k]) / (self.times[k + 1] - self.times[k])
        a = (self._flat[k[:, None], ii] * ww).sum(1)
        b = (self._flat[(k + 1)[:, None], ii] * ww).sum(1)
        return np.where(valid, a + (b - a) * ratio, np.nan).reshape(shape)

    def value(self, states):
        """First numerical success time; NaN means unresolved or invalid, not escape."""
        ii, ww, valid, shape = self.stencil(states)

        def sample(k):
            return (self._flat[np.asarray(k)[:, None], ii] * ww).sum(1)

        lo = np.zeros(len(ii), int)
        hi = np.full(len(ii), len(self.times) - 1, int)
        valid &= sample(hi) <= 0
        while np.any(hi - lo > 1):
            mid = (hi + lo) // 2
            inside = sample(mid) <= 0
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)
        a, b = sample(lo), sample(hi)
        fraction = np.divide(a, a - b, out=np.zeros_like(a), where=abs(a - b) > 1e-14)
        result = self.times[lo] + np.clip(fraction, 0, 1) * (
            self.times[hi] - self.times[lo]
        )
        s = np.asarray(states).reshape(-1, 6)
        c = self.config
        p_valid = np.max(abs(s[:, :2]), axis=1) < c.arena_half_width
        capture = np.linalg.norm(s[:, :2] - s[:, 3:5], axis=1) <= c.capture_radius
        e_fault = np.max(abs(s[:, 3:5]), axis=1) >= c.arena_half_width
        result = np.where(valid & p_valid, result, np.nan)
        result = np.where(p_valid & (capture | e_fault), 0.0, result)
        return result.reshape(shape)
