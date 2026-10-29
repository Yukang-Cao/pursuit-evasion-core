"""Map-independent signed-front queries for reversible relative-game fields."""

from itertools import product
import json
from pathlib import Path
import numpy as np

from .free_dp import physics_signature


class RelativeTable:
    state_dimension = 5

    def __init__(self, folder, config=None, *, max_budget=None):
        self.folder = Path(folder).resolve()
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if self.meta.get("schema") != "reversible_relative5d_v1":
            raise ValueError("Not a reversible relative5D table")
        if self.meta["status"] != "complete" or not self.meta["history_saved"]:
            raise ValueError("A completed solve with signed history is required")
        if self.meta["tube_projection"] != "min over every internal update":
            raise ValueError("Unsupported budget convention")
        source_times = np.asarray(self.meta["times"], dtype=float)
        self.times = source_times.copy()
        self.axes = tuple(np.asarray(a, dtype=float) for a in self.meta["axes"])
        self.shape = tuple(len(a) for a in self.axes)
        if len(self.axes) != 5 or min(self.shape) < 2:
            raise ValueError("Expected five nondegenerate coordinate axes")
        self.spacings = np.array([a[1] - a[0] for a in self.axes])
        if any(
            not np.isfinite(a).all() or not np.allclose(np.diff(a), h) or h <= 0
            for a, h in zip(self.axes, self.spacings)
        ):
            raise ValueError("Axes must be finite, increasing and uniformly spaced")
        if not np.isclose(self.axes[2][0], -np.pi) or not np.isclose(
            self.spacings[2] * self.shape[2], 2 * np.pi
        ):
            raise ValueError(
                "Heading axis must be periodic without a repeated endpoint"
            )
        if "shape" in self.meta and tuple(self.meta["shape"]) != self.shape:
            raise ValueError("Metadata shape differs from coordinate axes")
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Invalid output times")
        if max_budget is not None:
            matches = np.flatnonzero(
                np.isclose(source_times, max_budget, rtol=0, atol=1e-10)
            )
            if not len(matches) or matches[0] == 0:
                raise ValueError("max_budget must be a positive retained output budget")
            self.times = source_times[: matches[0] + 1].copy()
        expected = int(np.prod(self.shape)) * len(source_times) * 4
        if (self.folder / "fields.f32").stat().st_size != expected:
            raise ValueError("History byte length differs from metadata")
        self.fields = np.memmap(
            self.folder / "fields.f32",
            mode="r",
            dtype=np.float32,
            shape=(len(source_times), *self.shape),
        )
        self.fields = self.fields[: len(self.times)]
        self._flat = self.fields.reshape(len(self.times), -1)
        self._strides = np.cumprod((1,) + self.shape[:0:-1])[::-1]
        self._bits = np.array(list(product([0, 1], repeat=5)), dtype=np.int64)
        self.physics = self.meta["physics"]
        self.horizon = float(self.times[-1])
        if config is not None:
            self.assert_compatible(config)

    def assert_compatible(self, config):
        if physics_signature(config) != self.physics:
            raise ValueError(
                "The free-space table does not match the vehicle/capture model"
            )

    def stencil(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim < 1 or states.shape[-1] != 5:
            raise ValueError("Relative states must end in [x,y,psi,vP,vE]")
        shape = states.shape[:-1]
        values = states.reshape(-1, 5)[:, [3, 4, 2, 1, 0]]
        valid = np.isfinite(values).all(1)
        values = np.where(np.isfinite(values), values, 0)
        ii = np.zeros((len(values), 32), dtype=np.int64)
        ww = np.ones((len(values), 32))
        for k, (axis, step, n) in enumerate(zip(self.axes, self.spacings, self.shape)):
            q = values[:, k]
            if k == 2:
                f = ((q + np.pi) % (2 * np.pi)) / step
                j = np.floor(f).astype(np.int64) % n
                w = f - np.floor(f)
            else:
                valid &= (q >= axis[0] - 1e-10) & (q <= axis[-1] + 1e-10)
                f = (np.clip(q, axis[0], axis[-1]) - axis[0]) / step
                j = np.minimum(np.floor(f).astype(np.int64), n - 2)
                w = f - j
            coordinates = j[:, None] + self._bits[None, :, k]
            if k == 2:
                coordinates %= n
            ii += coordinates * self._strides[k]
            ww *= np.where(self._bits[None, :, k], w[:, None], 1 - w[:, None])
        return ii, ww, valid, shape

    def field(self, states, budget):
        ii, weights, valid, shape = self.stencil(states)
        tau = np.broadcast_to(np.asarray(budget, dtype=float), shape).reshape(-1)
        valid &= np.isfinite(tau) & (tau >= 0) & (tau <= self.horizon + 1e-10)
        tau = np.where(np.isfinite(tau), tau, 0)
        k = np.clip(
            np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2
        )
        fraction = (tau - self.times[k]) / (self.times[k + 1] - self.times[k])
        a = (self._flat[k[:, None], ii] * weights).sum(1)
        b = (self._flat[(k + 1)[:, None], ii] * weights).sum(1)
        return np.where(valid, a + fraction * (b - a), np.nan).reshape(shape)

    def value(self, states):
        """First crossing of interpolated signed fields, never interpolated arrivals."""
        ii, weights, valid, shape = self.stencil(states)

        def sample(k):
            return (self._flat[np.asarray(k)[:, None], ii] * weights).sum(1)

        lo = np.zeros(len(ii), dtype=int)
        hi = np.full(len(ii), len(self.times) - 1, dtype=int)
        valid &= sample(hi) <= 0
        while np.any(hi - lo > 1):
            mid = (lo + hi) // 2
            inside = sample(mid) <= 0
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)
        a, b = sample(lo), sample(hi)
        fraction = np.divide(a, a - b, out=np.zeros_like(a), where=abs(a - b) > 1e-14)
        out = self.times[lo] + np.clip(fraction, 0, 1) * (
            self.times[hi] - self.times[lo]
        )
        out = np.where(valid, out, np.nan)
        flat = np.asarray(states, dtype=float).reshape(-1, 5)
        capture = np.linalg.norm(flat[:, :2], axis=1) <= self.physics["capture_radius"]
        physics_valid = np.isfinite(flat).all(1)
        for k, key in [(3, "pursuer"), (4, "evader")]:
            physics_valid &= (flat[:, k] >= self.physics[key]["speed_min"] - 1e-10) & (
                flat[:, k] <= self.physics[key]["speed_max"] + 1e-10
            )
        return np.where(capture & physics_valid, 0, out).reshape(shape)

    def gradient(self, states, budget):
        """Finite differences of the signed field, in public [x,y,psi,vP,vE] order."""
        s = np.asarray(states, dtype=float)
        out = []
        for j, axis_k in enumerate([4, 3, 2, 0, 1]):
            h = self.spacings[axis_k]
            left = s.copy()
            right = s.copy()
            left[..., j] -= h
            right[..., j] += h
            if j != 2:
                left[..., j] = np.clip(
                    left[..., j], self.axes[axis_k][0], self.axes[axis_k][-1]
                )
                right[..., j] = np.clip(
                    right[..., j], self.axes[axis_k][0], self.axes[axis_k][-1]
                )
            width = right[..., j] - left[..., j]
            difference = self.field(right, budget) - self.field(left, budget)
            out.append(
                np.divide(
                    difference,
                    width,
                    out=np.full_like(difference, np.nan),
                    where=width > 0,
                )
            )
        return np.stack(out, axis=-1)

    def hamiltonian_controls(self, states, budget):
        """Pointwise P-min/E-max controls for the interpolated field gradient.

        This is a numerical feedback extraction, not a claim of finite-hold
        optimality or wall safety. `field <= 0` is the numerical tube membership.
        """
        s = np.asarray(states, dtype=float)
        g = self.gradient(s, budget)
        p, e = self.physics["pursuer"], self.physics["evader"]
        cp = s[..., 3] * (s[..., 1] * g[..., 0] - s[..., 0] * g[..., 1] - g[..., 2])
        ce = s[..., 4] * g[..., 2]
        ap = -p["acceleration"] * np.sign(g[..., 3])
        ae = e["acceleration"] * np.sign(g[..., 4])
        ap = np.where(s[..., 3] <= p["speed_min"] + 1e-10, np.maximum(ap, 0), ap)
        ap = np.where(s[..., 3] >= p["speed_max"] - 1e-10, np.minimum(ap, 0), ap)
        ae = np.where(s[..., 4] <= e["speed_min"] + 1e-10, np.maximum(ae, 0), ae)
        ae = np.where(s[..., 4] >= e["speed_max"] - 1e-10, np.minimum(ae, 0), ae)
        return (
            np.stack([ap, -p["max_curvature"] * np.sign(cp)], axis=-1),
            np.stack([ae, e["max_curvature"] * np.sign(ce)], axis=-1),
        )
