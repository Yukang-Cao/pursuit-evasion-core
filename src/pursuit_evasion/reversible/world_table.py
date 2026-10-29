"""Queries of retained 8D signed fields; unknown states remain explicit."""

from itertools import product
import json
from pathlib import Path

import numpy as np

from .model import ReversibleConfig
from .world_dp import WorldGrid, PERIODIC_AXES


def interpolation_stencil(states, axes, periodic_axes):
    s = np.asarray(states, dtype=float)
    ndim = len(axes)
    if s.shape[-1] != ndim:
        raise ValueError(f"Expected final state dimension {ndim}.")
    batch = s.shape[:-1]
    flat = s.reshape(-1, ndim)
    valid = np.isfinite(flat).all(1)
    safe = np.where(np.isfinite(flat), flat, 0.0)
    shape = tuple(len(a) for a in axes)
    strides = np.cumprod((1,) + shape[:0:-1], dtype=np.int64)[::-1]
    bits = np.array(list(product([0, 1], repeat=ndim)), dtype=np.int64)
    indices = np.zeros((len(flat), len(bits)), dtype=np.int64)
    weights = np.ones((len(flat), len(bits)), dtype=float)
    for k, axis in enumerate(axes):
        n = len(axis)
        dx = float(axis[1] - axis[0])
        q = safe[:, k]
        if k in periodic_axes:
            f = ((q - axis[0]) % (dx * n)) / dx
            lower = np.floor(f).astype(np.int64) % n
            w = f - np.floor(f)
        else:
            valid &= (q >= axis[0] - 1e-10) & (q <= axis[-1] + 1e-10)
            f = (np.clip(q, axis[0], axis[-1]) - axis[0]) / dx
            lower = np.minimum(np.floor(f).astype(np.int64), n - 2)
            w = np.clip(f - lower, 0, 1)
        coord = lower[:, None] + bits[None, :, k]
        if k in periodic_axes:
            coord %= n
        indices += coord * strides[k]
        weights *= np.where(bits[None, :, k], w[:, None], 1 - w[:, None])
    return indices, weights, valid, batch


class WorldTable:
    state_dimension = 8

    def __init__(self, folder, *, allow_legacy_speed_boundary=False):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if self.meta.get("status") != "complete" or self.meta.get("backend") not in (
            "native_world8d",
            "optimized_dp_world8d",
        ):
            raise ValueError("A completed reversible world8D solve is required.")
        self.speed_boundary = self.meta.get("speed_boundary", "legacy_signed")
        if self.speed_boundary != "linear" and not allow_legacy_speed_boundary:
            raise ValueError(
                "This preliminary table used signed-distance extrapolation on speed axes. "
                "Use a corrected linear-speed-boundary table for control; explicit "
                "allow_legacy_speed_boundary=True is available for diagnostics only."
            )
        if (
            self.meta.get("tube_projection")
            != "min over completed numerical updates, then max with avoid"
        ):
            raise ValueError("Unsupported numerical field convention.")
        self.config = ReversibleConfig.from_dict(self.meta["config"])
        g = self.meta["grid"]
        self.grid = WorldGrid(
            **{
                k: g[k]
                for k in ("positions", "headings", "p_speeds", "e_speeds", "extent")
            }
        )
        self.axes = self.grid.axes(self.config)
        self.local_times = np.asarray(self.meta["times"], dtype=float)
        self.times = self.local_times.copy()
        self.horizon = float(self.meta["horizon"])
        self.fields = None
        self._segments = []
        self._owners = np.array([], dtype=int)
        self._rows = np.array([], dtype=int)
        self.has_history = bool(self.meta["history_saved"])
        if self.meta["history_saved"]:
            self.fields = self._map(
                "fields.f32", (len(self.local_times), self.grid.states)
            )
            self._segments = [self.fields]
            self._owners = np.zeros(len(self.local_times), dtype=int)
            self._rows = np.arange(len(self.local_times))
        if "parent_table" in self.meta and self.has_history:
            self.parent = WorldTable(
                (self.folder / self.meta["parent_table"]).resolve(),
                allow_legacy_speed_boundary=allow_legacy_speed_boundary,
            )
            if (
                not self.parent.has_history
                or self.parent.config != self.config
                or self.parent.grid != self.grid
            ):
                raise ValueError("The retained parent history/grid/model is required.")
            if self.local_times[0] <= self.parent.times[-1] or not np.isclose(
                self.meta["start_time"], self.parent.horizon
            ):
                raise ValueError(
                    "Continuation history must start after the parent boundary."
                )
            owner = len(self.parent._segments)
            self._segments = self.parent._segments + [self.fields]
            self._owners = np.concatenate(
                (self.parent._owners, np.full(len(self.local_times), owner, int))
            )
            self._rows = np.concatenate(
                (self.parent._rows, np.arange(len(self.local_times)))
            )
            self.times = np.concatenate((self.parent.times, self.local_times))
        if self.meta.get("final_source", "final.f32") == "history_last":
            if self.fields is None:
                raise ValueError("The final field is stored in the retained history.")
            self.final = self.fields[-1]
        else:
            self.final = self._map("final.f32", (self.grid.states,))

    def _map(self, name, shape):
        path = self.folder / name
        if path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * 4:
            raise ValueError(f"Field size mismatch: {name}")
        return np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    def _sample(self, frame_indices, stencil, weights):
        """Spatial samples from logical time frames, possibly across files."""
        k = np.broadcast_to(np.asarray(frame_indices, dtype=int), (len(stencil),))
        owners = self._owners[k]
        rows = self._rows[k]
        result = np.empty(len(stencil), dtype=float)
        for owner in np.unique(owners):
            mask = owners == owner
            result[mask] = (
                self._segments[owner][rows[mask, None], stencil[mask]] * weights[mask]
            ).sum(1)
        return result

    def field(self, states, budget=None):
        ii, ww, valid, batch = interpolation_stencil(states, self.axes, PERIODIC_AXES)
        if budget is None:
            result = (self.final[ii] * ww).sum(1)
            return np.where(valid, result, np.nan).reshape(batch)
        tau = np.broadcast_to(np.asarray(budget, dtype=float), batch).reshape(-1)
        valid &= np.isfinite(tau) & (tau >= 0) & (tau <= self.horizon + 1e-10)
        if not self.has_history:
            if np.any(valid & ~np.isclose(tau, self.horizon, atol=1e-10, rtol=0)):
                raise ValueError(
                    "Only the final field is retained; arbitrary budgets require history."
                )
            return np.where(valid, (self.final[ii] * ww).sum(1), np.nan).reshape(batch)
        k = np.clip(
            np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2
        )
        w = (tau - self.times[k]) / (self.times[k + 1] - self.times[k])
        lower = self._sample(k, ii, ww)
        upper = self._sample(k + 1, ii, ww)
        return np.where(valid, lower + (upper - lower) * w, np.nan).reshape(batch)

    def value(self, states):
        """First crossing of spatially interpolated signed histories.

        It is numerical P-success time, including E faults, not an unconditional
        physical capture guarantee. Arrival-node arrays are not interpolated.
        """
        if not self.has_history:
            raise ValueError(
                "Time-to-success queries require retained signed histories."
            )
        ii, ww, valid, batch = interpolation_stencil(states, self.axes, PERIODIC_AXES)

        def sample(k):
            return self._sample(k, ii, ww)

        lo = np.zeros(len(ii), int)
        hi = np.full(len(ii), len(self.times) - 1, int)
        valid &= sample(hi) <= 0
        while np.any(hi - lo > 1):
            mid = (lo + hi) // 2
            inside = sample(mid) <= 0
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)
        lower, upper = sample(lo), sample(hi)
        f = np.divide(
            lower,
            lower - upper,
            out=np.zeros_like(lower),
            where=abs(lower - upper) > 1e-14,
        )
        values = self.times[lo] + np.clip(f, 0, 1) * (self.times[hi] - self.times[lo])
        s = np.asarray(states, dtype=float).reshape(-1, 8)
        finite = np.isfinite(s).all(1)
        legal_speeds = (
            (s[:, 3] >= self.config.pursuer.speed_min)
            & (s[:, 3] <= self.config.pursuer.speed_max)
            & (s[:, 7] >= self.config.evader.speed_min)
            & (s[:, 7] <= self.config.evader.speed_max)
        )
        p_legal = np.max(abs(s[:, :2]), axis=1) < self.config.arena_half_width
        captured = (
            np.linalg.norm(s[:, :2] - s[:, 4:6], axis=1) <= self.config.capture_radius
        )
        e_fault = np.max(abs(s[:, 4:6]), axis=1) >= self.config.arena_half_width
        values = np.where(valid & p_legal, values, np.nan)
        values = np.where(
            finite & legal_speeds & p_legal & (captured | e_fault), 0.0, values
        )
        return values.reshape(batch)

    def contains(self, states, budget=None):
        field = self.field(states, budget)
        return np.isfinite(field) & (field <= 0)
