"""Read-only query of a retained finite-budget footprint-game field."""

from pathlib import Path
import json
import numpy as np
from .model import ReversibleConfig
from .world_dp import WorldGrid
from .world_table import interpolation_stencil
from .footprint_map import FootprintMap


class MapGameField:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if (
            self.meta.get("schema") != "footprint_map_game8d_v1"
            or self.meta["status"] != "complete"
        ):
            raise ValueError("A completed footprint-map reference is required")
        self.config = ReversibleConfig.from_dict(self.meta["config"])
        self.geometry = FootprintMap.from_dict(self.meta["geometry"])
        d = self.meta["grid"]
        self.grid = WorldGrid(
            **{
                k: d[k]
                for k in ["positions", "headings", "p_speeds", "e_speeds", "extent"]
            }
        )
        self.axes = self.grid.axes(self.config)
        self.horizon = float(self.meta["times"][-1])
        self.history = None
        if self.meta.get("history_saved"):
            p = self.folder / "fields.f32"
            nt = len(self.meta["times"])
            if p.stat().st_size != 4 * self.grid.states * nt:
                raise ValueError("Incomplete signed field history")
            self.history = np.memmap(
                p, np.float32, mode="r", shape=(nt, *self.grid.shape)
            )
            self.times = np.asarray(self.meta["times"])
        p = self.folder / "final.f32"
        if p.exists() and p.stat().st_size == 4 * self.grid.states:
            self.values = np.memmap(p, np.float32, mode="r", shape=self.grid.shape)
        elif self.history is not None:
            self.values = self.history[-1]
        else:
            raise ValueError("No complete final field retained")

    def field(self, states, budget=None):
        ii, weights, valid, shape = interpolation_stencil(states, self.axes, (2, 6))
        if budget is None or np.isclose(budget, self.horizon):
            out = (self.values.reshape(-1)[ii] * weights).sum(axis=-1)
        else:
            if self.history is None or not 0 <= budget <= self.horizon:
                raise ValueError("Requested budget is not available")
            k = min(
                np.searchsorted(self.times, budget, side="right") - 1,
                len(self.times) - 2,
            )
            w = (budget - self.times[k]) / (self.times[k + 1] - self.times[k])
            flat = self.history.reshape(len(self.times), -1)
            out = ((1 - w) * flat[k, ii] * weights + w * flat[k + 1, ii] * weights).sum(
                axis=-1
            )
        return np.where(valid, out, np.nan).reshape(shape)

    def time_value(self, states):
        """First crossing of spatially interpolated fields; unresolved is NaN."""
        if self.history is None:
            raise ValueError("A signed history is required for time queries")
        ii, weights, valid, shape = interpolation_stencil(states, self.axes, (2, 6))
        flat = self.history.reshape(len(self.times), -1)
        values = (flat[:, ii] * weights[None]).sum(axis=-1)
        # The zero-budget target/avoid geometry is known exactly at the query.
        # Do not let spatial interpolation invent an instantaneous win.
        values[0] = self.geometry.surfaces(
            np.asarray(states).reshape(-1, 8), self.config.capture_radius
        )[0]
        return first_crossing(self.times, values, valid).reshape(shape)


def first_crossing(times, values, valid=None):
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if values.ndim != 2 or len(times) != len(values) or np.any(np.diff(times) <= 0):
        raise ValueError("Expected increasing budgets and fields[budget,query]")
    mask = np.isfinite(values) & (values <= 0)
    found = mask.any(axis=0)
    j = mask.argmax(axis=0)
    before = np.maximum(j - 1, 0)
    cols = np.arange(values.shape[1])
    a, b = values[before, cols], values[j, cols]
    frac = np.divide(a, a - b, out=np.zeros_like(a), where=abs(a - b) > 1e-12)
    out = times[before] + np.clip(frac, 0, 1) * (times[j] - times[before])
    if valid is not None:
        found &= np.asarray(valid, bool)
    return np.where(found, out, np.nan)
