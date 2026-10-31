"""Same contact/time query interface for the map-independent5D baseline."""

import numpy as np
from ..reversible.relative_table import RelativeTable
from ..reversible.model import relative_state
from .geometry import rectangle_separation


class ContactFreeValue:
    name = "free-space DP"

    def __init__(self, folder, config, p_body, e_body):
        self.table = RelativeTable(folder, config)
        self.times = self.table.times
        self.horizon = self.table.horizon
        meta = self.table.meta
        if meta.get("capture_rule") != "rectangle_contact":
            raise ValueError("A rectangle-contact free-space table is required")
        if meta["capture_footprints"] != {
            "pursuer": p_body.to_dict(),
            "evader": e_body.to_dict(),
        }:
            raise ValueError("Footprint mismatch")
        self.p_body = p_body
        self.e_body = e_body
        self.exit_radius = meta["local_exit_radius"]

    def fields(self, states):
        s = np.asarray(states, float)
        shape = s.shape[:-1]
        z = relative_state(s)
        ids, weights, valid, _ = self.table.stencil(z)
        values = (self.table._flat[:, ids] * weights[None]).sum(-1)
        phi = rectangle_separation(
            s[..., :4], s[..., 4:], self.p_body, self.e_body
        ).reshape(-1)
        values = np.minimum(values, phi[None])
        inside = np.linalg.norm(z[..., :2], axis=-1).reshape(-1) <= self.exit_radius
        values[:, ~inside] = self.exit_radius
        values[:, ~valid] = np.nan
        values[0, valid & inside] = phi[valid & inside]
        return values, valid & inside, shape

    def time_value(self, states):
        from ..reversible.map_game_field import first_crossing

        values, valid, shape = self.fields(states)
        return first_crossing(self.times, values, valid).reshape(shape)
