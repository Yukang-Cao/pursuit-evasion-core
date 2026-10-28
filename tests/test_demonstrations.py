import copy
import json

import numpy as np

from pursuit_evasion.controllers.hybrid import LionMPPIDP
from pursuit_evasion.controllers.mppi import MPPISettings
from pursuit_evasion.demonstrations import export_demonstrations
from pursuit_evasion.model import GameConfig


def test_current_and_legacy_approach_labels_export_identical_imitation_masks(tmp_path):
    config = GameConfig()
    state = [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    controller = LionMPPIDP(
        config,
        None,
        use_dp=False,
        mppi_settings=MPPISettings(samples=16, horizon=0.2),
    )
    controller.reset(state)
    command = controller.command(state, 0.0)
    episode = {
        "config": config.to_dict(),
        "initial": state,
        "case": "fixture",
        "seed": 0,
        "outcome": "timeout",
        "physical_capture": False,
        "trace": [
            {
                "state": state,
                "time": 0.0,
                "p_turn": command,
                "pursuer": controller.last,
            },
            {
                "state": state,
                "time": 0.05,
                "p_turn": 0.0,
                "pursuer": {"control_owner": "dp"},
            },
            {"state": state, "time": 0.1},
        ],
    }
    legacy = copy.deepcopy(episode)
    legacy["trace"][0]["pursuer"]["control_owner"] = "teacher"
    exports = []
    for name, record in (("current", episode), ("legacy", legacy)):
        path = tmp_path / f"{name}.json"
        # Controller diagnostics contain arrays; the episode writer uses the
        # same list conversion before saving them.
        path.write_text(json.dumps(record, default=lambda value: value.tolist()))
        output = tmp_path / name
        meta = export_demonstrations([path], output)
        assert meta["approach_rows"] == 1
        with np.load(output / "dataset.npz") as dataset:
            exports.append({key: dataset[key].copy() for key in dataset.files})

    np.testing.assert_array_equal(exports[0]["approach_mask"], [True, False])
    assert exports[0].keys() == exports[1].keys()
    for key in exports[0]:
        np.testing.assert_equal(exports[0][key], exports[1][key])
