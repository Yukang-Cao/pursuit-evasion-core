import json
import subprocess
import sys

import numpy as np
import pytest

from pursuit_evasion.dp.options import validate_options
from pursuit_evasion.game import advance, predict_pairs
from pursuit_evasion.model import GameConfig


def test_existing_metadata_survives_invalid_configuration(tmp_path):
    out = tmp_path / "completed"
    out.mkdir()
    meta = out / "meta.json"
    original = '{"status":"complete","sentinel":"preserve-me"}\n'
    meta.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pursuit_evasion",
            "solve",
            "--config",
            str(tmp_path / "missing.json"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert meta.read_text() == original


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_controls_are_rejected(bad):
    c = GameConfig()
    state = [0, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        advance(state, bad, 0, c)
    with pytest.raises(ValueError):
        predict_pairs(state, [0], [bad], c)
    with pytest.raises(ValueError):
        predict_pairs(state, [0], [0], c, duration=bad)


@pytest.mark.parametrize("cfl", [0, -1, np.nan, np.inf, 1.1])
def test_invalid_cfl_cannot_start_an_integration_loop(cfl):
    with pytest.raises(ValueError):
        validate_options(2.0, 0.1, 2, cfl, 8)
