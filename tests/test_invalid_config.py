"""Reject invalid physical and numerical inputs before simulation or allocation."""

import numpy as np
import pytest

from pursuit_evasion.model import GameConfig, Vehicle
from pursuit_evasion.dp.grid import GridSpec, output_times


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_vehicle_parameters_are_rejected(bad):
    for field in ("speed", "max_turn_rate"):
        parameters = {"speed": 1.0, "max_turn_rate": 1.0, field: bad}
        with pytest.raises(ValueError, match="finite and positive"):
            Vehicle(**parameters)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_game_parameters_are_rejected(bad):
    for field in ("arena_half_width", "capture_radius", "control_dt"):
        with pytest.raises(ValueError, match="finite and positive"):
            GameConfig(**{field: bad})


def test_nonfinite_nested_vehicle_is_rejected_when_loading_config():
    config = GameConfig().to_dict()
    config["evader"]["max_turn_rate"] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        GameConfig.from_dict(config)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_grid_extent_is_rejected(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        GridSpec(extent=bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 5.5, 5.0, True, 0, -1])
def test_invalid_grid_counts_are_rejected_before_array_allocation(bad):
    for field in ("positions", "headings"):
        with pytest.raises(ValueError, match="Grid counts must be integers"):
            GridSpec(**{field: bad})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_output_budgets_are_rejected(bad):
    for field in ("horizon", "output_dt"):
        arguments = {"horizon": 0.2, "output_dt": 0.1, field: bad}
        with pytest.raises(ValueError, match="finite and positive"):
            output_times(**arguments)


def test_valid_physical_config_and_small_grid_contract_are_preserved():
    config = GameConfig()
    assert GameConfig.from_dict(config.to_dict()) == config
    assert config.pursuer.minimum_turn_radius == 1.0
    grid = GridSpec(positions=np.int64(5), headings=np.int64(4), extent=3.5)
    assert grid.shape == (5, 5, 4, 5, 5, 4)
    assert grid.axes[0][[0, -1]].tolist() == [-3.5, 3.5]
    np.testing.assert_allclose(output_times(0.25, 0.1), [0, 0.1, 0.2, 0.25])
