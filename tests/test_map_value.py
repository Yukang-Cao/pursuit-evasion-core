import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pursuit_evasion.reversible.map_value import MapValue, bilinear_features


def test_bilinear_world_coordinates_and_parameter_derivatives():
    xy = torch.tensor(
        [[[0.2, 0.4], [-0.7, 0.8]]], dtype=torch.float64, requires_grad=True
    )
    y, x = torch.meshgrid(
        torch.linspace(-2, 2, 7, dtype=torch.float64),
        torch.linspace(-2, 2, 9, dtype=torch.float64),
        indexing="ij",
    )
    image = (x + 2 * y)[None, None].clone().requires_grad_()
    result = bilinear_features(image, xy, 2).squeeze(-1)
    torch.testing.assert_close(result, xy[..., 0] + 2 * xy[..., 1])
    grad = torch.autograd.grad(result.sum(), xy, create_graph=True)[0]
    torch.testing.assert_close(
        grad, torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float64)
    )
    grad.square().sum().backward()
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize("with_map", [False, True])
def test_value_model_uses_two_pose_queries_and_exact_time_zero(with_map):
    m = MapValue(use_map=with_map)
    images = torch.randn(2, 1, 32, 32)
    states = torch.randn(2, 5, 8)
    geometry = torch.randn(2, 5, 3)
    times = torch.zeros(2, 5)
    torch.testing.assert_close(m(images, states, times, geometry), geometry[..., 0])
    times.fill_(1)
    y = m(images, states, times, geometry)
    assert y.shape == (2, 5)
    (y - geometry[..., 0] + 1).square().mean().backward()
    assert m.head[-1].weight.grad.abs().sum() > 0


def test_neural_query_needs_no_table_and_checks_model_contract(tmp_path):
    from dataclasses import replace
    from pursuit_evasion.reversible.model import ReversibleConfig
    from pursuit_evasion.reversible.footprint_map import FootprintMap
    from pursuit_evasion.reversible.neural_map_field import NeuralMapField

    config = replace(ReversibleConfig(), capture_radius=0.5, point_vehicles=False)
    geometry = FootprintMap(2.5)
    model = MapValue(False, hidden=16)
    checkpoint = tmp_path / "small.pt"
    torch.save(
        dict(
            state_dict=model.state_dict(),
            use_map=False,
            hidden=16,
            max_budget=2,
            physics=config.to_dict(),
            footprints={r: geometry.to_dict()[r] for r in ["pursuer", "evader"]},
        ),
        checkpoint,
    )
    source = NeuralMapField(checkpoint, config, geometry)
    state = np.array([0, 0, 0, 0, 1, 0, 0, 0], float)
    assert source.field(state, 0) == pytest.approx(0.5)
    assert np.isnan(
        source.time_value(state)
    )  # Untrained zero correction does not invent success.
    state[0] = 4
    assert np.isnan(source.field(state, 1))
    with pytest.raises(ValueError, match="physics mismatch"):
        NeuralMapField(checkpoint, replace(config, capture_radius=0.3), geometry)
