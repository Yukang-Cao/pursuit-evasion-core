import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pursuit_evasion.local_game.learning import ContactValueNet


def test_contact_curve_has_exact_initial_geometry_and_monotone_budgets():
    m = ContactValueNet(False, hidden=32)
    state = torch.randn(2, 5, 8)
    geometry = torch.randn(2, 5, 3)
    prediction = m.query(None, state, geometry)
    torch.testing.assert_close(prediction[..., 0], geometry[..., 0])
    assert torch.all(torch.diff(prediction, dim=-1) <= 0)
    prediction.sum().backward()
    assert m.head[-1].weight.grad.abs().sum() > 0


def test_known_mask_is_a_separate_input_channel():
    m = ContactValueNet(True, hidden=32)
    image = torch.zeros(1, 2, 32, 32)
    s = torch.zeros(1, 2, 8)
    g = torch.ones(1, 2, 3)
    features = m.encode(image)
    image[:, 1] = 1
    known_features = m.encode(image)
    assert not torch.allclose(features, known_features)
    assert m.query(features, s, g).shape == (1, 2, 9)


def test_checkpoint_query_needs_only_observation_and_joint_state(tmp_path):
    from dataclasses import replace
    from pursuit_evasion.reversible.model import ReversibleConfig
    from pursuit_evasion.reversible.planning_frame import PlanningFrame
    from pursuit_evasion.local_game.learning import ContactNeuralField

    c = replace(
        ReversibleConfig(),
        point_vehicles=False,
        capture_rule="rectangle_contact",
        boundary_rule="collision_constrained",
    )
    m = ContactValueNet(False, hidden=32)
    path = tmp_path / "field.pt"
    torch.save(
        dict(
            state_dict=m.state_dict(),
            use_map=False,
            hidden=32,
            channels=16,
            extent=6.0,
            times=m.times.tolist(),
            exit_radius=3.0,
            physics=c.to_dict(),
            footprints={
                "pursuer": {"length": 0.8, "width": 0.5},
                "evader": {"length": 0.4, "width": 0.25},
            },
        ),
        path,
    )
    source = ContactNeuralField(path, c, device="cpu")
    image = np.stack([np.full((32, 32), 5.0), np.ones((32, 32))]).astype(np.float32)
    source.prepare(image, PlanningFrame())
    s = np.array([0, 0, 0, 0, 0.5, 0, 0, 0], float)
    assert source.time_value(s) == 0
    s[4] = 4
    assert np.isnan(source.time_value(s))
