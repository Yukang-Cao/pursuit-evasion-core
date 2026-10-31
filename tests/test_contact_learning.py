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
