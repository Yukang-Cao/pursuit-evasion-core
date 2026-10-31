import numpy as np
from pursuit_evasion.reversible.planning_frame import PlanningFrame
from pursuit_evasion.reversible.model import ReversibleConfig, joint_step
from pursuit_evasion.reversible.footprint_map import FootprintMap


def test_both_cars_move_consistently_in_a_fixed_frame():
    c = ReversibleConfig()
    frame = PlanningFrame(1.2, -0.7, 0.4)
    s = np.array([1.2, -0.7, 0.4, 1.0, -0.3, 0.2, -1.0, 0.3])
    p = np.array([1.0, -0.7])
    e = np.array([-0.4, 1.2])
    local = frame.to_local(s)
    np.testing.assert_allclose(local[:3], [0, 0, 0], atol=1e-14)
    next_world = joint_step(s, p, e, 0.05, c)
    next_local = joint_step(local, p, e, 0.05, c)
    np.testing.assert_allclose(frame.to_local(next_world), next_local, atol=1e-14)
    assert np.linalg.norm(next_local[:2]) > 0  # Future P is not frozen at origin.
    np.testing.assert_allclose(frame.to_world(next_local), next_world, atol=1e-14)


def test_map_observation_transforms_geometry_too():
    m = FootprintMap(2.5, ((0.2, 0.4, -1, 1),))
    f = PlanningFrame(1.0, 0, 0)
    image = f.sdf_image(m, extent=2, size=5)
    # Image center is world(1,0), not world(0,0).
    assert image[2, 2] == np.float32(m.point_sdf([1.0, 0]))
    assert image[2, 2] != np.float32(m.point_sdf([0.0, 0]))
