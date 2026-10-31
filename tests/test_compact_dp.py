import numpy as np
import pytest
from pursuit_evasion.local_game.grid_game import (
    VehicleGrid,
    gpu_arrays,
    backup_gpu,
    initialize_gpu,
)
from pursuit_evasion.local_game.geometry import CircleArena, rectangle_separation


def fake_grid(n, na, rng):
    state = rng.uniform(-1, 1, (n, 4))
    ids = rng.integers(n, size=(n, na, 16), dtype=np.int32)
    weights = rng.random((n, na, 16)).astype(np.float32)
    weights /= weights.sum(-1, keepdims=True)
    allowed = rng.random((n, na)) > 0.3
    allowed[:, 0] = True
    return VehicleGrid(
        (),
        state,
        np.arange(n, dtype=np.int32),
        np.zeros((na, 2)),
        ids,
        weights,
        allowed,
        np.ones((n, na)),
    )


def brute_backup(f, p, e, exit_radius):
    out = f.copy()
    for i in range(len(p.states)):
        for j in range(len(e.states)):
            if np.linalg.norm(p.states[i, :2] - e.states[j, :2]) > exit_radius:
                continue
            values = []
            for a in np.flatnonzero(p.allowed[i]):
                response = []
                for b in np.flatnonzero(e.allowed[j]):
                    response.append(
                        np.sum(
                            f[np.ix_(p.successors[i, a], e.successors[j, b])]
                            * p.weights[i, a, :, None]
                            * e.weights[j, b, None, :]
                        )
                    )
                values.append(max(response))
            out[i, j] = min(out[i, j], min(values))
    return out


def test_factorized_backup_equals_joint_action_enumeration():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(114)
    p, e = fake_grid(5, 3, rng), fake_grid(7, 4, rng)
    a = gpu_arrays(p, e, chunk=2)
    arena = CircleArena([], 3)
    initialize_gpu(a, p, e, arena, exit_radius=1.2)
    expected = rectangle_separation(
        p.states[:, None], e.states[None], arena.pursuer, arena.evader
    )
    expected = np.where(
        np.linalg.norm(p.states[:, None, :2] - e.states[None, :, :2], axis=-1) > 1.2,
        1.2,
        expected,
    )
    np.testing.assert_allclose(
        cp.asnumpy(a["current"]).reshape(5, 7), expected, atol=5e-7
    )
    for _ in range(2):
        expected = brute_backup(expected, p, e, 1.2)
        backup_gpu(a, p, e)
        np.testing.assert_allclose(
            cp.asnumpy(a["current"]).reshape(5, 7), expected, atol=8e-7
        )


@pytest.mark.parametrize("tiled", [False, True])
def test_local_pair_packing_preserves_the_same_eight_state_game(tiled):
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(57)
    from pursuit_evasion.local_game.grid_game import (
        PairDomain,
        packed_gpu_arrays,
        initialize_packed,
        backup_packed,
    )

    p, e = fake_grid(5, 3, rng), fake_grid(7, 4, rng)
    domain = PairDomain.build(p, e, 1.2)
    a = packed_gpu_arrays(p, e, domain, chunk=2, tiled_output=tiled)
    arena = CircleArena([], 3)
    initialize_packed(a, p, e, domain, arena)
    expected = rectangle_separation(
        p.states[:, None], e.states[None], arena.pursuer, arena.evader
    )
    expected = np.where(
        np.linalg.norm(p.states[:, None, :2] - e.states[None, :, :2], axis=-1) > 1.2,
        1.2,
        expected,
    )

    def expanded():
        flat = cp.asnumpy(a["current"])
        out = np.full((5, 7), 1.2)
        for i in range(5):
            ids = domain.e_indices[
                domain.p_position[i], : domain.counts[domain.p_position[i]]
            ]
            out[i, ids] = flat[domain.offsets[i] : domain.offsets[i + 1]]
        return out

    np.testing.assert_allclose(expanded(), expected, atol=5e-7)
    for _ in range(2):
        expected = brute_backup(expected, p, e, 1.2)
        backup_packed(a, p, e, domain)
        np.testing.assert_allclose(expanded(), expected, atol=8e-7)


def test_initializer_handles_more_than_65535_vehicle_states():
    cp=pytest.importorskip('cupy')
    from pursuit_evasion.local_game.grid_game import PairDomain,packed_gpu_arrays,initialize_packed
    n=65536
    def identity_grid(count,x):
        states=np.zeros((count,4));states[:,0]=x
        ids=np.repeat(np.arange(count,dtype=np.int32)[:,None,None],16,axis=2)
        weights=np.zeros((count,1,16),np.float32);weights[:,:,0]=1
        return VehicleGrid((),states,np.arange(count),np.zeros((1,2)),ids,weights,np.ones((count,1),bool),np.ones((count,1)))
    p=identity_grid(n,0);e=identity_grid(1,1)
    d=PairDomain(np.zeros(n,np.int32),np.array([[0]],np.int32),np.array([[0]],np.int32),np.array([1],np.int32),np.arange(n+1,dtype=np.int64),3.)
    a=packed_gpu_arrays(p,e,d);initialize_packed(a,p,e,d,CircleArena([],3))
    np.testing.assert_allclose(cp.asnumpy(a['current']),.4,atol=1e-6)
