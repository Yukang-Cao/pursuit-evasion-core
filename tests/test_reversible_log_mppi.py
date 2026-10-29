import numpy as np
import pytest
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.mppi import MPPISettings, ReversibleMPPI


def test_gaussian_rng_unchanged_and_log_noise_is_signed_heavy_tailed():
    config = ReversibleConfig()
    seed = 921
    regular = ReversibleMPPI(config, MPPISettings(), seed)
    shape = (200000, 2)
    expected = np.random.default_rng(seed).normal(size=shape) * [1, 0.5]
    np.testing.assert_array_equal(regular.sample_noise(shape), expected)
    log = ReversibleMPPI(
        config, MPPISettings(sampler="log_mppi", noise_std=(2, 1)), seed
    )
    noise = log.sample_noise(shape)
    assert np.isfinite(noise).all()
    assert np.all(np.abs((noise > 0).mean(axis=0) - 0.5) < 0.01)
    assert np.all((np.abs(noise) > [2, 1]).mean(axis=0) > 0.3)
    assert np.all(
        (np.abs(noise) > [2, 1]).mean(axis=0)
        > (np.abs(expected) > [2, 1]).mean(axis=0) * 4
    )
    assert np.all(np.std(noise, axis=0) > [4, 2])


def test_log_sampler_still_emits_admissible_command():
    c = ReversibleConfig()
    tracker = ReversibleMPPI(
        c,
        MPPISettings(sampler="log_mppi", noise_std=(2, 1), samples=64, horizon=0.2),
        0,
    )
    u = tracker.command([0, 0, 0, 0], [1, 0])
    assert np.all(np.isfinite(u)) and abs(u[0]) <= 2 and abs(u[1]) <= 1
    assert tracker.last["sampler"] == "log_mppi"
    with pytest.raises(ValueError):
        ReversibleMPPI(c, MPPISettings(sampler="unknown"))


def test_b1_defaults_to_log_workaround_but_gaussian_is_explicitly_reproducible():
    from pursuit_evasion.reversible.controllers import Baseline1

    c = ReversibleConfig()
    b = Baseline1(c, mode="approach")
    assert b.tracker.settings.sampler == "log_mppi"
    np.testing.assert_array_equal(b.tracker.sigma, [2, 1])
    old = Baseline1(c, mode="approach", mppi_settings=MPPISettings())
    assert old.tracker.settings.sampler == "gaussian"
    np.testing.assert_array_equal(old.tracker.sigma, [1, 0.5])
