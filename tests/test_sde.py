import numpy as np
import pytest

from adoptsim.models.bass import bass_cumulative
from adoptsim.models.sde import fan_quantiles, simulate_bass_sde


def test_zero_noise_converges_to_ode() -> None:
    p, q = 0.03, 0.4
    t, F = simulate_bass_sde(p, q, 0.0, 20.0, dt=1e-3, n_paths=1, seed=0)
    np.testing.assert_allclose(F[0], bass_cumulative(t, p, q, 1.0), atol=2e-3)


def test_paths_stay_in_unit_interval() -> None:
    _, F = simulate_bass_sde(0.02, 0.5, 0.8, 30.0, dt=0.01, n_paths=300, seed=1)
    assert F.min() >= 0.0
    assert F.max() <= 1.0


def test_small_noise_mean_close_to_ode() -> None:
    p, q = 0.02, 0.4
    t, F = simulate_bass_sde(p, q, 0.05, 25.0, dt=0.01, n_paths=2000, seed=2)
    assert np.max(np.abs(F.mean(axis=0) - bass_cumulative(t, p, q, 1.0))) < 0.01


def test_fan_quantiles_ordered() -> None:
    _, F = simulate_bass_sde(0.02, 0.4, 0.3, 20.0, n_paths=500, seed=3)
    fq = fan_quantiles(F)
    assert np.all(fq[0.1] <= fq[0.5]) and np.all(fq[0.5] <= fq[0.9])


def test_effort_accelerates_adoption() -> None:
    kw = {"dt": 0.01, "n_paths": 1, "seed": 0}
    _, base = simulate_bass_sde(0.02, 0.3, 0.0, 10.0, **kw)
    _, push = simulate_bass_sde(0.02, 0.3, 0.0, 10.0, effort=lambda t: 1.5 + 0 * t, **kw)
    assert push[0, -1] > base[0, -1]


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        simulate_bass_sde(0.0, 0.3, 0.1, 10.0)
