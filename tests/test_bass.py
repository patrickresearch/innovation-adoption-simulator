import numpy as np
import pytest
from scipy.integrate import solve_ivp

from adoptsim.models.bass import (
    BassParams,
    bass_cumulative,
    bass_rate,
    fit_bass,
    peak_time,
    takeoff_duration,
    time_to_fraction,
)


@pytest.mark.parametrize("n0", [0.0, 0.05, 0.4])
def test_closed_form_matches_ode(n0: float) -> None:
    p, q, m = 0.03, 0.38, 0.9
    t = np.linspace(0, 30, 61)
    sol = solve_ivp(
        lambda _t, n: (p + q * n / m) * (m - n), (0, 30), [n0], t_eval=t, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(bass_cumulative(t, p, q, m, 0.0, n0), sol.y[0], atol=1e-7)


def test_initial_condition_and_limit() -> None:
    y = bass_cumulative([5.0, 500.0], 0.01, 0.5, 2.0, t0=5.0, n0=0.3)
    assert y[0] == pytest.approx(0.3)
    assert y[1] == pytest.approx(2.0)


def test_rate_is_derivative() -> None:
    p, q, m = 0.02, 0.4, 1.0
    t = np.linspace(0.5, 20, 40)
    h = 1e-6
    num = (bass_cumulative(t + h, p, q, m) - bass_cumulative(t - h, p, q, m)) / (2 * h)
    np.testing.assert_allclose(bass_rate(t, p, q, m), num, rtol=1e-5)


def test_peak_time_maximises_rate() -> None:
    p, q = 0.01, 0.5
    t = np.linspace(0, 30, 30001)
    assert t[np.argmax(bass_rate(t, p, q, 1.0))] == pytest.approx(peak_time(p, q), abs=1e-3)


def test_time_to_fraction_inverts_closed_form() -> None:
    p, q, f0 = 0.02, 0.35, 0.05
    for frac in (0.1, 0.5, 0.9):
        tt = time_to_fraction(frac, p, q, f0)
        assert bass_cumulative(tt, p, q, 1.0, 0.0, f0) == pytest.approx(frac, abs=1e-10)


def test_takeoff_faster_with_more_imitation() -> None:
    assert takeoff_duration(0.01, 0.6) < takeoff_duration(0.01, 0.2)


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        BassParams(p=0.0, q=0.3, m=1.0)


@pytest.mark.parametrize(
    ("p", "q", "m"), [(0.03, 0.38, 0.8), (0.005, 0.6, 0.95), (0.05, 0.15, 0.6)]
)
def test_parameter_recovery(p: float, q: float, m: float) -> None:
    rng = np.random.default_rng(1)
    t = np.arange(1950, 1990, dtype=float)
    y = bass_cumulative(t - 1950 + 1.0, p, q, m) + rng.normal(0, 0.002, t.size)
    fit = fit_bass(t, y, m_bounds=(y.max(), 1.0))
    assert fit.r2 > 0.999
    assert fit.params.q == pytest.approx(q, rel=0.1)
    assert fit.params.m == pytest.approx(m, rel=0.05)


def _late_start_series(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    # series begins at F ~ 0.35: the data say little about p
    rng = np.random.default_rng(seed)
    t = np.arange(1960, 1990, dtype=float)
    y = bass_cumulative(t - 1960, 0.02, 0.3, 1.0, 0.0, 0.35) + rng.normal(0, 0.003, t.size)
    return t, np.clip(y, 0.0, 1.0)


def test_log_p_prior_pins_p_when_tight() -> None:
    t, y = _late_start_series()
    fit = fit_bass(t, y, m_bounds=(y.max() * 1.0001, 1.5), log_p_prior=(np.log(0.01), 0.01))
    assert fit.params.p == pytest.approx(0.01, rel=0.02)
    assert fit.r2 > 0.99


def test_log_p_prior_keeps_p_off_the_bound_without_hurting_fit() -> None:
    t = np.arange(1960, 1990, dtype=float)
    # p is irrelevant for this curve, so the plain fit has nothing to hold it
    y = bass_cumulative(t - 1960, 1e-7, 0.3, 1.0, 0.0, 0.35)
    bounds = (y.max(), 1.0)
    ml = fit_bass(t, y, m_bounds=bounds)
    reg = fit_bass(t, y, m_bounds=bounds, log_p_prior=(np.log(0.01), 1.5))
    assert ml.params.p < 1e-5
    assert reg.params.p > ml.params.p
    assert reg.rmse < 0.01  # shrinkage costs almost no fit quality


def test_fit_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        fit_bass([1, 2, 3], [0.1, 0.2, 0.3])
