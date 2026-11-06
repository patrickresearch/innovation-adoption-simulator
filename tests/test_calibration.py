import numpy as np
import pandas as pd
import pytest
from scipy import stats

from adoptsim.evaluation import (
    crps_ensemble,
    increment_sigma2,
    interval_covered,
    loo_forecast_scores,
    pit_histogram,
    randomized_pit,
    summarize_scores,
)
from adoptsim.models.sde import simulate_bass_sde


def test_crps_degenerate_ensemble_is_absolute_error() -> None:
    samples = np.full((50, 3), 0.4)
    got = crps_ensemble(samples, np.array([0.1, 0.4, 0.9]))
    np.testing.assert_allclose(got, [0.3, 0.0, 0.5], atol=1e-12)


def test_crps_matches_gaussian_closed_form() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40000, 1))
    # CRPS(N(0,1), y=0) = (sqrt(2) - 1) / sqrt(pi)
    got = crps_ensemble(x, np.array([0.0]))[0]
    assert got == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi), abs=0.01)


def test_crps_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = rng.normal(size=4)
    pair = np.abs(x[:, None, :] - x[None, :, :]).mean(axis=(0, 1))
    expected = np.abs(x - y).mean(axis=0) - 0.5 * pair
    np.testing.assert_allclose(crps_ensemble(x, y), expected, atol=1e-12)


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9])
def test_coverage_of_calibrated_forecast_is_nominal(level: float) -> None:
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(2000, 3000))
    y = rng.normal(size=3000)
    assert interval_covered(samples, y, level).mean() == pytest.approx(level, abs=0.03)


def test_pit_uniform_for_calibrated_and_u_shaped_for_overconfident() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=3000)
    ok = randomized_pit(rng.normal(size=(1000, 3000)), y, rng)
    assert stats.kstest(ok, "uniform").pvalue > 0.01
    narrow = randomized_pit(0.3 * rng.normal(size=(1000, 3000)), y, rng)
    h = pit_histogram(narrow)
    assert h[0] + h[-1] > 0.3  # mass piles up in the tails when the ensemble is too narrow


def test_pit_randomises_ties() -> None:
    rng = np.random.default_rng(4)
    pit = randomized_pit(np.ones((100, 500)), np.ones(500), rng)
    assert 0.4 < pit.mean() < 0.6 and pit.min() >= 0.0 and pit.max() <= 1.0


def test_sde_accepts_per_path_parameters() -> None:
    n = 200
    p = np.full(n, 0.03)
    q = np.full(n, 0.4)
    _, a = simulate_bass_sde(p, q, 0.1, 10.0, n_paths=n, seed=5)
    _, b = simulate_bass_sde(0.03, 0.4, 0.1, 10.0, n_paths=n, seed=5)
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="length n_paths"):
        simulate_bass_sde(p[:10], q, 0.1, 10.0, n_paths=n)
    with pytest.raises(ValueError, match="p > 0"):
        simulate_bass_sde(np.zeros(n), q, 0.1, 10.0, n_paths=n)


def test_sigma_moment_estimator_recovers_planted_noise() -> None:
    p, q, sigma, dt = 0.02, 0.5, 0.15, 0.01
    t, F = simulate_bass_sde(p, q, sigma, 25.0, dt=dt, n_paths=300, seed=6)
    step = 25  # observe every 0.25 time units
    s2 = [increment_sigma2(t[::step], f[::step], p, q) for f in F]
    assert np.sqrt(np.nanmean(s2)) == pytest.approx(sigma, rel=0.2)


def _planted_dataset(n: int, sigma: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curves from a well-specified model: one Gaussian over (log p, log q), SDE noise."""
    rng = np.random.default_rng(seed)
    mu, sd = np.array([np.log(0.02), np.log(0.4)]), np.array([0.5, 0.3])
    rows, fit_rows = [], []
    for i in range(n):
        lp, lq = rng.normal(mu, sd)
        p, q = float(np.exp(lp)), float(np.exp(lq))
        _, F = simulate_bass_sde(
            p, q, sigma, 30.0, dt=0.05, n_paths=1, seed=int(rng.integers(10**9))
        )
        obs = F[0, ::20]  # annual
        rows += [(f"t{i}", 1950 + k, float(v)) for k, v in enumerate(obs)]
        fit_rows.append((f"t{i}", p, q, 1.0, lp, lq))
    curves = pd.DataFrame(rows, columns=["technology", "year", "adoption"])
    fits = pd.DataFrame(fit_rows, columns=["technology", "p", "q", "m", "log_p", "log_q"])
    return curves, fits


def test_loo_scores_are_calibrated_when_model_is_well_specified() -> None:
    curves, fits = _planted_dataset(40, sigma=0.1, seed=7)
    scores = loo_forecast_scores(curves, fits, sigma=0.1, n_paths=800, seed=0)
    assert {"prior_sde", "prior_ode", "point_ode"} == set(scores.method)
    assert scores.crps.notna().all()
    cov = scores[scores.method == "prior_sde"].groupby("technology")[["cov_50", "cov_80"]].mean()
    assert 0.65 < cov.cov_80.mean() < 0.95
    assert 0.35 < cov.cov_50.mean() < 0.70
    # the deterministic point forecast cannot cover anything beyond chance hits
    pt = scores[scores.method == "point_ode"].cov_90.mean()
    assert pt < cov.cov_80.mean()


def test_probabilistic_forecast_beats_point_forecast_in_crps() -> None:
    curves, fits = _planted_dataset(30, sigma=0.1, seed=8)
    scores = loo_forecast_scores(curves, fits, sigma=0.1, n_paths=800, seed=1)
    mean_crps = scores.groupby("method").crps.mean()
    assert mean_crps["prior_sde"] < mean_crps["point_ode"]


def test_group_column_holds_out_whole_groups() -> None:
    from adoptsim.evaluation.calibration import training_indices

    fits = pd.DataFrame({"technology": ["a|x", "a|y", "b|x", "c"], "group": ["a", "a", "b", "c"]})
    assert training_indices(fits, 0) == [2, 3]  # sibling a|y is held out with a|x
    assert training_indices(fits, 3) == [0, 1, 2]
    # without a group column every other curve is a training curve
    assert training_indices(fits.drop(columns="group"), 1) == [0, 2, 3]


def test_summary_has_cluster_ci_per_horizon_bin() -> None:
    curves, fits = _planted_dataset(20, sigma=0.1, seed=9)
    scores = loo_forecast_scores(curves, fits, sigma=None, n_paths=300, seed=2)
    s = summarize_scores(scores)
    assert {"method", "horizon", "n_tech", "crps", "crps_lo", "crps_hi", "cov_80"} <= set(s.columns)
    assert (s.crps_lo <= s.crps).all() and (s.crps <= s.crps_hi).all()
    assert (s.n_tech <= 20).all()
