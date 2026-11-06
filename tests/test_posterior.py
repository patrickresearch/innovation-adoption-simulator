import numpy as np
import pandas as pd
import pytest

from adoptsim.evaluation.early_data import early_data_scores, summarize_early
from adoptsim.inference import log_likelihood, ode_step, posterior_from_data, predict_from_state
from adoptsim.models.bass import bass_cumulative
from adoptsim.models.sde import simulate_bass_sde

PRIOR_MU = np.array([np.log(0.02), np.log(0.4)])
PRIOR_COV = np.diag([0.5**2, 0.3**2])
TRUE = np.array([np.log(0.03), np.log(0.5)])
SIGMA = 0.08


def _observed(theta: np.ndarray, n_years: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _, F = simulate_bass_sde(
        np.exp(theta[0]), np.exp(theta[1]), SIGMA, n_years, dt=0.05, n_paths=1, seed=seed
    )
    return np.arange(n_years + 1, dtype=float), F[0, ::20]


def test_ode_step_matches_closed_form() -> None:
    f = np.array([0.0, 0.1, 0.6])
    dt = np.array([1.0, 2.0, 0.5])
    got = ode_step(f, dt, np.array([[0.03]]), np.array([[0.4]]))[0]
    for k in range(3):
        assert got[k] == pytest.approx(
            float(bass_cumulative(dt[k], 0.03, 0.4, 1.0, 0.0, f[k])), abs=1e-9
        )


def test_anchor_only_returns_prior() -> None:
    post = posterior_from_data(
        np.array([0.0]), np.array([0.05]), PRIOR_MU, PRIOR_COV, sigma=SIGMA, n_samples=4000
    )
    np.testing.assert_allclose(post.mode, PRIOR_MU, atol=1e-3)
    np.testing.assert_allclose(post.samples.mean(axis=0), PRIOR_MU, atol=0.03)
    np.testing.assert_allclose(np.cov(post.samples.T), PRIOR_COV, atol=0.03)


@pytest.mark.parametrize("likelihood", ["transition", "curve"])
def test_importance_posterior_matches_grid(likelihood: str) -> None:
    t, f = _observed(TRUE, 8, seed=1)
    post = posterior_from_data(
        t,
        f,
        PRIOR_MU,
        PRIOR_COV,
        sigma=SIGMA,
        likelihood=likelihood,  # type: ignore[arg-type]
        n_samples=20000,
        n_proposals=60000,
        seed=0,
    )
    sd = np.sqrt(np.diag(PRIOR_COV))
    g0 = np.linspace(PRIOR_MU[0] - 4 * sd[0], PRIOR_MU[0] + 4 * sd[0], 241)
    g1 = np.linspace(PRIOR_MU[1] - 4 * sd[1], PRIOR_MU[1] + 4 * sd[1], 241)
    grid = np.array([(a, b) for a in g0 for b in g1])
    ll = log_likelihood(grid, t, f, sigma=SIGMA, kind=likelihood)  # type: ignore[arg-type]
    d = grid - PRIOR_MU
    lp = ll - 0.5 * np.sum(d**2 / np.diag(PRIOR_COV), axis=1)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = w @ grid
    sd_post = np.sqrt(w @ (grid - mean) ** 2)
    assert np.all(np.abs(post.samples.mean(axis=0) - mean) < 0.1 * sd)
    np.testing.assert_allclose(post.samples.std(axis=0), sd_post, rtol=0.1)
    assert post.ess > 500


def test_posterior_tightens_and_moves_towards_truth_with_more_data() -> None:
    sds, errs = [], []
    for n_years in (2, 5, 10):
        t, f = _observed(TRUE, n_years, seed=2)
        post = posterior_from_data(t, f, PRIOR_MU, PRIOR_COV, sigma=SIGMA, n_samples=3000)
        sds.append(post.samples[:, 1].std())
        errs.append(abs(post.samples[:, 1].mean() - TRUE[1]))
    assert sds[0] > sds[-1]
    assert errs[-1] < abs(PRIOR_MU[1] - TRUE[1]) + 0.05


def test_laplace_and_importance_agree_when_posterior_is_near_gaussian() -> None:
    t, f = _observed(TRUE, 10, seed=3)
    a = posterior_from_data(
        t, f, PRIOR_MU, PRIOR_COV, sigma=SIGMA, method="laplace", n_samples=5000
    )
    b = posterior_from_data(t, f, PRIOR_MU, PRIOR_COV, sigma=SIGMA, n_samples=5000)
    np.testing.assert_allclose(a.samples.mean(axis=0), b.samples.mean(axis=0), atol=0.15)
    assert np.isnan(a.ess) and b.ess > 100


def test_predict_from_state_matches_ode_without_noise() -> None:
    theta = np.tile(np.array([np.log(0.03), np.log(0.4)]), (4, 1))
    offsets = np.array([1.0, 3.0, 7.0])
    got = predict_from_state(theta, 0.2, offsets, sigma=0.0, dt=0.01, seed=0)
    expected = bass_cumulative(offsets, 0.03, 0.4, 1.0, 0.0, 0.2)
    np.testing.assert_allclose(got, np.tile(expected, (4, 1)), atol=2e-3)


def _planted_dataset(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sd = np.array([0.5, 0.3])
    rows, fit_rows = [], []
    for i in range(n):
        theta = rng.normal(PRIOR_MU, sd)
        t, f = _observed(theta, 25, seed=int(rng.integers(10**9)))
        rows += [(f"t{i}", 1950 + int(k), float(v)) for k, v in zip(t, f, strict=True)]
        fit_rows.append((f"t{i}", float(np.exp(theta[0])), float(np.exp(theta[1])), 1.0, *theta))
    curves = pd.DataFrame(rows, columns=["technology", "year", "adoption"])
    fits = pd.DataFrame(fit_rows, columns=["technology", "p", "q", "m", "log_p", "log_q"])
    return curves, fits


def test_update_beats_prior_with_enough_data_in_well_specified_model() -> None:
    curves, fits = _planted_dataset(12, seed=4)
    sc = early_data_scores(curves, fits, n_obs=(1, 8), horizon=8, sigma=SIGMA, n_paths=300, seed=0)
    s = summarize_early(sc).set_index(["method", "n_obs"])
    assert {"prior", "post_transition", "post_curve", "nls", "nls_map"} <= set(sc.method.unique())
    assert s.loc[("post_transition", 8), "crps"] < s.loc[("prior", 8), "crps"]
    # with the anchor only, the update has nothing to learn from: posterior == prior
    assert s.loc[("post_transition", 1), "crps"] == pytest.approx(
        s.loc[("prior", 1), "crps"], rel=0.15
    )
    assert ("nls", 1) not in s.index  # NLS needs at least four observations
