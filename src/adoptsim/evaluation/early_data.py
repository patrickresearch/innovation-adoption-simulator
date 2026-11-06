"""H4: does the Bayesian update beat the prior and the NLS fit when few periods are observed?

For every held-out technology the first T observations (the anchor included) are revealed
and the next `horizon` observations are forecast. The population prior is refitted on the
other technologies in every fold. Methods:

- `prior`: prior draws of theta, forecast from the last observed state (no parameter update)
- `post_transition` / `post_curve`: posterior draws under the two likelihoods
- `nls`, `nls_map`: deterministic fit on the T observations, anchored at the first one
  (only for T >= 4, the minimum the NLS routine accepts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adoptsim._typing import FloatArray
from adoptsim.evaluation.analogs import bootstrap_ci
from adoptsim.evaluation.calibration import (
    LEVELS,
    crps_ensemble,
    estimate_sigma,
    fit_population_prior,
    interval_covered,
    normalised_series,
    randomized_pit,
    training_indices,
)
from adoptsim.inference.posterior import posterior_from_data, predict_from_state
from adoptsim.models.bass import bass_cumulative, fit_bass

LOG_P_PRIOR = (float(np.log(0.01)), 1.5)


def _nls_forecast(
    tau: FloatArray, f: FloatArray, targets: FloatArray, *, regularised: bool
) -> FloatArray:
    fit = (
        fit_bass(tau, f, m_bounds=(1.0, 1.0001), log_p_prior=LOG_P_PRIOR)
        if regularised
        else fit_bass(tau, f, m_bounds=(1.0, 1.0001))
    )
    p, q = fit.params.p, fit.params.q
    return bass_cumulative(targets, p, q, 1.0, t0=0.0, n0=float(f[0]))


def early_data_scores(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    n_obs: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    horizon: int = 10,
    sigma: float | None = None,
    temper: float = 1.0,
    forecast_sigma_scale: float = 1.0,
    likelihoods: tuple[str, ...] = ("transition", "curve"),
    include_nls: bool = True,
    n_paths: int = 1000,
    dt: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Leave-one-technology-out scores of next-`horizon` forecasts given T observed points.

    `temper` is the likelihood power of the update, `forecast_sigma_scale` multiplies the
    process noise used for forecasting (sensitivity knobs for the calibration study).
    """
    fits = fits.reset_index(drop=True)
    series = normalised_series(curves, fits)
    names = fits.technology.astype(str).tolist()
    theta_all = fits[["log_p", "log_q"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    frames = []
    for i, name in enumerate(names):
        train = training_indices(fits, i)
        mu, cov = fit_population_prior(theta_all[train])
        sig = sigma
        if sig is None:
            sig = estimate_sigma(series, fits, [names[j] for j in train])
        tau, y = series[name]
        for T in n_obs:
            if len(y) <= T:
                continue
            stop = min(T + horizon, len(y))
            offsets = tau[T:stop] - tau[T - 1]
            y_f = y[T:stop]
            ens: dict[str, FloatArray] = {}
            fsig = forecast_sigma_scale * sig
            prior_draws = rng.multivariate_normal(mu, cov, size=n_paths)
            ens["prior"] = predict_from_state(
                prior_draws, y[T - 1], offsets, sigma=fsig, dt=dt, seed=int(rng.integers(2**31))
            )
            for lik in likelihoods:
                post = posterior_from_data(
                    tau[:T],
                    y[:T],
                    mu,
                    cov,
                    sigma=sig,
                    likelihood=lik,  # type: ignore[arg-type]
                    temper=temper,
                    n_samples=n_paths,
                    seed=int(rng.integers(2**31)),
                )
                ens[f"post_{lik}"] = predict_from_state(
                    post.samples,
                    y[T - 1],
                    offsets,
                    sigma=fsig,
                    dt=dt,
                    seed=int(rng.integers(2**31)),
                )
            if include_nls and T >= 4:
                for label, reg in (("nls", False), ("nls_map", True)):
                    point = _nls_forecast(tau[:T], y[:T], tau[T:stop], regularised=reg)
                    ens[label] = np.tile(point, (2, 1))
            for method, samples in ens.items():
                row: dict[str, object] = {
                    "technology": name,
                    "n_obs": T,
                    "method": method,
                    "ahead": np.arange(1, len(y_f) + 1),
                    "y": y_f,
                    "crps": crps_ensemble(samples, y_f),
                    "pit": randomized_pit(samples, y_f, rng),
                }
                for lv in LEVELS:
                    row[f"cov_{round(lv * 100)}"] = interval_covered(samples, y_f, lv)
                frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def summarize_early(scores: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Mean CRPS and coverage per method and T; per-technology means, cluster bootstrap CIs."""
    cols = ["crps", *[c for c in scores.columns if c.startswith("cov_")]]
    out = []
    for (method, T), g in scores.groupby(["method", "n_obs"]):
        per_tech = g.groupby("technology")[cols].mean()
        rec: dict[str, object] = {"method": method, "n_obs": T, "n_tech": len(per_tech)}
        for c in cols:
            lo, hi = bootstrap_ci(per_tech[c].to_numpy(), seed=seed)
            rec[c], rec[f"{c}_lo"], rec[f"{c}_hi"] = float(per_tech[c].mean()), lo, hi
        out.append(rec)
    return pd.DataFrame(out)
