"""Scoring of probabilistic adoption forecasts and leave-one-technology-out calibration.

Forecasts are made for the market-normalised curve F = N/m, anchored at the first
observation. The market potential m is treated as known (a user scenario in the model),
so the scores isolate the uncertainty in the curve shape and the process noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adoptsim._typing import FloatArray
from adoptsim.evaluation.analogs import bootstrap_ci
from adoptsim.models.bass import bass_cumulative
from adoptsim.models.sde import simulate_bass_sde

LEVELS = (0.5, 0.8, 0.9)

type Series = dict[str, tuple[FloatArray, FloatArray]]


def crps_ensemble(samples: FloatArray, y: FloatArray) -> FloatArray:
    """CRPS of an ensemble, axis 0 holding the draws: mean|X - y| - 0.5 E|X - X'|.

    Uses the sorted-sample identity, O(n log n) instead of O(n^2) pairwise distances.
    """
    x = np.sort(np.asarray(samples, dtype=float), axis=0)
    n = x.shape[0]
    w = (2.0 * np.arange(1, n + 1) - n - 1.0).reshape((n,) + (1,) * (x.ndim - 1))
    term1 = np.abs(x - np.asarray(y, dtype=float)).mean(axis=0)
    term2 = (w * x).sum(axis=0) / n**2
    return np.asarray(term1 - term2, dtype=float)


def interval_covered(samples: FloatArray, y: FloatArray, level: float) -> FloatArray:
    """1.0 where y lies in the central `level` interval of the ensemble, else 0.0."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a], axis=0)
    return np.asarray((y >= lo) & (y <= hi), dtype=float)


def randomized_pit(samples: FloatArray, y: FloatArray, rng: np.random.Generator) -> FloatArray:
    """PIT value with randomisation over ties (paths absorbed at F = 1 produce point masses)."""
    below = (samples < y).mean(axis=0)
    ties = (samples == y).mean(axis=0)
    return np.asarray(below + rng.uniform(size=np.shape(y)) * ties, dtype=float)


def fit_population_prior(theta: FloatArray, ridge: float = 1e-6) -> tuple[FloatArray, FloatArray]:
    """Gaussian prior (mean, covariance) over theta = (log p, log q) from fitted curves."""
    mu = theta.mean(axis=0)
    cov = np.cov(theta, rowvar=False) + ridge * np.eye(theta.shape[1])
    return mu, cov


def increment_sigma2(t: FloatArray, f: FloatArray, p: float, q: float) -> float:
    """Moment estimate of sigma^2 from one-step deviations of the observed path.

    Conditional on F_k the SDE has Var(F_{k+1}) ~ sigma^2 F_k (1 - F_k) dt, so the
    scaled squared deviation from the deterministic one-step map estimates sigma^2.
    Includes observation noise and misfit, hence an upper bound on process noise.
    """
    vals = []
    for k in range(len(t) - 1):
        fk = float(f[k])
        if not 0.02 < fk < 0.98:
            continue
        dt = float(t[k + 1] - t[k])
        pred = float(bass_cumulative(t[k + 1], p, q, 1.0, t0=float(t[k]), n0=fk))
        vals.append((float(f[k + 1]) - pred) ** 2 / (fk * (1.0 - fk) * dt))
    return float(np.mean(vals)) if vals else float("nan")


def training_indices(fits: pd.DataFrame, i: int) -> list[int]:
    """Curves usable as training data for curve i in a leave-one-out fold.

    With a `group` column all curves of the same group are held out together, e.g. all
    country curves of one technology: a new technology has no sibling curves to learn from.
    """
    key = fits["group"] if "group" in fits.columns else fits["technology"]
    return [int(j) for j in np.flatnonzero((key != key.iloc[i]).to_numpy())]


def normalised_series(curves: pd.DataFrame, fits: pd.DataFrame) -> Series:
    """Per technology: time since first observation and adoption divided by fitted m."""
    series: Series = {}
    for tech, m in zip(fits["technology"], fits["m"].to_numpy(float), strict=True):
        g = curves[curves.technology == tech].sort_values("year")
        t = g["year"].to_numpy(float)
        f = np.clip(g["adoption"].to_numpy(float) / m, 0.0, 1.0)
        series[str(tech)] = (t - t[0], f)
    return series


def estimate_sigma(series: Series, fits: pd.DataFrame, names: list[str]) -> float:
    """Pooled sigma over the given technologies: sqrt of the median per-curve sigma^2."""
    keys = fits["technology"].astype(str)
    p_of = dict(zip(keys, fits["p"].to_numpy(float), strict=True))
    q_of = dict(zip(keys, fits["q"].to_numpy(float), strict=True))
    s2 = [increment_sigma2(*series[n], p_of[n], q_of[n]) for n in names]
    return float(np.sqrt(np.nanmedian(np.asarray(s2))))


def loo_forecast_scores(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    sigma: float | None = None,
    cov_scale: float = 1.0,
    n_paths: int = 2000,
    dt: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Leave-one-technology-out predictive scores for the population prior.

    For each held-out technology the prior is refitted on the others. Three methods:
    `prior_sde` (parameter + process noise), `prior_ode` (parameter noise only) and
    `point_ode` (prior mean, deterministic). `sigma=None` re-estimates sigma on the
    training curves in every fold. `cov_scale` multiplies the prior covariance (a
    sensitivity knob for over- or under-dispersion). Rows are per held-out observation
    after the anchor.
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
        cov = cov_scale * cov
        sig = sigma
        if sig is None:
            sig = estimate_sigma(series, fits, [names[j] for j in train])
        draws = np.exp(rng.multivariate_normal(mu, cov, size=n_paths))
        p, q = draws[:, 0], draws[:, 1]
        tau, y = series[name]
        f0 = float(min(y[0], 1.0 - 1e-6))
        tau_f, y_f = tau[1:], y[1:]

        _, paths = simulate_bass_sde(
            p, q, sig, float(tau[-1]), f0=f0, dt=dt, n_paths=n_paths, seed=int(rng.integers(2**31))
        )
        idx = np.rint(tau_f / dt).astype(int)
        point = bass_cumulative(tau_f, float(np.exp(mu[0])), float(np.exp(mu[1])), 1.0, 0.0, f0)
        ens = {
            "prior_sde": paths[:, idx],
            "prior_ode": bass_cumulative(tau_f[None, :], p[:, None], q[:, None], 1.0, 0.0, f0),
            "point_ode": np.tile(point, (2, 1)),
        }
        for method, samples in ens.items():
            row = {
                "technology": name,
                "method": method,
                "tau": tau_f,
                "y": y_f,
                "sigma": sig,
                "crps": crps_ensemble(samples, y_f),
                "pit": randomized_pit(samples, y_f, rng),
            }
            for lv in LEVELS:
                row[f"cov_{round(lv * 100)}"] = interval_covered(samples, y_f, lv)
            frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def summarize_scores(
    scores: pd.DataFrame, horizon_bins: tuple[float, ...] = (0.0, 5.0, 15.0, np.inf), seed: int = 0
) -> pd.DataFrame:
    """Mean CRPS and coverage per method and horizon bin, with technology-cluster bootstrap CIs.

    Observations within a curve are strongly dependent, so the effective sample size is the
    number of technologies: scores are averaged per technology before bootstrapping.
    """
    df = scores.copy()
    df["horizon"] = pd.cut(df["tau"], bins=list(horizon_bins), right=True)
    cols = ["crps", *[c for c in df.columns if c.startswith("cov_")]]
    out = []
    for (method, hz), g in df.groupby(["method", "horizon"], observed=True):
        per_tech = g.groupby("technology")[cols].mean()
        rec: dict[str, object] = {"method": method, "horizon": str(hz), "n_tech": len(per_tech)}
        for c in cols:
            lo, hi = bootstrap_ci(per_tech[c].to_numpy(), seed=seed)
            rec[c] = float(per_tech[c].mean())
            rec[f"{c}_lo"], rec[f"{c}_hi"] = lo, hi
        out.append(rec)
    return pd.DataFrame(out)


def pit_histogram(pit: FloatArray, bins: int = 10) -> FloatArray:
    """Relative frequencies of PIT values; flat at 1/bins for a calibrated forecast."""
    h, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return np.asarray(h / h.sum(), dtype=float)
