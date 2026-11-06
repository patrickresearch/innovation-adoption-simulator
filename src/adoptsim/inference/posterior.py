"""Bayesian update of the Bass shape parameters from the first observations.

theta = (log p, log q) with a Gaussian prior (e.g. the population prior); the market
potential is known, so F = N/m is observed and anchored at its first value. Two
likelihoods are offered:

- "transition": Euler-type transition densities of the SDE,
  F_{k+1} | F_k ~ N(ode_step(F_k), sigma^2 max(F_k (1 - F_k), floor) dt + obs_sd^2),
  which conditions every step on the observed state and is consistent with the forecast model.
- "curve": Gaussian errors around the deterministic curve anchored at F_0, the Bayesian
  counterpart of the NLS fit.

The posterior is explored by a Laplace approximation, optionally corrected by importance
resampling with a Student-t proposal (the exact two-dimensional posterior is not Gaussian
when the data are weakly informative about p).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_t

from adoptsim._typing import FloatArray
from adoptsim.models.sde import simulate_bass_sde

Likelihood = Literal["transition", "curve"]
STATE_FLOOR = 1e-2  # variance floor for F (1 - F): the diffusion term vanishes at the boundaries


def ode_step(f: FloatArray, dt: FloatArray, p: FloatArray, q: FloatArray) -> FloatArray:
    """Closed-form Bass solution (m = 1) after time dt starting from fraction f.

    Broadcasts: p and q of shape (n, 1) against f and dt of shape (K,) give (n, K).
    """
    f = np.clip(f, 0.0, 1.0 - 1e-9)
    r = q / p
    c = (1.0 - f) / (1.0 + r * f)
    e = np.exp(-(p + q) * dt)
    return np.asarray((1.0 - c * e) / (1.0 + r * c * e), dtype=float)


def log_likelihood(
    theta: FloatArray,
    t: FloatArray,
    f: FloatArray,
    *,
    sigma: float,
    obs_sd: float = 0.005,
    kind: Likelihood = "transition",
    curve_sd: float = 0.02,
) -> FloatArray:
    """Log-likelihood for a batch of theta (shape (n, 2)); zero when only the anchor is seen."""
    theta = np.atleast_2d(theta)
    n = theta.shape[0]
    if len(t) < 2:
        return np.zeros(n)
    p = np.exp(theta[:, [0]])
    q = np.exp(theta[:, [1]])
    if kind == "transition":
        dt = np.diff(t)
        mean = ode_step(f[:-1], dt, p, q)
        var = sigma**2 * np.maximum(f[:-1] * (1.0 - f[:-1]), STATE_FLOOR) * dt + obs_sd**2
    else:
        mean = ode_step(np.array([f[0]]), t[1:] - t[0], p, q)
        var = np.full(mean.shape[1], curve_sd**2)
    ll = -0.5 * ((f[1:] - mean) ** 2 / var + np.log(2.0 * np.pi * var))
    return np.asarray(ll.sum(axis=1), dtype=float)


def _log_prior(theta: FloatArray, mu: FloatArray, cov: FloatArray) -> FloatArray:
    d = np.atleast_2d(theta) - mu
    sol = np.linalg.solve(cov, d.T).T
    _, logdet = np.linalg.slogdet(cov)
    return np.asarray(-0.5 * (np.sum(d * sol, axis=1) + logdet + len(mu) * np.log(2 * np.pi)))


def _hessian(fun: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-3) -> FloatArray:
    k = len(x)
    H = np.empty((k, k))
    eye = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (
                fun(x + eye[i] + eye[j])
                - fun(x + eye[i] - eye[j])
                - fun(x - eye[i] + eye[j])
                + fun(x - eye[i] - eye[j])
            ) / (4 * h * h)
    return 0.5 * (H + H.T)


@dataclass(frozen=True)
class Posterior:
    """Posterior summary: Laplace mode/covariance and draws of theta = (log p, log q)."""

    mode: FloatArray
    cov: FloatArray
    samples: FloatArray
    ess: float  # effective sample size of the importance weights; nan for pure Laplace


def posterior_from_data(
    t: FloatArray,
    f: FloatArray,
    prior_mu: FloatArray,
    prior_cov: FloatArray,
    *,
    sigma: float,
    obs_sd: float = 0.005,
    likelihood: Likelihood = "transition",
    curve_sd: float = 0.02,
    n_samples: int = 2000,
    method: Literal["laplace", "importance"] = "importance",
    n_proposals: int = 20000,
    seed: int = 0,
) -> Posterior:
    """Posterior of theta given observations (t, f) of the normalised adoption curve."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    def log_post(theta: FloatArray) -> FloatArray:
        ll = log_likelihood(
            theta, t, f, sigma=sigma, obs_sd=obs_sd, kind=likelihood, curve_sd=curve_sd
        )
        return ll + _log_prior(theta, prior_mu, prior_cov)

    def neg(x: FloatArray) -> float:
        return -float(log_post(x)[0])

    sd = np.sqrt(np.diag(prior_cov))
    starts = [prior_mu, prior_mu + sd, prior_mu - sd]
    runs = [
        minimize(neg, s, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
        for s in starts
    ]
    mode = np.asarray(min(runs, key=lambda r: r.fun).x, dtype=float)

    evals, evecs = np.linalg.eigh(_hessian(neg, mode))
    # not concave at the mode: fall back to the prior curvature
    cov = evecs @ np.diag(1.0 / evals) @ evecs.T if np.all(evals > 1e-9) else prior_cov.copy()

    if method == "laplace":
        draws = rng.multivariate_normal(mode, cov, size=n_samples)
        return Posterior(mode, cov, draws, float("nan"))

    nu = 5.0
    prop = multivariate_t(loc=mode, shape=1.5 * cov * (nu - 2.0) / nu, df=nu, seed=rng)
    cand = np.atleast_2d(prop.rvs(size=n_proposals))
    logw = log_post(cand) - prop.logpdf(cand)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    idx = rng.choice(n_proposals, size=n_samples, replace=True, p=w)
    return Posterior(mode, cov, cand[idx], ess)


def predict_from_state(
    theta: FloatArray,
    f_last: float,
    offsets: FloatArray,
    *,
    sigma: float,
    dt: float = 0.05,
    seed: int | None = None,
) -> FloatArray:
    """SDE forecast of F at times `offsets` after the last observation; shape (n, len(offsets))."""
    p, q = np.exp(theta[:, 0]), np.exp(theta[:, 1])
    f0 = float(min(f_last, 1.0 - 1e-6))
    _, paths = simulate_bass_sde(
        p, q, sigma, float(np.max(offsets)), f0=f0, dt=dt, n_paths=len(p), seed=seed
    )
    return np.asarray(paths[:, np.rint(offsets / dt).astype(int)], dtype=float)
