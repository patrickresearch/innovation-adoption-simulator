"""Deterministic Bass diffusion: closed form, fitting and shape descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from adoptsim._typing import FloatArray


@dataclass(frozen=True)
class BassParams:
    """Bass parameters on the original scale.

    p: coefficient of innovation, q: coefficient of imitation,
    m: market potential (same unit as the adoption series).
    """

    p: float
    q: float
    m: float

    def __post_init__(self) -> None:
        if self.p <= 0 or self.q < 0 or self.m <= 0:
            raise ValueError(f"invalid Bass parameters: {self}")


@dataclass(frozen=True)
class BassFit:
    params: BassParams
    t0: float
    n0: float
    rmse: float
    r2: float
    n_obs: int


def bass_cumulative(
    t: ArrayLike, p: float, q: float, m: float, t0: float = 0.0, n0: float = 0.0
) -> FloatArray:
    """Cumulative adoption N(t) solving dN/dt = (p + q N/m)(m - N) with N(t0) = n0.

    The general initial condition avoids having to estimate an unobserved launch
    date, which is poorly identified jointly with p when series start late.
    """
    t_arr = np.asarray(t, dtype=float)
    f0 = n0 / m
    if not 0.0 <= f0 < 1.0:
        raise ValueError("n0 must lie in [0, m)")
    r = q / p
    c = (1.0 - f0) / (1.0 + r * f0)
    e = np.exp(-(p + q) * (t_arr - t0))
    return np.asarray(m * (1.0 - c * e) / (1.0 + r * c * e), dtype=float)


def bass_rate(
    t: ArrayLike, p: float, q: float, m: float, t0: float = 0.0, n0: float = 0.0
) -> FloatArray:
    """Instantaneous adoption rate dN/dt."""
    n = bass_cumulative(t, p, q, m, t0, n0)
    return np.asarray((p + q * n / m) * (m - n), dtype=float)


def time_to_fraction(frac: float, p: float, q: float, f0: float = 0.0) -> float:
    """Time (relative to t0) at which F = N/m reaches `frac`; inverse of the closed form."""
    if not f0 <= frac < 1.0:
        raise ValueError("frac must lie in [f0, 1)")
    r = q / p
    c = (1.0 - f0) / (1.0 + r * f0)
    # Solve frac = (1 - c e) / (1 + r c e) for e = exp(-(p+q) t).
    e = (1.0 - frac) / (c * (1.0 + r * frac))
    return float(-np.log(e) / (p + q))


def takeoff_duration(p: float, q: float, lo: float = 0.1, hi: float = 0.9) -> float:
    """Duration from `lo` to `hi` of saturation; invariant to the time origin."""
    return time_to_fraction(hi, p, q) - time_to_fraction(lo, p, q)


def peak_time(p: float, q: float) -> float:
    """Time of maximum adoption rate for n0 = 0; negative means no interior peak (q <= p)."""
    return float(np.log(q / p) / (p + q))


def fit_bass(
    t: ArrayLike,
    y: ArrayLike,
    *,
    m_bounds: tuple[float, float] | None = None,
    n_starts: int = 12,
    seed: int = 0,
) -> BassFit:
    """Nonlinear least-squares fit anchored at the first observation.

    Parameters are optimised in log space for scale invariance. Multiple random
    starts guard against the flat likelihood ridges typical for p.
    """
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.shape != y_arr.shape or t_arr.ndim != 1:
        raise ValueError("t and y must be 1-d arrays of equal length")
    if len(t_arr) < 4:
        raise ValueError("need at least 4 observations")
    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]
    t0, n0 = float(t_arr[0]), float(max(y_arr[0], 0.0))
    y_max = float(y_arr.max())
    if y_max <= 0:
        raise ValueError("series contains no adoption")

    lo_m, hi_m = m_bounds if m_bounds is not None else (y_max * 1.0001, y_max * 3.0)
    lo_m = max(lo_m, n0 * 1.0001, 1e-12)
    lb = np.log([1e-6, 1e-4, lo_m])
    ub = np.log([1.0, 3.0, hi_m])

    def residuals(theta: FloatArray) -> FloatArray:
        p, q, m = np.exp(theta)
        return np.asarray(bass_cumulative(t_arr, p, q, m, t0, n0) - y_arr, dtype=float)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(lb, ub)
        res = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    p, q, m = (float(v) for v in np.exp(best.x))
    resid = best.fun
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    return BassFit(
        params=BassParams(p, q, m),
        t0=t0,
        n0=n0,
        rmse=float(np.sqrt(ss_res / len(y_arr))),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n_obs=len(y_arr),
    )
