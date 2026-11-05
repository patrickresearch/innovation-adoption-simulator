"""Stochastic Bass diffusion on the adoption fraction F in [0, 1].

    dF = (p + q F)(1 - F) x(t) dt + sigma sqrt(F (1 - F)) dW

The Wright-Fisher diffusion term vanishes at both boundaries, so the process
cannot leave [0, 1] in continuous time. Euler-Maruyama can still overshoot by
O(sqrt(dt)); paths are projected back, which is adequate for dt << 1/(p+q).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from adoptsim._typing import FloatArray


def simulate_bass_sde(
    p: float,
    q: float,
    sigma: float,
    horizon: float,
    *,
    f0: float = 0.0,
    dt: float = 0.01,
    n_paths: int = 1000,
    effort: Callable[[FloatArray], FloatArray] | None = None,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate paths of the adoption fraction.

    Returns (t, F) with t of shape (n_steps + 1,) and F of shape (n_paths, n_steps + 1).
    `effort` is the Generalized Bass multiplier x(t); None means x = 1.
    """
    if p <= 0 or q < 0 or sigma < 0:
        raise ValueError("require p > 0, q >= 0, sigma >= 0")
    if not 0.0 <= f0 < 1.0:
        raise ValueError("f0 must lie in [0, 1)")
    n_steps = int(np.ceil(horizon / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.ones_like(t) if effort is None else np.asarray(effort(t), dtype=float)

    rng = np.random.default_rng(seed)
    dw = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    F = np.empty((n_paths, n_steps + 1))
    F[:, 0] = f0
    for k in range(n_steps):
        f = F[:, k]
        drift = (p + q * f) * (1.0 - f) * x[k]
        diff = sigma * np.sqrt(np.clip(f * (1.0 - f), 0.0, None))
        F[:, k + 1] = np.clip(f + drift * dt + diff * dw[:, k], 0.0, 1.0)
    return t, F


def fan_quantiles(
    paths: FloatArray, qs: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> dict[float, FloatArray]:
    """Pointwise quantiles across paths, e.g. for P10/P50/P90 fan charts."""
    return {q: np.quantile(paths, q, axis=0) for q in qs}
