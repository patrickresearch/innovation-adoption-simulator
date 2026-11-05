"""Forecasting-by-analogy evaluation: kNN leave-one-out, Mantel and permutation tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from adoptsim._typing import BoolArray, FloatArray


def cosine_distance_matrix(emb: FloatArray) -> FloatArray:
    """Cosine distance for L2-normalised rows."""
    sim = np.clip(emb @ emb.T, -1.0, 1.0)
    return np.asarray(1.0 - sim, dtype=float)


def _candidates(n: int, allowed: BoolArray | None) -> BoolArray:
    mask = np.ones((n, n), dtype=bool) if allowed is None else allowed.astype(bool).copy()
    np.fill_diagonal(mask, False)
    return mask


def knn_loo_predict(
    emb: FloatArray, targets: FloatArray, k: int = 5, *, allowed: BoolArray | None = None
) -> FloatArray:
    """Leave-one-out prediction as the mean target of the k nearest analogs.

    Unweighted averaging follows Goodwin et al. (2013), who found little gain
    beyond five or six analogs. `allowed[i, j]` restricts which j may serve as an
    analog for i (e.g. only technologies launched earlier). Rows without any
    allowed analog are NaN.
    """
    Y = np.atleast_2d(np.asarray(targets, dtype=float).T).T
    n = len(Y)
    D = cosine_distance_matrix(emb)
    cand = _candidates(n, allowed)
    out = np.full_like(Y, np.nan)
    for i in range(n):
        idx = np.flatnonzero(cand[i])
        if idx.size == 0:
            continue
        nn = idx[np.argsort(D[i, idx], kind="stable")[:k]]
        out[i] = Y[nn].mean(axis=0)
    return out


def mean_loo_predict(targets: FloatArray, *, allowed: BoolArray | None = None) -> FloatArray:
    """Baseline: mean over all other (allowed) items."""
    Y = np.atleast_2d(np.asarray(targets, dtype=float).T).T
    cand = _candidates(len(Y), allowed)
    out = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        idx = np.flatnonzero(cand[i])
        if idx.size:
            out[i] = Y[idx].mean(axis=0)
    return out


def random_analog_loo_error(
    targets: FloatArray,
    k: int = 5,
    *,
    n_rep: int = 200,
    allowed: BoolArray | None = None,
    seed: int = 0,
) -> FloatArray:
    """Expected absolute error per item when k analogs are picked at random."""
    Y = np.atleast_2d(np.asarray(targets, dtype=float).T).T
    cand = _candidates(len(Y), allowed)
    rng = np.random.default_rng(seed)
    err = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        idx = np.flatnonzero(cand[i])
        if idx.size == 0:
            continue
        kk = min(k, idx.size)
        draws = np.stack([Y[rng.choice(idx, kk, replace=False)].mean(axis=0) for _ in range(n_rep)])
        err[i] = np.abs(draws - Y[i]).mean(axis=0)
    return err


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    n_perm: int


def mantel_test(
    d1: FloatArray, d2: FloatArray, *, n_perm: int = 9999, seed: int = 0, method: str = "spearman"
) -> TestResult:
    """One-sided Mantel test for positive association between two distance matrices."""
    n = d1.shape[0]
    iu = np.triu_indices(n, 1)

    def corr(a: FloatArray, b: FloatArray) -> float:
        if method == "spearman":
            a, b = rankdata(a), rankdata(b)
        return float(np.corrcoef(a, b)[0, 1])

    obs = corr(d1[iu], d2[iu])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if corr(d1[iu], d2[np.ix_(perm, perm)][iu]) >= obs:
            hits += 1
    return TestResult(obs, (hits + 1) / (n_perm + 1), n_perm)


def paired_permutation_test(
    err_model: FloatArray, err_baseline: FloatArray, *, n_perm: int = 9999, seed: int = 0
) -> TestResult:
    """One-sided sign-flip test of H1: mean(err_model) < mean(err_baseline).

    Statistic is the mean paired difference (baseline - model); positive favours the model.
    """
    a = np.asarray(err_model, dtype=float)
    b = np.asarray(err_baseline, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    diff = b[ok] - a[ok]
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return TestResult(obs, float(p), n_perm)


def bootstrap_ci(
    x: FloatArray, *, n_boot: int = 5000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
