"""Helpers for the phase-0 spike. Exploratory code, not part of the public API."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from adoptsim.embeddings import Encoder, SentenceTransformerEncoder, TfidfEncoder
from adoptsim.evaluation import (
    bootstrap_ci,
    cosine_distance_matrix,
    knn_loo_predict,
    mantel_test,
    mean_loo_predict,
    paired_permutation_test,
    random_analog_loo_error,
)
from adoptsim.models import fit_bass, takeoff_duration

TARGETS = ("log_q", "log_p", "takeoff_10_90")


def fit_all(curves: pd.DataFrame, **fit_kwargs: Any) -> pd.DataFrame:
    """Fit a Bass curve per technology with m bounded by 100 % adoption."""
    rows = []
    for tech, g in curves.groupby("technology", sort=True):
        t, y = g["year"].to_numpy(float), g["adoption"].to_numpy(float)
        y_max = float(y.max())
        try:
            m_bounds = (y_max * 1.0001, max(1.0, y_max * 1.001))
            fit = fit_bass(t, y, m_bounds=m_bounds, **fit_kwargs)
        except ValueError as exc:
            warnings.warn(f"{tech}: {exc}", stacklevel=2)
            continue
        p, q, m = fit.params.p, fit.params.q, fit.params.m
        rows.append(
            {
                "technology": tech,
                "first_year": int(t.min()),
                "n_obs": fit.n_obs,
                "p": p,
                "q": q,
                "m": m,
                "r2": fit.r2,
                "rmse": fit.rmse,
                "log_p": np.log(p),
                "log_q": np.log(q),
                "takeoff_10_90": takeoff_duration(p, q),
            }
        )
    return pd.DataFrame(rows)


def make_encoder(kind: str) -> Encoder:
    if kind == "tfidf":
        return TfidfEncoder()
    if kind.startswith("sentence"):
        model = kind.split(":", 1)[1] if ":" in kind else None
        return SentenceTransformerEncoder(model) if model else SentenceTransformerEncoder()
    raise ValueError(kind)


def try_encode(kind: str, texts: list[str]) -> tuple[np.ndarray | None, str]:
    try:
        return make_encoder(kind).encode(texts), kind
    except ImportError as exc:
        warnings.warn(f"{kind} unavailable ({exc}); skipped", stacklevel=2)
        return None, kind


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


@dataclass
class SuiteResult:
    mantel: pd.DataFrame
    errors: pd.DataFrame
    tests: pd.DataFrame


def run_suite(
    embeddings: dict[str, np.ndarray],
    fits: pd.DataFrame,
    *,
    k: int = 5,
    allowed: np.ndarray | None = None,
    n_perm: int = 9999,
    seed: int = 0,
) -> SuiteResult:
    """Mantel tests (H1) and LOO analog errors with paired tests (H2) for all encoders."""
    mantel_rows, err_rows, test_rows = [], [], []
    for target in TARGETS:
        y = fits[target].to_numpy(float)
        d_y = np.abs(y[:, None] - y[None, :])
        base_err = {
            "global_mean": np.abs(mean_loo_predict(y, allowed=allowed)[:, 0] - y),
            "random_analogs": random_analog_loo_error(y, k, allowed=allowed, seed=seed)[:, 0],
        }
        enc_err = {}
        for name, emb in embeddings.items():
            res = mantel_test(cosine_distance_matrix(emb), d_y, n_perm=n_perm, seed=seed)
            mantel_rows.append(
                {"target": target, "encoder": name, "r": res.statistic, "p": res.p_value}
            )
            enc_err[name] = np.abs(knn_loo_predict(emb, y, k, allowed=allowed)[:, 0] - y)
        for name, err in {**base_err, **enc_err}.items():
            lo, hi = bootstrap_ci(err, seed=seed)
            err_rows.append(
                {"target": target, "method": name, "mae": np.nanmean(err), "ci_lo": lo, "ci_hi": hi}
            )
        raw_p = {}
        for name, err in enc_err.items():
            comparators = {**base_err, **{o: e for o, e in enc_err.items() if o != name}}
            for bname, berr in comparators.items():
                r = paired_permutation_test(err, berr, n_perm=n_perm, seed=seed)
                key = f"{target}|{name}|{bname}"
                raw_p[key] = r.p_value
                test_rows.append(
                    {
                        "target": target,
                        "model": name,
                        "baseline": bname,
                        "mae_gain": r.statistic,
                        "p": r.p_value,
                        "key": key,
                    }
                )
        adj = holm(raw_p)
        for row in test_rows:
            if row["key"] in adj:
                row["p_holm"] = adj[row["key"]]
    tests = pd.DataFrame(test_rows).drop(columns="key")
    return SuiteResult(pd.DataFrame(mantel_rows), pd.DataFrame(err_rows), tests)


def top_analogs(emb: np.ndarray, names: list[str], n: int = 3) -> pd.DataFrame:
    D = cosine_distance_matrix(emb)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :n]
    return pd.DataFrame(
        {
            "technology": names,
            **{
                f"analog_{j + 1}": [
                    f"{names[i]} ({1 - D[r, i]:.2f})" for r, i in enumerate(idx[:, j])
                ]
                for j in range(n)
            },
        }
    )


def go_no_go(suite: SuiteResult, primary: str, alpha: float = 0.05) -> dict[str, object]:
    """Decision rule from the concept, section 7, evaluated on log q."""
    m = suite.mantel.query("target == 'log_q' and encoder == @primary")
    t = suite.tests.query("target == 'log_q' and model == @primary")

    def p_of(b: str) -> float:
        row = t[t.baseline == b]
        return float(row["p_holm"].iloc[0]) if len(row) else float("nan")

    mantel_ok = bool(len(m) and m["p"].iloc[0] < alpha)
    beats_mean = p_of("global_mean") < alpha
    beats_random = p_of("random_analogs") < alpha
    beats_tfidf = p_of("tfidf") < alpha if primary != "tfidf" else None
    if mantel_ok and beats_mean and beats_random:
        verdict = "GO" if beats_tfidf in (True, None) else "SOFT GO"
    else:
        verdict = "NO-GO"
    return {
        "primary_encoder": primary,
        "mantel_log_q_p": float(m["p"].iloc[0]) if len(m) else float("nan"),
        "beats_global_mean_p_holm": p_of("global_mean"),
        "beats_random_analogs_p_holm": p_of("random_analogs"),
        "beats_tfidf_p_holm": p_of("tfidf") if primary != "tfidf" else None,
        "verdict": verdict,
    }
