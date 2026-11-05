"""Robustness run for the phase-0 spike (exploratory, not public API).

Re-evaluates the go/no-go rule on log q under sample definitions that address known
weaknesses of the baseline run: non-household series and near-duplicate technologies
(which act as trivially close analogs in LOO), and the strict fit filters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import spike_lib as sl

from adoptsim.data import load_descriptions, load_owid_us_households, mask_leakage, prepare_curves

PRIMARY = "sentence:sentence-transformers/all-MiniLM-L6-v2"
ENCODERS = [PRIMARY, "sentence:BAAI/bge-small-en-v1.5", "tfidf"]
SEED = 0
N_PERM = 4999

# Not household shares: component penetration in cars, institutional or infrastructure series.
NON_HOUSEHOLD = {
    "Disk brakes",
    "Electronic ignition",
    "Power steering",
    "Radial tires",
    "RTGS adoption",
    "Shipping container port infrastructure",
    "Nox pollution controls (boilers)",
}
# Same underlying product measured twice; each would be the other's nearest analog.
DUPLICATES = {"Refrigerator", "Microcomputer"}

VARIANTS = {
    "baseline": dict(exclude=set(), min_points=8, min_r2=0.95),
    "households_only": dict(exclude=NON_HOUSEHOLD, min_points=8, min_r2=0.95),
    "households_dedup": dict(exclude=NON_HOUSEHOLD | DUPLICATES, min_points=8, min_r2=0.95),
    "households_dedup_relaxed": dict(
        exclude=NON_HOUSEHOLD | DUPLICATES, min_points=6, min_r2=0.90
    ),
}


def run_variant(raw: pd.DataFrame, exclude: set[str], min_points: int, min_r2: float) -> dict:
    curves = prepare_curves(raw, min_points=min_points, min_peak=0.15)
    curves = curves[~curves.technology.isin(exclude)]
    fits_all = sl.fit_all(curves)
    fits = fits_all[fits_all.r2 >= min_r2].reset_index(drop=True)
    desc = load_descriptions(
        fits.technology.tolist(),
        cache_path=ROOT / "data" / "interim" / "descriptions.csv",
        title_overrides=ROOT / "data" / "external" / "wikipedia_titles.csv",
        manual_path=ROOT / "data" / "external" / "descriptions_manual.csv",
    )
    ok = desc.description.str.len() > 0
    dropped = desc[~ok].technology.tolist()
    fits = fits[fits.technology.isin(desc[ok].technology)].reset_index(drop=True)
    desc = desc[ok].set_index("technology").loc[fits.technology].reset_index()
    texts = [mask_leakage(t, n_sentences=2) for t in desc.description]

    emb = {}
    for kind in ENCODERS:
        e, name = sl.try_encode(kind, texts)
        if e is not None:
            emb[name] = e
    suite = sl.run_suite(emb, fits, k=5, n_perm=N_PERM, seed=SEED)
    decision = sl.go_no_go(suite, PRIMARY)
    return {
        "n": len(fits),
        "no_description": dropped,
        "technologies": fits.technology.tolist(),
        "decision": decision,
        "mantel_log_q": suite.mantel.query("target == 'log_q'").to_dict(orient="records"),
        "loo_mae_log_q": suite.errors.query("target == 'log_q'").to_dict(orient="records"),
        "tests_log_q": suite.tests.query("target == 'log_q'").to_dict(orient="records"),
    }


def main() -> None:
    raw = load_owid_us_households(ROOT / "data" / "raw")
    out = {}
    for name, cfg in VARIANTS.items():
        out[name] = run_variant(raw, **cfg)
        d = out[name]["decision"]
        print(
            f"{name:28s} n={out[name]['n']:3d}  verdict={d['verdict']:8s} "
            f"mantel_p={d['mantel_log_q_p']:.3f}"
        )
        for r in out[name]["mantel_log_q"]:
            print(f"    {r['encoder'][-28:]:>28s}  r={r['r']:+.3f}  p={r['p']:.3f}")
    path = ROOT / "experiments" / "results" / "robustness_v0_owid.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
