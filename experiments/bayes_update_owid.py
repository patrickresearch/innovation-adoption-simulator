"""H4 on OWID: forecast error against the number of observed periods T.

Compares the population prior, the Bayesian update (two likelihoods) and NLS fits on the
first T observations. Leave-one-technology-out, MAP shape fits with a weak log p prior,
non-household and duplicate series removed (see robustness_owid.py).
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
from robustness_owid import DUPLICATES, NON_HOUSEHOLD

from adoptsim.data import load_owid_us_households, prepare_curves
from adoptsim.evaluation.early_data import early_data_scores, summarize_early

LOG_P_PRIOR = (float(np.log(0.01)), 1.5)
N_OBS = (1, 2, 3, 4, 6, 8)


def main() -> None:
    raw = load_owid_us_households(ROOT / "data" / "raw")
    curves = prepare_curves(raw, min_points=8, min_peak=0.15)
    fits = sl.fit_all(curves, log_p_prior=LOG_P_PRIOR)
    drop = NON_HOUSEHOLD | DUPLICATES
    fits = fits[(fits.r2 >= 0.95) & ~fits.technology.isin(drop)].reset_index(drop=True)
    cur = curves[curves.technology.isin(fits.technology)]
    print(f"n technologies: {len(fits)}")

    scores = early_data_scores(cur, fits, n_obs=N_OBS, horizon=10, n_paths=1000, seed=0)
    summ = summarize_early(scores)
    pd.set_option("display.width", 160)
    crps = summ.pivot(index="method", columns="n_obs", values="crps").round(3)
    print("\nmean CRPS (next 10 observations), by observed periods T:\n", crps)
    for col in ("cov_50", "cov_80", "cov_90"):
        print(f"\n{col}:\n", summ.pivot(index="method", columns="n_obs", values=col).round(2))
    n_tech = summ.pivot(index="method", columns="n_obs", values="n_tech")
    print("\ntechnologies per cell:\n", n_tech.loc["prior"].to_dict())

    path = ROOT / "experiments" / "results" / "bayes_update_v0_owid.json"
    path.write_text(json.dumps(summ.to_dict(orient="records"), indent=2, default=float))
    print("\nsaved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
