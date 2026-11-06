"""Does regularising p in the Bass fits repair the over-dispersed population prior?

ML fits push p to its lower bound when a series starts late (p is not identified); the
Gaussian prior over (log p, log q) then inherits a spurious spread. MAP fits with a weak
literature-based prior on log p (Sultan et al. 1990 report mean p of about 0.03) are
compared against the plain fits, together with a sensitivity sweep of the prior covariance.
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
from adoptsim.evaluation import loo_forecast_scores

LOG_P_MEAN = float(np.log(0.01))
FIT_MODES: dict[str, dict[str, object]] = {
    "ml": {},
    "map_tau3.0": {"log_p_prior": (LOG_P_MEAN, 3.0)},
    "map_tau1.5": {"log_p_prior": (LOG_P_MEAN, 1.5)},
    "map_tau1.0": {"log_p_prior": (LOG_P_MEAN, 1.0)},
}
COV_SCALES = (1.0, 0.5, 0.25)


def main() -> None:
    raw = load_owid_us_households(ROOT / "data" / "raw")
    curves = prepare_curves(raw, min_points=8, min_peak=0.15)
    drop = NON_HOUSEHOLD | DUPLICATES
    rows = []
    for mode, kw in FIT_MODES.items():
        fits = sl.fit_all(curves, **kw)
        fits = fits[(fits.r2 >= 0.95) & ~fits.technology.isin(drop)].reset_index(drop=True)
        cur = curves[curves.technology.isin(fits.technology)]
        th = fits[["log_p", "log_q"]].to_numpy()
        at_bound = int((fits.p < 2e-6).sum())
        print(
            f"\n{mode}: n={len(fits)} p_at_bound={at_bound} "
            f"sd(log p)={th[:, 0].std(ddof=1):.2f} sd(log q)={th[:, 1].std(ddof=1):.2f}"
        )
        for cs in COV_SCALES:
            sc = loo_forecast_scores(cur, fits, cov_scale=cs, n_paths=2000, seed=0)
            per = sc.groupby(["method", "technology"])[
                ["crps", "cov_50", "cov_80", "cov_90"]
            ].mean()
            m = per.groupby("method").mean()
            sde = m.loc["prior_sde"]
            row = {
                "fit": mode,
                "cov_scale": cs,
                "n": len(fits),
                "p_at_bound": at_bound,
                "crps_sde": sde.crps,
                "crps_point": m.loc["point_ode", "crps"],
                "cov_50": sde.cov_50,
                "cov_80": sde.cov_80,
                "cov_90": sde.cov_90,
            }
            rows.append(row)
            print(
                f"  scale={cs:<4} CRPS sde={sde.crps:.3f} point={row['crps_point']:.3f} | "
                f"cov50={sde.cov_50:.2f} cov80={sde.cov_80:.2f} cov90={sde.cov_90:.2f}"
            )
    out = pd.DataFrame(rows)
    path = ROOT / "experiments" / "results" / "prior_calibration_v0_owid.json"
    path.write_text(json.dumps(out.to_dict(orient="records"), indent=2, default=float))
    print("\nsaved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
