"""Pool HCCTA country curves with the OWID technologies and re-run calibration and H4.

HCCTA (Comin & Hobijn 2004, the predecessor of CHAT) reports counts, not shares. Per-capita
series of technologies that saturated inside the sample (telephones, radios, televisions,
private cars) are cut at their peak and divided by it, so the market potential is "known"
in the same sense as in the OWID analysis (m = observed saturation level). Series that are
still rising at the end of the sample are dropped: their peak would not be a saturation level.

Evaluation holds out whole groups (all countries of one technology), the realistic setting
for a new technology, instead of single curves, which would let sibling countries leak.
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
from adoptsim.data.schema import validate_curves
from adoptsim.evaluation import loo_forecast_scores, summarize_scores
from adoptsim.evaluation.early_data import early_data_scores, summarize_early

HCCTA_PATH = ROOT / "data" / "raw" / "hccta" / "hcctad.txt"
TECHS = ("Phones", "Radios", "Televisions", "Private Cars")
MIN_POINTS = 10
MAX_PEAK_YEAR = 1998  # sample ends in 2001; later peaks are probably not saturation levels
MAX_START_FRACTION = 0.5  # series must start well below saturation to inform the shape
LOG_P_PRIOR = (float(np.log(0.01)), 1.5)
TEMPER = 0.25
N_OBS = (1, 2, 3, 4, 6, 8)


def load_hccta_curves(path: Path = HCCTA_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-capita, peak-normalised country curves in the tidy schema, plus drop counts."""
    d = pd.read_csv(path, na_values=".")
    countries = list(d.columns[2:])
    pop = d[d.Variable == "Population"].set_index("Year")[countries]
    rows, drops = [], {"short": 0, "late_peak": 0, "starts_high": 0}
    for tech in TECHS:
        raw = d[d.Variable == tech].set_index("Year")[countries]
        per_capita = raw / pop
        for c in countries:
            s = per_capita[c].dropna()
            s = s[s > 0]
            if len(s) < MIN_POINTS:
                drops["short"] += 1
                continue
            years = s.index.to_numpy()
            # keep the longest block without gaps larger than 10 years, then cut at the peak
            cuts = np.flatnonzero(np.diff(years) > 10)
            if cuts.size:
                b = np.concatenate([[0], cuts + 1, [len(s)]])
                i = int(np.argmax(np.diff(b)))
                s = s.iloc[b[i] : b[i + 1]]
            s = s.iloc[: int(np.argmax(s.to_numpy())) + 1]
            peak = float(s.max())
            if len(s) < MIN_POINTS:
                drops["short"] += 1
            elif int(s.index[-1]) > MAX_PEAK_YEAR:
                drops["late_peak"] += 1
            elif s.iloc[0] / peak > MAX_START_FRACTION:
                drops["starts_high"] += 1
            else:
                rows += [(f"{tech}|{c}", "HCCTA", int(y), float(v / peak)) for y, v in s.items()]
    tidy = pd.DataFrame(rows, columns=["technology", "region", "year", "adoption"])
    return validate_curves(tidy), drops


def build_fits() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_owid_us_households(ROOT / "data" / "raw")
    owid = prepare_curves(raw, min_points=8, min_peak=0.15)
    owid = owid[~owid.technology.isin(NON_HOUSEHOLD | DUPLICATES)]
    hccta, drops = load_hccta_curves()
    print("HCCTA curves dropped:", drops)
    out_fits = []
    for source, cur in (("OWID", owid), ("HCCTA", hccta)):
        f = sl.fit_all(cur, log_p_prior=LOG_P_PRIOR)
        f = f[f.r2 >= 0.95].copy()
        f["source"] = source
        f["group"] = f.technology if source == "OWID" else f.technology.str.split("|").str[0]
        out_fits.append(f)
    fits = pd.concat(out_fits, ignore_index=True)
    curves = pd.concat([owid, hccta], ignore_index=True)
    curves = curves[curves.technology.isin(fits.technology)]
    return curves, fits


def main(quick: bool = False) -> None:
    curves, fits = build_fits()
    print("\ncurves per group:")
    print(fits.groupby("source").group.value_counts().to_string())
    print("\nlog p / log q mean, sd by source:")
    print(fits.groupby("source")[["log_p", "log_q"]].agg(["mean", "std"]).round(2).to_string())
    pd.set_option("display.width", 200)
    result: dict[str, object] = {
        "n_curves": len(fits),
        "n_by_source": fits.source.value_counts().to_dict(),
    }

    n_paths = 500 if quick else 1000
    sc = loo_forecast_scores(curves, fits, n_paths=n_paths, seed=0)
    src = fits.set_index("technology").source
    sc["source"] = sc.technology.map(src)
    print("\n== calibration, group-wise leave-out, pooled prior (sigma estimated) ==")
    cal = []
    for source, g in sc.groupby("source"):
        s = summarize_scores(g)
        s["source"] = source
        cal.append(s)
    cal_df = pd.concat(cal)
    cols = ["source", "method", "horizon", "n_tech", "crps", "cov_50", "cov_80", "cov_90"]
    print(cal_df[cols].round(3).to_string(index=False))
    result["calibration"] = cal_df.to_dict(orient="records")

    print("\n== H4: update vs prior vs NLS, group-wise leave-out, temper", TEMPER, "==")
    es = early_data_scores(
        curves,
        fits,
        n_obs=N_OBS,
        horizon=10,
        temper=TEMPER,
        likelihoods=("transition",),
        n_paths=n_paths,
        seed=0,
    )
    es["source"] = es.technology.map(src)
    h4 = []
    for source, g in es.groupby("source"):
        s = summarize_early(g)
        s["source"] = source
        h4.append(s)
        print(f"\n{source}: mean CRPS by T")
        print(s.pivot(index="method", columns="n_obs", values="crps").round(3).to_string())
        print(f"{source}: 80% coverage by T")
        print(s.pivot(index="method", columns="n_obs", values="cov_80").round(2).to_string())
    result["h4"] = pd.concat(h4).to_dict(orient="records")

    path = ROOT / "experiments" / "results" / "pooled_hccta_v0.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print("\nsaved", path.relative_to(ROOT))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
