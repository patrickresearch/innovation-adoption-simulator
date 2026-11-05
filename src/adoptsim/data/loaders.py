"""Adoption-curve loaders.

Sources:
- Our World in Data, "Technology adoption by households in the United States"
  (compiled from Comin & Hobijn 2004 and others), share of US households in percent.
- Any CSV in the tidy schema (technology, region, year, adoption).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from adoptsim.data.schema import validate_curves

OWID_SLUG = "technology-adoption-by-households-in-the-united-states"
OWID_URL = (
    f"https://ourworldindata.org/grapher/{OWID_SLUG}.csv?v=1&csvType=full&useColumnShortNames=false"
)
USER_AGENT = "adoptsim/0.0.1 (research software)"


def download(url: str, dest: Path, *, timeout: float = 30.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_owid_csv(path: Path | str, region: str = "USA") -> pd.DataFrame:
    """Parse an OWID grapher CSV where `Entity` holds the technology name.

    The value column name differs between OWID exports, so the single numeric
    column besides Year is used. Percent values are rescaled to fractions.
    """
    raw = pd.read_csv(path)
    if not {"Entity", "Year"} <= set(raw.columns):
        raise ValueError(f"not an OWID grapher CSV: columns {list(raw.columns)}")
    value_cols = [
        c
        for c in raw.columns
        if c not in {"Entity", "Code", "Year"} and pd.api.types.is_numeric_dtype(raw[c])
    ]
    if len(value_cols) != 1:
        raise ValueError(f"expected one value column, found {value_cols}")
    df = raw.rename(columns={"Entity": "technology", "Year": "year", value_cols[0]: "adoption"})
    df = df.dropna(subset=["adoption"])
    if df["adoption"].max() > 1.5:
        df["adoption"] = df["adoption"] / 100.0
    # A few OWID series exceed 100% by rounding; clip instead of failing validation.
    df["adoption"] = df["adoption"].clip(0.0, 1.0)
    df["region"] = region
    return validate_curves(df)


def load_owid_us_households(
    cache_dir: Path | str = "data/raw", *, refresh: bool = False, path: Path | str | None = None
) -> pd.DataFrame:
    """Load the OWID US household adoption panel, downloading once into `cache_dir`.

    Pass `path` to use a manually downloaded CSV (e.g. behind a firewall).
    """
    if path is not None:
        return parse_owid_csv(path)
    dest = Path(cache_dir) / f"owid_{OWID_SLUG}.csv"
    if refresh or not dest.exists():
        download(OWID_URL, dest)
    return parse_owid_csv(dest)


def load_tidy_csv(path: Path | str) -> pd.DataFrame:
    return validate_curves(pd.read_csv(path))


def prepare_curves(
    df: pd.DataFrame,
    *,
    min_points: int = 8,
    min_peak: float = 0.1,
    truncate_at_peak: bool = True,
    max_gap: int | None = 10,
) -> pd.DataFrame:
    """Select curves suitable for fitting a first-adoption model.

    Truncation at the maximum removes decline phases (e.g. landlines, VCRs) that
    reflect substitution rather than diffusion. Series with large gaps are cut
    at the first gap so that interpolation does not dominate the fit.
    """
    kept = []
    for (_tech, _region), g in df.groupby(["technology", "region"], sort=True):
        g = g.sort_values("year")
        if max_gap is not None:
            gaps = np.diff(g["year"].to_numpy())
            big = np.flatnonzero(gaps > max_gap)
            if big.size:
                # Keep the longest contiguous block.
                bounds = np.concatenate([[0], big + 1, [len(g)]])
                lengths = np.diff(bounds)
                i = int(np.argmax(lengths))
                g = g.iloc[bounds[i] : bounds[i + 1]]
        if truncate_at_peak:
            g = g.iloc[: int(np.argmax(g["adoption"].to_numpy())) + 1]
        if len(g) >= min_points and g["adoption"].max() >= min_peak:
            kept.append(g)
    if not kept:
        return df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)
