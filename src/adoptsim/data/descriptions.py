"""Technology descriptions and leakage control.

Present-day encyclopedia text describes how successful a technology became.
Feeding that into a pre-launch model leaks the target, so descriptions are
truncated to their defining sentences and outcome vocabulary is masked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from adoptsim.data.loaders import USER_AGENT
from adoptsim.data.schema import validate_descriptions

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Words that encode diffusion outcome or timing rather than what the technology is.
OUTCOME_TERMS = (
    "popular",
    "popularity",
    "widely",
    "widespread",
    "common",
    "commonly",
    "ubiquitous",
    "mainstream",
    "standard",
    "success",
    "successful",
    "dominant",
    "adopted",
    "adoption",
    "household",
    "households",
    "million",
    "millions",
    "billion",
    "billions",
    "majority",
    "most",
    "obsolete",
    "replaced",
    "declined",
    "became",
    "early",
    "modern",
    "today",
    "century",
    "decade",
    "decades",
)
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})s?\b")
_PERCENT = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|percent|per cent)")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_OUTCOME = re.compile(r"\b(" + "|".join(OUTCOME_TERMS) + r")\b", flags=re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PAREN = re.compile(r"\s*\([^)]*\)")


def first_sentences(text: str, n: int) -> str:
    return " ".join(_SENTENCE.split(text.strip())[:n])


def mask_leakage(text: str, *, n_sentences: int | None = 2) -> str:
    """Strip timing and outcome cues; keep what the technology is and does."""
    out = _PAREN.sub("", text)
    if n_sentences is not None:
        out = first_sentences(out, n_sentences)
    out = _PERCENT.sub(" ", out)
    out = _YEAR.sub(" ", out)
    out = _NUMBER.sub(" ", out)
    out = _OUTCOME.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def fetch_wikipedia_summary(title: str, *, timeout: float = 20.0) -> str:
    url = WIKI_SUMMARY_URL.format(title=quote(title.replace(" ", "_"), safe=""))
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return str(resp.json().get("extract", ""))


def load_title_overrides(path: Path | str | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["technology"], df["wikipedia_title"], strict=True))


def load_descriptions(
    technologies: Iterable[str],
    *,
    cache_path: Path | str = "data/interim/descriptions.csv",
    title_overrides: Path | str | None = "data/external/wikipedia_titles.csv",
    manual_path: Path | str | None = "data/external/descriptions_manual.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return raw (unmasked) descriptions, one per technology.

    Precedence: manual file > cache > Wikipedia. Failed lookups are returned with
    an empty description so they can be curated, rather than silently dropped.
    """
    techs = list(dict.fromkeys(technologies))
    rows: dict[str, tuple[str, str]] = {}

    if manual_path is not None and Path(manual_path).exists():
        manual = validate_descriptions(pd.read_csv(manual_path))
        rows.update(
            {
                str(t): (str(d), str(s))
                for t, d, s in zip(
                    manual["technology"], manual["description"], manual["source"], strict=True
                )
            }
        )

    cache = Path(cache_path)
    if cache.exists() and not refresh:
        cached = validate_descriptions(pd.read_csv(cache).fillna(""))
        for t, d, s in zip(
            cached["technology"], cached["description"], cached["source"], strict=True
        ):
            if str(t) not in rows and d:
                rows[str(t)] = (str(d), str(s))

    overrides = load_title_overrides(title_overrides)
    for tech in techs:
        if tech in rows:
            continue
        title = overrides.get(tech, tech)
        try:
            rows[tech] = (fetch_wikipedia_summary(title), f"wikipedia:{title}")
        except requests.RequestException:
            rows[tech] = ("", f"missing:{title}")

    out = pd.DataFrame(
        [(t, *rows[t]) for t in techs], columns=["technology", "description", "source"]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return validate_descriptions(out)
