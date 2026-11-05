from pathlib import Path

import pandas as pd
import pytest

from adoptsim.data import (
    load_descriptions,
    make_synthetic_dataset,
    mask_leakage,
    parse_owid_csv,
    prepare_curves,
    validate_curves,
)

FIXTURE = Path(__file__).parent / "fixtures" / "owid_sample.csv"


def test_parse_owid_rescales_and_clips() -> None:
    df = parse_owid_csv(FIXTURE)
    assert set(df.columns) == {"technology", "region", "year", "adoption"}
    assert df["adoption"].between(0, 1).all()
    assert df.loc[df.technology == "Alpha", "adoption"].max() == pytest.approx(0.82)
    assert df["adoption"].isna().sum() == 0


def test_prepare_truncates_decline_gaps_and_short_series() -> None:
    df = prepare_curves(parse_owid_csv(FIXTURE), min_points=5, min_peak=0.1)
    alpha = df[df.technology == "Alpha"]
    assert alpha["year"].max() == 1958  # decline after the peak removed
    beta = df[df.technology == "Beta"]
    assert beta["year"].max() == 1988  # isolated 2010 point split off by the gap rule
    assert "Gamma" not in set(df.technology)  # never reaches min_peak


def test_validate_rejects_bad_values() -> None:
    bad = pd.DataFrame({"technology": ["a"], "region": ["x"], "year": [2000], "adoption": [1.5]})
    with pytest.raises(ValueError):
        validate_curves(bad)


def test_mask_leakage_removes_outcome_cues() -> None:
    text = (
        "A microwave oven (commonly referred to as a microwave) is an electric oven that heats "
        "food by exposing it to microwave radiation. It became popular in the 1970s and is used "
        "in 90% of households. Millions were sold."
    )
    out = mask_leakage(text, n_sentences=2)
    for cue in ("1970", "90", "popular", "households", "became", "Millions", "commonly"):
        assert cue not in out
    assert "electric oven" in out


def test_load_descriptions_prefers_manual(tmp_path: Path) -> None:
    manual = tmp_path / "manual.csv"
    pd.DataFrame(
        {"technology": ["Alpha"], "description": ["A test device."], "source": ["manual"]}
    ).to_csv(manual, index=False)
    out = load_descriptions(
        ["Alpha"], cache_path=tmp_path / "cache.csv", title_overrides=None, manual_path=manual
    )
    assert out.loc[0, "description"] == "A test device."
    assert (tmp_path / "cache.csv").exists()


def test_synthetic_dataset_shapes() -> None:
    curves, descs, truth = make_synthetic_dataset(n=12, n_years=20, seed=0)
    validate_curves(curves)
    assert len(descs) == len(truth) == 12
    assert curves.groupby("technology").size().eq(20).all()


@pytest.mark.network
def test_wikipedia_lookup(tmp_path: Path) -> None:
    out = load_descriptions(
        ["Microwave oven"], cache_path=tmp_path / "c.csv", title_overrides=None, manual_path=None
    )
    assert "microwave" in out.loc[0, "description"].lower()
