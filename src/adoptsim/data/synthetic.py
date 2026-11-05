"""Synthetic curves with controllable text signal.

Serves as positive control for the spike: if the analog test cannot recover a
signal that was planted on purpose, a null result on real data is uninformative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adoptsim.models.bass import bass_cumulative

# Latent archetypes with distinct diffusion speed; (log p, log q) centres.
ARCHETYPES: dict[str, tuple[tuple[float, float], tuple[str, ...]]] = {
    "network": (
        (np.log(0.005), np.log(0.65)),
        ("network", "connectivity", "communication", "platform", "online", "messaging"),
    ),
    "appliance": (
        (np.log(0.02), np.log(0.30)),
        ("kitchen", "appliance", "electric", "domestic", "cleaning", "laundry"),
    ),
    "infrastructure": (
        (np.log(0.01), np.log(0.15)),
        ("grid", "plumbing", "installation", "building", "utility", "heating"),
    ),
    "media": (
        (np.log(0.03), np.log(0.45)),
        ("broadcast", "screen", "entertainment", "audio", "video", "receiver"),
    ),
}
FILLER = ("device", "system", "product", "used", "allows", "users", "provides", "technology")


def make_synthetic_dataset(
    n: int = 60,
    *,
    signal: float = 0.8,
    param_noise: float = 0.15,
    obs_noise: float = 0.01,
    n_years: int = 30,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (curves, descriptions, truth).

    `signal` is the probability that a description word is drawn from the
    technology's archetype vocabulary instead of generic filler.
    """
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    curve_rows, desc_rows, truth_rows = [], [], []
    for i in range(n):
        arch = names[i % len(names)]
        (mu_lp, mu_lq), vocab = ARCHETYPES[arch]
        lp = mu_lp + rng.normal(0, param_noise)
        lq = mu_lq + rng.normal(0, param_noise)
        p, q = float(np.exp(lp)), float(np.exp(lq))
        m = float(rng.uniform(0.6, 1.0))
        start = int(rng.integers(1900, 1990))
        offset = float(rng.uniform(0, 3))  # series starts shortly after launch
        t = np.arange(n_years, dtype=float)
        y = bass_cumulative(t + offset, p, q, m)
        y = np.clip(y + rng.normal(0, obs_noise, size=n_years), 0.0, 1.0)
        y = np.maximum.accumulate(y)
        tech = f"tech_{i:03d}"
        curve_rows += [(tech, "SYN", start + k, float(v)) for k, v in enumerate(y)]
        words = [
            str(rng.choice(vocab)) if rng.random() < signal else str(rng.choice(FILLER))
            for _ in range(12)
        ]
        desc_rows.append((tech, " ".join(words), "synthetic"))
        truth_rows.append((tech, arch, p, q, m))
    curves = pd.DataFrame(curve_rows, columns=["technology", "region", "year", "adoption"])
    descs = pd.DataFrame(desc_rows, columns=["technology", "description", "source"])
    truth = pd.DataFrame(truth_rows, columns=["technology", "archetype", "p", "q", "m"])
    return curves, descs, truth
