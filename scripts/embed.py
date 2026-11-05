"""Batch-embed a descriptions CSV and write vectors to the shared cache.

Intended for the GPU container (docker/Dockerfile.gpu); runs locally as well.
"""

from __future__ import annotations

import argparse

import pandas as pd

from adoptsim.data import mask_leakage
from adoptsim.embeddings import CachedEncoder, SentenceTransformerEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interim/descriptions.csv")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cache-dir", default="data/interim/embeddings")
    ap.add_argument("--no-mask", action="store_true", help="embed raw text (leakage study)")
    args = ap.parse_args()

    df = pd.read_csv(args.input).fillna("")
    texts = df["description"].tolist()
    if not args.no_mask:
        texts = [mask_leakage(t) for t in texts]
    enc = CachedEncoder(SentenceTransformerEncoder(args.model), cache_dir=args.cache_dir)
    emb = enc.encode(texts)
    print(f"{emb.shape[0]} texts embedded with {args.model} -> {enc.path}")


if __name__ == "__main__":
    main()
