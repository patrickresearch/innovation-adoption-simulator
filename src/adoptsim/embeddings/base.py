"""Encoder interface and on-disk embedding cache."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from adoptsim._typing import FloatArray


def l2_normalize(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.asarray(x / np.where(norms == 0, 1.0, norms), dtype=float)


class Encoder(ABC):
    """Maps texts to L2-normalised vectors, so cosine similarity is a dot product."""

    #: False for corpus-dependent encoders (e.g. TF-IDF) whose vectors change with the corpus.
    cacheable: bool = True

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> FloatArray: ...

    def encode(self, texts: Sequence[str]) -> FloatArray:
        return l2_normalize(np.asarray(self._encode(list(texts)), dtype=float))


class CachedEncoder(Encoder):
    """Caches per-text vectors keyed by encoder name and text hash.

    Embedding is the only GPU-bound step; caching lets all downstream modelling
    run on CPU and keeps experiments reproducible across machines.
    """

    def __init__(self, inner: Encoder, cache_dir: Path | str = ".cache/embeddings") -> None:
        if not inner.cacheable:
            raise ValueError(f"{inner.name} is corpus-dependent and cannot be cached per text")
        self.inner = inner
        safe = inner.name.replace("/", "__")
        self.path = Path(cache_dir) / f"{safe}.npz"

    @property
    def name(self) -> str:
        return self.inner.name

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _load(self) -> dict[str, FloatArray]:
        if not self.path.exists():
            return {}
        with np.load(self.path) as z:
            return {k: z[k] for k in z.files}

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        store = self._load()
        keys = [self._key(t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in store]
        if todo:
            new = self.inner.encode([texts[i] for i in todo])
            for i, vec in zip(todo, new, strict=True):
                store[keys[i]] = vec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # stubs type **kwargs against savez' own keyword args; a plain mapping avoids that
            arrays: dict[str, Any] = dict(store)
            np.savez(self.path, **arrays)
        return np.stack([store[k] for k in keys])
