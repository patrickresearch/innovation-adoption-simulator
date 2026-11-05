"""Bag-of-words baseline encoder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from adoptsim._typing import FloatArray
from adoptsim.embeddings.base import Encoder


class TfidfEncoder(Encoder):
    """TF-IDF fitted on the texts it encodes.

    Fitting on the full corpus is transductive but label-free, so it does not
    leak diffusion outcomes into leave-one-out evaluation.
    """

    cacheable = False

    def __init__(self, **vectorizer_kwargs: object) -> None:
        kwargs: dict[str, object] = {"stop_words": "english", "sublinear_tf": True}
        kwargs.update(vectorizer_kwargs)
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "tfidf"

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        vec = TfidfVectorizer(**self._kwargs)
        return np.asarray(vec.fit_transform(texts).toarray(), dtype=float)
