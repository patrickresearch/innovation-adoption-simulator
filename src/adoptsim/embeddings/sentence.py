"""sentence-transformers backend (optional extra: `pip install adoptsim[embeddings]`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from adoptsim._typing import FloatArray
from adoptsim.embeddings.base import Encoder

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEncoder(Encoder):
    def __init__(
        self, model_name: str = DEFAULT_MODEL, *, device: str | None = None, batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.device = device or default_device()
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def name(self) -> str:
        return self.model_name

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("install the extra: pip install 'adoptsim[embeddings]'") from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        emb = self._model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(emb, dtype=float)
