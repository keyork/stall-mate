# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            _log.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        self._ensure_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        result = np.asarray(vec, dtype=np.float32).flatten()
        self._cache[text] = result
        return result

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            self._ensure_model()
            vecs = self._model.encode(
                uncached, normalize_embeddings=True, batch_size=32
            )
            for t, v in zip(uncached, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32).flatten()
        return np.stack([self._cache[t] for t in texts])
