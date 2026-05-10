"""Local sentence-transformers embedder. Free, runs on CPU."""
from __future__ import annotations

import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(arr, dtype=np.float32)
