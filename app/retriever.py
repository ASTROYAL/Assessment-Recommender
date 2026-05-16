from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/opt/render/project/src/.cache"


CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"

_catalog: list[dict[str, Any]] = []
_index: Any | None = None
_model: Any | None = None
_index_lock = threading.RLock()


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(f"Unable to load retrieval dependencies: {exc}") from exc

    return faiss, np, SentenceTransformer


def _load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        raise RuntimeError(
            f"Catalog file not found at {CATALOG_PATH}. Run `python scraper.py` "
            "from the project root before starting the API."
        )

    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Catalog file at {CATALOG_PATH} is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Catalog file at {CATALOG_PATH} must contain a non-empty JSON array.")

    normalized_catalog: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        normalized_catalog.append(
            {
                "name": str(entry.get("name", "")).strip(),
                "url": str(entry.get("url", "")).strip(),
                "description": str(entry.get("description", "")).strip(),
                "test_type": str(entry.get("test_type", "")).strip(),
                "job_levels": entry.get("job_levels") if isinstance(entry.get("job_levels"), list) else [],
                "languages": entry.get("languages") if isinstance(entry.get("languages"), list) else [],
                "duration": entry.get("duration") if entry.get("duration") is not None else None,
            }
        )

    normalized_catalog = [entry for entry in normalized_catalog if entry["name"] and entry["url"]]
    if not normalized_catalog:
        raise RuntimeError(f"Catalog file at {CATALOG_PATH} does not contain usable assessment entries.")

    return normalized_catalog


def _embedding_text(entry: dict[str, Any]) -> str:
    name = entry.get("name", "")
    description = entry.get("description", "")
    job_levels = entry.get("job_levels", [])
    test_type = entry.get("test_type", "")
    return f"{name}. {description}. Job levels: {job_levels}. Type: {test_type}"


def load_index() -> None:
    global _catalog, _index, _model

    with _index_lock:
        if _index is not None and _model is not None and _catalog:
            return

        faiss, np, SentenceTransformer = _load_dependencies()
        catalog = _load_catalog()
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [_embedding_text(entry) for entry in catalog]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise RuntimeError("Unable to build FAISS index because no embeddings were produced.")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        _catalog = catalog
        _model = model
        _index = index


def search(query: str, top_k: int = 10) -> list[dict]:
    if not query or not query.strip() or top_k <= 0:
        return []

    load_index()

    with _index_lock:
        if _index is None or _model is None or not _catalog:
            raise RuntimeError("Retriever index is not initialized.")

        _, np, _ = _load_dependencies()
        limit = min(top_k, len(_catalog))
        query_embedding = _model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        _, indices = _index.search(query_embedding, limit)

        results: list[dict] = []
        for index_position in indices[0]:
            if 0 <= int(index_position) < len(_catalog):
                results.append(dict(_catalog[int(index_position)]))

        return results
