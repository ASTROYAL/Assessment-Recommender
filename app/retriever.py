from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_EMBEDDING_MODEL}:embedContent"
)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.npy"
CATALOG_EMBEDDED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "catalog_embedded.json"
)

_catalog: list[dict[str, Any]] = []
_embeddings: Any = None
STOPWORDS = {
    "a",
    "an",
    "and",
    "assessment",
    "assessments",
    "for",
    "hire",
    "hiring",
    "i",
    "need",
    "of",
    "the",
    "to",
    "with",
}


def _embedding_text(assessment: dict[str, Any]) -> str:
    return (
        f"{assessment.get('name', '')}. {assessment.get('description', '')}. "
        f"Job levels: {assessment.get('job_levels', [])}. "
        f"Type: {assessment.get('test_type', '')}"
    )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to compute embeddings.")

    results = []
    for text in texts:
        body = {
            "model": f"models/{GEMINI_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        resp = httpx.post(
            EMBED_URL,
            params={"key": GEMINI_API_KEY},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        results.append(resp.json()["embedding"]["values"])
    return results


def _embed_query(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to embed search queries.")

    body = {
        "model": f"models/{GEMINI_EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    resp = httpx.post(
        EMBED_URL,
        params={"key": GEMINI_API_KEY},
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z0-9+#.]+", query.lower())
    return [term for term in terms if len(term) > 1 and term not in STOPWORDS]


def _lexical_search(query: str, top_k: int) -> list[dict]:
    terms = _query_terms(query)
    if not terms:
        return [dict(item) for item in _catalog[:top_k]]

    scored: list[tuple[int, int, dict]] = []
    for index, assessment in enumerate(_catalog):
        name = str(assessment.get("name", "")).lower()
        description = str(assessment.get("description", "")).lower()
        job_levels = " ".join(str(level).lower() for level in assessment.get("job_levels", []))
        test_type = str(assessment.get("test_type", "")).lower()
        searchable = f"{name} {description} {job_levels} {test_type}"
        score = 0
        for term in terms:
            if term in name:
                score += 5
            if term in description:
                score += 2
            if term in job_levels or term == test_type:
                score += 1
            if term in searchable:
                score += 1
        if score > 0:
            scored.append((score, -index, assessment))

    scored.sort(reverse=True)
    return [dict(assessment) for _, _, assessment in scored[:top_k]]


def init_retriever() -> None:
    global _catalog, _embeddings

    if EMBEDDINGS_PATH.exists() and CATALOG_EMBEDDED_PATH.exists():
        print("[retriever] Loading pre-computed embeddings...")
        with open(CATALOG_EMBEDDED_PATH, "r", encoding="utf-8") as f:
            _catalog = json.load(f)
        _embeddings = np.load(str(EMBEDDINGS_PATH))
        if len(_catalog) != int(_embeddings.shape[0]):
            raise ValueError(
                "Pre-computed embeddings count does not match embedded catalog count."
            )
        print(f"[retriever] Ready. Shape: {_embeddings.shape}")
        return

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"catalog.json not found at {CATALOG_PATH}. Run scraper.py first."
        )

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    print(f"[retriever] Loaded {len(_catalog)} assessments. Building embeddings...")

    texts = [_embedding_text(assessment) for assessment in _catalog]

    all_embeds: list[list[float]] = []
    for i, text in enumerate(texts):
        all_embeds.extend(_embed_batch([text]))
        if i % 50 == 0:
            print(f"[retriever] Embedded {i}/{len(texts)}...")

    _embeddings = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    _embeddings = _embeddings / np.maximum(norms, 1e-9)

    np.save(str(EMBEDDINGS_PATH), _embeddings)
    with open(CATALOG_EMBEDDED_PATH, "w", encoding="utf-8") as f:
        json.dump(_catalog, f, ensure_ascii=False)

    print(f"[retriever] Ready. Shape: {_embeddings.shape}")


# Alias so main.py works without changes
load_index = init_retriever


def search(query: str, top_k: int = 10) -> list[dict]:
    if _embeddings is None or not _catalog:
        return []

    try:
        q = np.array(_embed_query(query), dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = _embeddings @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [dict(_catalog[int(i)]) for i in top_indices]
    except Exception as exc:
        print(f"[retriever] Query embedding failed; using lexical fallback: {exc}")
        return _lexical_search(query, top_k)
