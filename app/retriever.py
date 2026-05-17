from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

_catalog: list[dict[str, Any]] = []
_embeddings: Any = None


def _embed_batch(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def init_retriever() -> None:
    global _catalog, _embeddings

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"catalog.json not found at {CATALOG_PATH}. Run scraper.py first."
        )

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    print(f"[retriever] Loaded {len(_catalog)} assessments. Building embeddings...")

    texts = [
        f"{a.get('name', '')}. {a.get('description', '')}. "
        f"Job levels: {a.get('job_levels', [])}. Type: {a.get('test_type', '')}"
        for a in _catalog
    ]

    all_embeds: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i: i + 100]
        all_embeds.extend(_embed_batch(batch))

    _embeddings = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    _embeddings = _embeddings / np.maximum(norms, 1e-9)
    print(f"[retriever] Ready. Shape: {_embeddings.shape}")


# Alias so main.py works without changes
load_index = init_retriever


def search(query: str, top_k: int = 10) -> list[dict]:
    if _embeddings is None or not _catalog:
        return []

    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    q = np.array(result["embedding"], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-9)

    scores = _embeddings @ q
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [dict(_catalog[int(i)]) for i in top_indices]
