from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv


EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent"
)
CATALOG_PATH = Path("data/catalog.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
CATALOG_EMBEDDED_PATH = Path("data/catalog_embedded.json")


def embedding_text(assessment: dict[str, Any]) -> str:
    return (
        f"{assessment.get('name', '')}. {assessment.get('description', '')}. "
        f"Job levels: {assessment.get('job_levels', [])}. "
        f"Type: {assessment.get('test_type', '')}"
    )


def embed_document(text: str, api_key: str, embedding_model: str) -> list[float]:
    response = httpx.post(
        EMBED_URL.format(model=embedding_model),
        params={"key": api_key},
        json={
            "model": f"models/{embedding_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required in .env to generate embeddings.")
    if not embedding_model:
        raise RuntimeError("GEMINI_EMBEDDING_MODEL cannot be empty.")
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found at {CATALOG_PATH}.")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("data/catalog.json must contain a non-empty JSON array.")

    texts = [embedding_text(assessment) for assessment in catalog]
    embeddings: list[list[float]] = []
    for index, text in enumerate(texts, start=1):
        embeddings.append(embed_document(text, api_key, embedding_model))
        if index == 1 or index % 20 == 0 or index == len(texts):
            print(f"[embed_catalog] Embedded {index}/{len(texts)}")

    embedding_array = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
    embedding_array = embedding_array / np.maximum(norms, 1e-9)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_PATH), embedding_array)
    CATALOG_EMBEDDED_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[embed_catalog] Saved {EMBEDDINGS_PATH}")
    print(f"[embed_catalog] Saved {CATALOG_EMBEDDED_PATH}")


if __name__ == "__main__":
    main()
