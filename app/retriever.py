import json
import os
import numpy as np
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

catalog = []
embeddings = None


def init_retriever():
    global catalog, embeddings

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"catalog.json not found at {CATALOG_PATH}. Run scraper.py first."
        )

    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    print(f"[retriever] Loaded {len(catalog)} assessments. Building embeddings...")

    texts = [
        f"{a['name']}. {a.get('description', '')}. "
        f"Job levels: {a.get('job_levels', [])}. Type: {a.get('test_type', '')}"
        for a in catalog
    ]

    all_embeds = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        result = genai.embed_content(
            model="models/embedding-001",
            content=batch,
            task_type="retrieval_document",
        )
        all_embeds.extend(result["embedding"])

    embeddings = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    print(f"[retriever] Ready. Shape: {embeddings.shape}")


def search(query: str, top_k: int = 10) -> list[dict]:
    if embeddings is None or len(catalog) == 0:
        return []

    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    q = np.array(result["embedding"], dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-9)

    scores = embeddings @ q
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [catalog[i] for i in top_indices]
