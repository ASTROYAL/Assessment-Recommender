# SHL Assessment Recommender

SHL Assessment Recommender is a stateless conversational FastAPI service that helps hiring managers and recruiters choose assessments from the official SHL product catalog. It uses a local FAISS vector index over `data/catalog.json` to ground each conversation turn in real catalog entries before asking Gemini for a JSON-only response, and automatically falls back to a local recommender if Gemini is unavailable.

The service stores no sessions and uses no database. Clients send the full conversation history to `/chat`, and the API returns a strict response schema containing a conversational reply, zero to ten assessment recommendations, and an `end_of_conversation` flag.

## How to run locally

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables.

```bash
cp .env.example .env
```

The app tries Gemini first when `AGENT_PROVIDER=gemini`. If Gemini fails because of quota, network, or response-format issues, it falls back to the local catalog recommender. To force local-only mode, set `AGENT_PROVIDER=local`.

```env
AGENT_PROVIDER=gemini
GEMINI_API_KEY=your_real_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash-lite
```

4. Run the scraper if `data/catalog.json` is missing or stale.

```bash
python scraper.py
```

5. Generate catalog embeddings once if `data/embeddings.npy` and `data/catalog_embedded.json` are missing.

```bash
python embed_catalog.py
```

Commit both generated files before deploying so Render can load them without rebuilding embeddings on startup.

6. Start the API.

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment variables

The app loads a local `.env` file from the project root. `AGENT_PROVIDER=gemini` calls the Gemini API and requires `GEMINI_API_KEY`; if that call fails, the app uses local retrieval-based recommendations. `AGENT_PROVIDER=local` uses only the local catalog retriever. `GEMINI_MODEL` defaults to `gemini-2.0-flash-lite`.

## Run the scraper

The scraper reads the SHL product catalog, keeps only the `Individual Test Solutions` table, visits each assessment detail page, and writes the result to `data/catalog.json`.

```bash
python scraper.py
```

It prints progress as it runs and waits one second after each request to avoid rate limiting.

## Generate embeddings

The retriever loads `data/embeddings.npy` and `data/catalog_embedded.json` when they exist. Generate them once locally:

```bash
python embed_catalog.py
```

This script calls Gemini's single `embedContent` endpoint once per assessment, normalizes the vectors, and writes both output files under `data/`. Commit those files so hosted startup does not burn quota by embedding the whole catalog again.

## API examples

Health check:

```bash
curl http://localhost:8000/health
```

Chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "I need assessments for a mid-level data analyst who needs SQL, numerical reasoning, and attention to detail."
      }
    ]
  }'
```

Expected response shape:

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "string"
    }
  ],
  "end_of_conversation": false
}
```
