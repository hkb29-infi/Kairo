# Kairo

Kairo is a document-grounded assistant with a **local-first** setup. Local
mode uses persistent ChromaDB, deterministic lexical embeddings, MMR or
similarity retrieval, and an extractive answerer. It runs without cloud
accounts, model downloads, or API calls. The optional cloud mode uses Gemini
embeddings and generation with ChromaDB Cloud.

## Run locally

From the repository root, copy the local configuration template and install
the local/test dependencies:

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements-dev.txt
```

The example config selects local mode and a development-only API key. Keep
the server bound to `127.0.0.1`; do not reuse that key or expose this
development setup to a network.

Put `.txt` documents under `data/` and index them:

```powershell
python ingest.py
```

Ingestion splits documents into overlapping chunks and upserts them into a
persistent local Chroma database under `.kairo/chroma/`. The default chunk size
is 1,000 characters with 150 characters of overlap. Re-running ingestion
updates matching chunks rather than creating duplicates. The current checked-in
sample is just a smoke-test document; replace or supplement it with your own
corpus.

Start the browser demo, which calls the same real local RAG service as the API:

```powershell
python -m uvicorn demo:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and try asking
“Who is Bernard?” against the checked-in sample. Or use the CLI:

```powershell
python main.py
```

The local embedding function is deterministic feature hashing over words,
word pairs, and character n-grams. It is a small, model-free lexical baseline;
it does **not** provide the semantic understanding of Gemini embeddings. The
local answerer extracts relevant sentences from retrieved passages rather
than generating new text. This makes the local setup useful for development
and an honest offline baseline, not a substitute for evaluating the cloud
version.

The HTTP API requires `KAIRO_API_KEY` and the development config provides a
local-only example value. The browser page receives this key to make localhost
requests, so use the included demo only on loopback. Configure TLS, real
credentials, and deployment-level security before any network exposure.

## Optional Gemini and Chroma Cloud mode

Install the additional cloud and RAGAS dependencies:

```powershell
python -m pip install -r requirements-cloud.txt
```

Set `KAIRO_MODE=cloud` in `.env` and configure:

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini generation and embedding API key |
| `CHROMADB_API_KEY` | ChromaDB Cloud API key |
| `TENANT_ID` | ChromaDB Cloud tenant |
| `DATABASE_ID` | ChromaDB Cloud database |
| `KAIRO_API_KEY` | Bearer key required by the HTTP API |
| `GEMINI_MODEL` | Optional generation model (default `gemini-2.5-flash`) |
| `GEMINI_JUDGE_MODEL` | Optional RAGAS judge model (default `gemini-2.5-flash`) |

Re-run `python ingest.py` after selecting cloud mode to index your documents
into the cloud collection. Start the API directly with:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The API's `POST /query` accepts a JSON body such as:

```json
{
  "question": "What does the handbook say about leave?",
  "retrieval_mode": "mmr",
  "k": 3
}
```

Include an `Authorization: Bearer <KAIRO_API_KEY>` header. Choose
`retrieval_mode: "similarity"` for the baseline. The response includes the
answer, retrieved contexts, selected variant, and end-to-end latency. More
workers can be run against the shared Chroma collection, but this code alone
does not establish production reliability or throughput.

## RAGAS evaluation (cloud mode)

Create a hand-reviewed JSONL dataset with at least 50 real questions and
reference answers based on your corpus. Each non-empty line has this form:

```json
{"question":"What is the documented leave approval process?","reference":"The answer verified against the source documents."}
```

With cloud mode and credentials configured, run:

```powershell
python evaluate_rag.py --dataset evaluation/golden.jsonl --output evaluation/results.json
```

The harness compares similarity with MMR on the same questions and uses Gemini
as a RAGAS judge for context precision, faithfulness, and answer relevancy. It
reports the composite (the arithmetic mean of those three metrics), relative
change, per-example scores, dataset hash, and mean/p50/p95 latency. Evaluation
calls Gemini and may incur API costs.
improvement unless an actual run on your reviewed dataset demonstrates it.

The testing for the mertrics is still to be done for proper improvement tracking

## Tests

```powershell
python -m pytest -q
```
