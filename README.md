# Kairo

Kairo is a document-grounded assistant built with LangChain, ChromaDB Cloud,
and Gemini. It provides an interactive CLI and a stateless HTTP API. The
default retrieval strategy is maximum marginal relevance (MMR); similarity
retrieval is also available as a comparison baseline.

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set these environment variables (for example in a local `.env` file):

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini generation and embedding API key |
| `CHROMADB_API_KEY` | ChromaDB Cloud API key |
| `TENANT_ID` | ChromaDB Cloud tenant |
| `DATABASE_ID` | ChromaDB Cloud database |
| `KAIRO_API_KEY` | Required bearer key for the HTTP API |
| `GEMINI_MODEL` | Optional generation model; defaults to `gemini-2.5-flash` |
| `GEMINI_JUDGE_MODEL` | Optional evaluation judge model; defaults to `gemini-2.5-flash` |

Place `.txt` files under `data/`, then ingest them:

```powershell
python ingest.py
```

Ingestion streams documents into Chroma in batches of 100 and derives stable
IDs from each document's relative path and content, so rerunning ingestion
updates existing records rather than creating duplicates.

Start the interactive chatbot with:

```powershell
python main.py
```

## HTTP API and scaling

Run the API with multiple worker processes:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

Send `POST /query` with a JSON body such as:

```json
{
  "question": "What does the handbook say about leave?",
  "retrieval_mode": "mmr",
  "k": 3
}
```

Include `Authorization: Bearer <KAIRO_API_KEY>` with API requests. The API
refuses to start if no key is configured. Use TLS and configure deployment-level
rate limits before exposing it beyond a trusted network.

Use `retrieval_mode: "similarity"` for the baseline arm. The response includes
the selected variant, retrieved contexts, answer, and end-to-end latency in
milliseconds. Each worker initializes its own service/client while using the
shared ChromaDB Cloud collection; the API keeps no per-user session state.
Deploying additional workers is an implementation path for horizontal scaling,
not a benchmark proving throughput or reliability under production load.

## RAGAS evaluation

Create a **hand-reviewed** JSONL file with one reference answer per query. Use
edge cases from the actual document collection; do not use generated or
unverified answers as ground truth. Each non-empty line must have this shape:

```json
{"question":"What is the documented leave approval process?","reference":"The reference answer verified against the source documents."}
```

Run the same golden set against similarity and MMR:

```powershell
python evaluate_rag.py --dataset evaluation/golden.jsonl --output evaluation/results.json
```

The evaluator requires at least 50 examples by default. It uses RAGAS with
Gemini-as-judge to score context precision, faithfulness, and answer relevancy.
The reported composite is this project's arithmetic mean of those three
metrics; relative improvement is calculated against the similarity baseline.
The report includes the dataset SHA-256, per-example metric scores, and mean,
p50, and p95 end-to-end latency. Evaluation makes Gemini API calls and incurs
their associated cost. Review the golden answers and per-example scores before
using an aggregate score as a résumé claim.

The repository contains the evaluation harness, not a 50-query golden dataset
or a measured 40% result. Do not claim a 40% gain unless a run on your real,
reviewed dataset produces and supports that number. Similarly, the existing
code does not establish a 10,000-document production corpus or sub-200 ms
end-to-end latency; validate those separately against the deployed collection
and load profile before using those figures.

Run the local test suite with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
