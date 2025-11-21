from pathlib import Path
from time import perf_counter

from fastapi.responses import HTMLResponse

from api import create_app
from rag import RAGResponse, RetrievalMode

DEMO_API_KEY = "local-demo-only"
SAMPLE_DOCUMENT = Path(__file__).parent / "data" / "document2.txt"

DEMO_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kairo | Local Demo</title>
  <style>
    :root { color-scheme: dark; font: 16px/1.55 system-ui, sans-serif; }
    body { margin: 0; background: #101522; color: #e8edf8; }
    main { max-width: 880px; margin: 48px auto; padding: 0 24px; }
    h1 { margin-bottom: 4px; font-size: 2.1rem; }
    .muted { color: #aab5cb; }
    .notice { margin: 24px 0; padding: 16px; border: 1px solid #e0a14a;
      border-radius: 10px; background: #302616; }
    form, .card { padding: 20px; border: 1px solid #303a50;
      border-radius: 12px; background: #171e2d; }
    label { display: block; margin: 12px 0 6px; font-weight: 600; }
    textarea, select, input, button { box-sizing: border-box; width: 100%;
      padding: 11px 12px; border: 1px solid #46526b; border-radius: 8px;
      background: #101522; color: inherit; font: inherit; }
    textarea { min-height: 86px; resize: vertical; }
    button { width: auto; margin-top: 16px; padding: 10px 20px;
      border: 0; background: #7787ff; color: #101522; font-weight: 700;
      cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    .grid { display: grid; grid-template-columns: 1fr 140px; gap: 16px; }
    .results { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
      margin-top: 20px; }
    .card h2 { margin-top: 0; font-size: 1.1rem; }
    pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
    #status { min-height: 1.5em; margin-top: 12px; color: #aab5cb; }
    @media (max-width: 640px) {
      main { margin: 28px auto; }
      .grid, .results { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Kairo</h1>
    <div class="muted">Retrieval-augmented assistant · local demo</div>
    <div class="notice"><strong>Offline demo, not the live Gemini/Chroma system.</strong>
      This page sends requests to a localhost-only demo endpoint. It returns the
      sample file as context and a fixed explanatory response; no real model
      generation or vector search is performed. Latency is only local request
      handling time and is not a performance benchmark.</div>
    <form id="query-form">
      <label for="question">Ask a question</label>
      <textarea id="question" required>What does the sample document say about Kairo?</textarea>
      <div class="grid">
        <div>
          <label for="retrieval-mode">Retrieval variant</label>
          <select id="retrieval-mode">
            <option value="mmr">MMR</option>
            <option value="similarity">Similarity baseline</option>
          </select>
        </div>
        <div>
          <label for="k">Top K</label>
          <input id="k" type="number" min="1" max="20" value="3">
        </div>
      </div>
      <button id="submit" type="submit">Send demo query</button>
      <div id="status" role="status" aria-live="polite"></div>
    </form>
    <section class="results" aria-label="Query result">
      <article class="card">
        <h2>Response</h2>
        <pre id="answer">Submit a query to see the local request flow.</pre>
      </article>
      <article class="card">
        <h2>Retrieved context (sample file)</h2>
        <pre id="context">No request yet.</pre>
        <p class="muted" id="metadata"></p>
      </article>
    </section>
  </main>
  <script>
    const form = document.querySelector("#query-form");
    const button = document.querySelector("#submit");
    const status = document.querySelector("#status");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.textContent = "Sending request to the local Kairo API…";
      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer local-demo-only"
          },
          body: JSON.stringify({
            question: document.querySelector("#question").value,
            retrieval_mode: document.querySelector("#retrieval-mode").value,
            k: Number(document.querySelector("#k").value)
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        document.querySelector("#answer").textContent = data.answer;
        document.querySelector("#context").textContent =
          data.contexts.join("\\n\\n") || "No context found.";
        document.querySelector("#metadata").textContent =
          `${data.retrieval_mode.toUpperCase()} · local handler: ${data.latency_ms.toFixed(2)} ms`;
        status.textContent = "Request completed.";
      } catch (error) {
        status.textContent = `Request failed: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class OfflineDemoService:
    def __init__(self) -> None:
        if not SAMPLE_DOCUMENT.is_file():
            raise FileNotFoundError(f"Demo sample document is missing: {SAMPLE_DOCUMENT}")
        self._sample_text = SAMPLE_DOCUMENT.read_text(encoding="utf-8").strip()
        if not self._sample_text:
            raise ValueError(f"Demo sample document is empty: {SAMPLE_DOCUMENT}")

    def ask(
        self,
        question: str,
        retrieval_mode: RetrievalMode = "mmr",
        k: int = 3,
    ) -> RAGResponse:
        started_at = perf_counter()
        del question, k
        answer = (
            "This is a fixed offline-demo response. The request reached the "
            "Kairo API, but no LLM was called. See the sample-file context "
            "alongside this response."
        )
        contexts = [self._sample_text]
        return RAGResponse(
            answer=answer,
            contexts=contexts,
            latency_ms=(perf_counter() - started_at) * 1000,
            retrieval_mode=retrieval_mode,
        )


app = create_app(OfflineDemoService, api_key=DEMO_API_KEY)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_PAGE)
