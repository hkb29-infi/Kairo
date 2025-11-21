import html

from fastapi import Request
from fastapi.responses import HTMLResponse

from api import create_app

DEMO_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="kairo-api-key" content="__KAIRO_API_KEY__">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kairo | Local RAG</title>
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
    <div class="muted">Retrieval-augmented assistant · local mode</div>
    <div class="notice"><strong>Local-first mode.</strong>
      This app performs real local Chroma vector search using deterministic
      lexical embeddings, then returns an extractive answer from the retrieved
      documents. It requires no Gemini API key or cloud account. For semantic
      generation and Gemini embeddings, switch to cloud mode after configuring
      credentials. Local embeddings are a simple baseline, not a semantic model.</div>
    <form id="query-form">
      <label for="question">Ask a question</label>
      <textarea id="question" required>Who is Bernard?</textarea>
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
      <button id="submit" type="submit">Ask Kairo</button>
      <div id="status" role="status" aria-live="polite"></div>
    </form>
    <section class="results" aria-label="Query result">
      <article class="card">
        <h2>Answer</h2>
        <pre id="answer">Ask a question about your indexed documents.</pre>
      </article>
      <article class="card">
        <h2>Retrieved context</h2>
        <pre id="context">No documents retrieved yet.</pre>
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
      status.textContent = "Searching the local Chroma index…";
      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${document.querySelector('meta[name="kairo-api-key"]').content}`
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
          data.contexts.join("\\n\\n") || "No relevant documents found.";
        document.querySelector("#metadata").textContent =
          `${data.retrieval_mode.toUpperCase()} · local end-to-end: ${data.latency_ms.toFixed(2)} ms`;
        status.textContent = "Local query completed.";
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

app = create_app()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page(request: Request) -> HTMLResponse:
    api_key = html.escape(request.app.state.api_key, quote=True)
    return HTMLResponse(DEMO_PAGE.replace("__KAIRO_API_KEY__", api_key))
