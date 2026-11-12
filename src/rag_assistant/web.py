# MIT License
#
# Copyright (c) 2026 Harsh Vardhan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Local web UI for the RAG pipeline (optional, ``pip install ollama-rag-assistant[web]``).

A single-page chat that asks your indexed documents. Everything runs on
localhost against the same store and the same Ollama server as the CLI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .errors import RagAssistantError
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, embed
from .rag import answer as rag_answer
from .store import VectorStore

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RAG Assistant</title>
<style>
  :root { --bg:#0f0f13; --panel:#17171d; --line:#26262e; --text:#e8e8ec; --muted:#8b8b96; --accent:#7C3AED; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:Inter,system-ui,-apple-system,sans-serif; height:100vh; display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:10px; padding:14px 20px; border-bottom:1px solid var(--line); }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--accent); box-shadow:0 0 12px var(--accent); }
  header b { font-size:15px; } header span { color:var(--muted); font-size:12px; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:78%; padding:10px 14px; border-radius:12px; font-size:14px; line-height:1.55; white-space:pre-wrap; }
  .user { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:4px; }
  .bot { align-self:flex-start; background:var(--panel); border:1px solid var(--line); border-bottom-left-radius:4px; }
  .bot small { display:block; margin-top:8px; color:var(--muted); font-size:11px; }
  form { display:flex; gap:10px; padding:16px 20px; border-top:1px solid var(--line); }
  input { flex:1; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; color:var(--text); font-size:14px; outline:none; }
  input:focus { border-color:var(--accent); }
  button { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:0 20px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
</style>
</head>
<body>
<header><div class="dot"></div><div><b>RAG Assistant</b><br><span>local &amp; keyless · powered by Ollama</span></div></header>
<div id="log">
  <div class="msg bot">Ask me anything about the indexed documents.<small>index them first with `rag-assistant index &lt;path&gt;`</small></div>
</div>
<form id="chat">
  <input id="q" autocomplete="off" placeholder="Ask a question…" autofocus />
  <button id="send" type="submit">Ask</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('chat');
const input = document.getElementById('q');
const send = document.getElementById('send');
function add(text, cls, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  if (sources && sources.length) {
    const small = document.createElement('small');
    small.textContent = 'sources: ' + sources.join(', ');
    div.appendChild(small);
  }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  add(q, 'user');
  input.value = '';
  send.disabled = true;
  try {
    const res = await fetch('/api/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ question: q }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'request failed');
    add(data.answer, 'bot', data.sources);
  } catch (err) {
    add('Error: ' + err.message, 'bot');
  } finally {
    send.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""


class AskRequest(BaseModel):
    question: str


def create_app(store_path: str = 'rag-assistant.db', *, chat_model: str = DEFAULT_CHAT_MODEL,
               embed_model: str = DEFAULT_EMBED_MODEL) -> FastAPI:
    """Build the FastAPI app. ``store_path`` is the SQLite store to read."""
    app = FastAPI(title='RAG Assistant', version='1.0.0')
    store = VectorStore(store_path)

    @app.get('/', response_class=HTMLResponse)
    def index_page() -> str:
        return PAGE

    @app.get('/api/stats')
    def stats() -> dict:
        return store.stats()

    @app.post('/api/ask')
    def ask(request: AskRequest) -> dict:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail='question must not be empty')
        try:
            result = rag_answer(
                request.question,
                store=store,
                embed_fn=embed,
                chat_model=chat_model,
                embed_model=embed_model,
            )
        except RagAssistantError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {'answer': result.answer, 'sources': result.sources}

    return app
