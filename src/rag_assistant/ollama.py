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

"""Minimal client for the local Ollama HTTP API.

No SDK dependency: the API is plain JSON over HTTP, so ``urllib`` from the
standard library is all this needs. Any failure to reach Ollama raises
:class:`OllamaUnavailableError` with instructions on how to start it.
"""

import json
import urllib.error
import urllib.request

from .errors import OllamaUnavailableError, RagAssistantError

DEFAULT_BASE_URL = 'http://localhost:11434'
DEFAULT_EMBED_MODEL = 'nomic-embed-text'
DEFAULT_CHAT_MODEL = 'qwen2.5:3b'


def _post(path: str, payload: dict, *, timeout: int, base_url: str = DEFAULT_BASE_URL) -> dict:
    url = base_url.rstrip('/') + path
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace').strip()
        raise RagAssistantError(f'Ollama answered {exc.code} for {path}: {detail[:200]}') from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise OllamaUnavailableError(
            f'Cannot reach Ollama at {base_url}. Start it with `ollama serve` '
            f'(install from https://ollama.com), then run `ollama pull {DEFAULT_EMBED_MODEL}`.'
        ) from exc


def embed(text: str, *, model: str = DEFAULT_EMBED_MODEL, base_url: str = DEFAULT_BASE_URL) -> list:
    """Embed one string. Returns the embedding as a list of floats."""
    if not text.strip():
        raise ValueError('cannot embed empty text')
    data = _post('/api/embeddings', {'model': model, 'prompt': text}, timeout=60, base_url=base_url)
    vector = data.get('embedding')
    if not isinstance(vector, list) or not vector:
        raise RagAssistantError(f'Ollama returned no embedding for model {model!r}. Is the model pulled?')
    return [float(x) for x in vector]


def embed_batch(texts, *, model: str = DEFAULT_EMBED_MODEL, base_url: str = DEFAULT_BASE_URL) -> list:
    """Embed many strings, one API call each."""
    return [embed(text, model=model, base_url=base_url) for text in texts]


def chat(
    prompt: str,
    *,
    model: str = DEFAULT_CHAT_MODEL,
    system: str | None = None,
    temperature: float = 0.1,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Send a chat request and return the assistant's reply as text."""
    if not prompt.strip():
        raise ValueError('prompt must not be empty')
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    data = _post(
        '/api/chat',
        {'model': model, 'messages': messages, 'stream': False, 'options': {'temperature': temperature}},
        timeout=300,
        base_url=base_url,
    )
    reply = (data.get('message') or {}).get('content', '')
    if not reply.strip():
        raise RagAssistantError(f'Ollama returned an empty reply for model {model!r}. Is the model pulled?')
    return reply.strip()
