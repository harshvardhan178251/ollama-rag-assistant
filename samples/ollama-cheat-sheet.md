# Ollama Cheat Sheet

Ollama runs large language models locally. Install it from https://ollama.com
and the `ollama` command becomes available on your path.

## Common commands

- `ollama serve` - start the local server (usually automatic).
- `ollama list` - list pulled models.
- `ollama pull qwen2.5:3b` - download a chat model (~2 GB).
- `ollama pull nomic-embed-text` - download the embedding model (~275 MB).
- `ollama run qwen2.5:3b` - chat interactively in the terminal.
- `ollama rm <model>` - delete a model.

## The HTTP API

Ollama exposes a JSON HTTP API on `http://localhost:11434`.

- `GET /api/tags` - JSON list of installed models.
- `POST /api/embeddings` - body `{"model": "...", "prompt": "..."}`, returns
  `{"embedding": [float, ...]}`.
- `POST /api/chat` - body `{"model": "...", "messages": [...], "stream": false}`,
  returns `{"message": {"role": "assistant", "content": "..."}}`.
- `POST /api/generate` - single prompt completion.

## Embedding models

`nomic-embed-text` produces 768-dimensional vectors optimized for semantic
similarity. It is the default embedding model for RAG pipelines and runs in
well under a second per chunk on a modern CPU.

## Chat models

`qwen2.5:3b` is a strong small model: fast on CPU, surprisingly capable at
reading-comprehension style questions. Larger models such as `qwen3:8b` give
stronger answers on complex sources at the cost of latency.
