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

"""A tiny, dependency-free vector store backed by SQLite.

Embeddings are stored as JSON arrays next to the chunk text. Search loads the
rows and ranks them in pure Python, which is perfectly adequate for the
thousands of chunks a personal knowledge base holds - and means the store
works everywhere SQLite works, with no native extensions.
"""

import json
import math
import sqlite3


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is zero."""
    if len(a) != len(b):
        raise ValueError('vectors must have the same length')
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """SQLite-backed store of text chunks and their embedding vectors."""

    def __init__(self, path: str = 'rag-assistant.db'):
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS chunks ('
                'id INTEGER PRIMARY KEY,'
                'source TEXT NOT NULL,'
                'idx INTEGER NOT NULL,'
                'text TEXT NOT NULL,'
                'vector TEXT NOT NULL)'
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)')

    def add_chunks(self, chunks, vectors) -> int:
        """Insert chunks and their vectors. Returns the number of rows written."""
        if len(chunks) != len(vectors):
            raise ValueError('chunks and vectors must have the same length')
        if not chunks:
            return 0
        rows = [(c.source, c.index, c.text, json.dumps(v)) for c, v in zip(chunks, vectors)]
        with sqlite3.connect(self.path) as conn:
            conn.executemany('INSERT INTO chunks(source, idx, text, vector) VALUES (?, ?, ?, ?)', rows)
        return len(rows)

    def search(self, query_vector: list, *, top_k: int = 4) -> list:
        """Return the ``top_k`` chunks most similar to ``query_vector``.

        Each hit is a dict with ``text``, ``source``, ``index`` and ``score``.
        """
        if top_k < 1:
            raise ValueError('top_k must be >= 1')
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute('SELECT id, source, idx, text, vector FROM chunks').fetchall()
        hits = []
        for _id, source, index, text, vector_json in rows:
            vector = json.loads(vector_json)
            score = cosine_similarity(query_vector, vector)
            hits.append({'text': text, 'source': source, 'index': index, 'score': score})
        hits.sort(key=lambda h: h['score'], reverse=True)
        return hits[:top_k]

    def stats(self) -> dict:
        """Snapshot of the store: chunk count, distinct sources, vector dimension."""
        with sqlite3.connect(self.path) as conn:
            chunks = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            sources = conn.execute('SELECT COUNT(DISTINCT source) FROM chunks').fetchone()[0]
            row = conn.execute('SELECT vector FROM chunks LIMIT 1').fetchone()
        dimension = len(json.loads(row[0])) if row else 0
        return {'chunks': chunks, 'sources': sources, 'dimension': dimension, 'path': self.path}

    def clear(self) -> int:
        """Delete every chunk. Returns the number of rows removed."""
        with sqlite3.connect(self.path) as conn:
            count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            conn.execute('DELETE FROM chunks')
        return count
