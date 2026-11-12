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

"""Retrieval-augmented generation over the indexed chunks.

Answers are built in three steps: embed the question, pull the most similar
chunks from the store, then hand them to a local chat model with a strict
"answer only from the context" instruction. The reply names its sources and
the similarity of each used chunk, so the answer is auditable.
"""

from dataclasses import dataclass

from .errors import RagAssistantError
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL

SYSTEM_PROMPT = (
    'You are a precise assistant. Answer the question using ONLY the context '
    'provided. If the context does not contain the answer, reply exactly: '
    '"I do not know - the indexed documents do not cover this." '
    'Cite the context sources inline as [1], [2] etc.'
)

QUESTION_TEMPLATE = """Context:
{context}

Question: {question}

Answer using only the context above."""


def build_context(hits: list) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for position, hit in enumerate(hits, start=1):
        lines.append(f'[{position}] (source: {hit["source"]} | similarity {hit["score"]:.3f})\n{hit["text"]}')
    return '\n\n'.join(lines)


@dataclass
class Answer:
    """A grounded answer plus the evidence behind it."""

    answer: str
    sources: list
    hits: list


def answer(question: str, *, store, embed_fn, chat_fn=None, chat_model: str = DEFAULT_CHAT_MODEL,
           embed_model: str = DEFAULT_EMBED_MODEL, top_k: int = 4, top_k_min_score: float = 0.0) -> Answer:
    """Answer ``question`` from the indexed documents.

    Args:
        question: What the user asked.
        store: A :class:`rag_assistant.store.VectorStore`.
        embed_fn: Callable ``(text, model=...)`` returning a vector.
        chat_fn: Callable ``(prompt, model=..., system=...)`` returning text.
            Defaults to :func:`rag_assistant.ollama.chat`.
        chat_model / embed_model: Ollama model names.
        top_k: Maximum number of chunks to retrieve.
        top_k_min_score: Drop retrieved chunks below this cosine similarity.

    Raises:
        ValueError: On an empty question or top_k < 1.
        RagAssistantError: If the store is empty.
    """
    if not question or not question.strip():
        raise ValueError('question must not be empty')
    if top_k < 1:
        raise ValueError('top_k must be >= 1')

    if store.stats()['chunks'] == 0:
        raise RagAssistantError('the knowledge base is empty - index some documents first (`rag-assistant index <path>`)')

    query_vector = embed_fn(question, model=embed_model)
    hits = store.search(query_vector, top_k=top_k)
    if top_k_min_score > 0:
        hits = [h for h in hits if h['score'] >= top_k_min_score]
    if not hits:
        raise RagAssistantError('no retrieved chunk met the similarity threshold')

    if chat_fn is None:
        from .ollama import chat as chat_fn  # local import keeps rag.py importable offline

    prompt = QUESTION_TEMPLATE.format(context=build_context(hits), question=question)
    reply = chat_fn(prompt, model=chat_model, system=SYSTEM_PROMPT)

    seen = set()
    sources = []
    for hit in hits:
        if hit['source'] not in seen:
            seen.add(hit['source'])
            sources.append(hit['source'])

    return Answer(answer=reply, sources=sources, hits=hits)
