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

import pytest

from rag_assistant.chunker import Chunk
from rag_assistant.errors import RagAssistantError
from rag_assistant.rag import Answer, answer, build_context
from rag_assistant.store import VectorStore


def fake_embed(text, *, model):
    """Deterministic stand-in: apple/banana/car each get their own axis."""
    t = text.lower()
    if 'apple' in t:
        return [1.0, 0.0, 0.0]
    if 'banana' in t:
        return [0.0, 1.0, 0.0]
    if 'car' in t or 'truck' in t:
        return [0.0, 0.0, 1.0]
    return [0.5, 0.5, 0.5]


def fake_chat(prompt, *, model, system=None):
    assert 'Context:' in prompt
    return 'Banana is a yellow fruit [1].'


@pytest.fixture
def store(tmp_path):
    store = VectorStore(str(tmp_path / 'test.db'))
    store.add_chunks(
        [Chunk(text='apples are red fruit', source='a.md', index=0),
         Chunk(text='bananas are yellow fruit', source='b.md', index=0),
         Chunk(text='cars have wheels', source='c.md', index=0)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return store


def test_build_context_numbers_the_hits():
    hits = [
        {'text': 'alpha', 'source': 'a.md', 'score': 0.9},
        {'text': 'beta', 'source': 'b.md', 'score': 0.5},
    ]
    context = build_context(hits)
    assert '[1] (source: a.md' in context
    assert '[2] (source: b.md' in context
    assert 'alpha' in context and 'beta' in context


def test_answer_returns_grounded_reply_with_sources(store):
    result = answer('What is a banana?', store=store, embed_fn=fake_embed, chat_fn=fake_chat)
    assert isinstance(result, Answer)
    assert result.answer == 'Banana is a yellow fruit [1].'
    # Only three chunks exist, so all three come back - best first.
    assert result.sources == ['b.md', 'a.md', 'c.md']
    assert result.hits[0]['source'] == 'b.md'


def test_answer_requires_non_empty_question(store):
    with pytest.raises(ValueError):
        answer('   ', store=store, embed_fn=fake_embed, chat_fn=fake_chat)


def test_answer_requires_top_k_at_least_one(store):
    with pytest.raises(ValueError):
        answer('question', store=store, embed_fn=fake_embed, chat_fn=fake_chat, top_k=0)


def test_answer_fails_clearly_when_store_is_empty(tmp_path):
    empty = VectorStore(str(tmp_path / 'empty.db'))
    with pytest.raises(RagAssistantError, match='empty'):
        answer('question', store=empty, embed_fn=fake_embed, chat_fn=fake_chat)


def test_answer_filters_by_min_score(store):
    result = answer('apple', store=store, embed_fn=fake_embed, chat_fn=fake_chat, top_k=3, top_k_min_score=0.99)
    assert len(result.hits) == 1
    assert result.hits[0]['source'] == 'a.md'


def test_answer_lists_each_source_once(store):
    def chat_repeat(prompt, *, model, system=None):
        return 'ok'

    store.add_chunks(
        [Chunk(text='banana again', source='b.md', index=1)],
        [[0.0, 1.0, 0.0]],
    )
    result = answer('banana?', store=store, embed_fn=fake_embed, chat_fn=chat_repeat, top_k=2)
    assert result.sources == ['b.md']


def test_question_far_from_every_axis_still_answers(store):
    result = answer('something else entirely', store=store, embed_fn=fake_embed, chat_fn=fake_chat, top_k=2)
    assert result.hits
    assert result.answer == 'Banana is a yellow fruit [1].'
