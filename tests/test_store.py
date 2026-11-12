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
from rag_assistant.store import VectorStore, cosine_similarity


def chunk(text, index=0, source='doc.md'):
    return Chunk(text=text, source=source, index=index)


def test_cosine_similarity_of_identical_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / 'test.db'))


def test_add_and_search_returns_best_match_first(store):
    chunks = [chunk('apples', 0), chunk('oranges', 1), chunk('trucks', 2)]
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    assert store.add_chunks(chunks, vectors) == 3

    hits = store.search([1.0, 0.0], top_k=2)
    assert [h['text'] for h in hits] == ['apples', 'oranges']
    assert hits[0]['score'] > hits[1]['score'] > 0


def test_search_scores_are_ordered(store):
    assert store.add_chunks([chunk('dogs', 0), chunk('cats', 1)], [[1.0, 0.0], [0.0, 1.0]]) == 2
    hits = store.search([0.2, 0.8], top_k=2)
    assert hits[0]['text'] == 'cats'
    assert hits[0]['score'] >= hits[1]['score']


def test_search_includes_source_and_index(store):
    store.add_chunks([chunk('alpha', 7, 'notes.md')], [[1.0]])
    (hit,) = store.search([1.0], top_k=1)
    assert hit['source'] == 'notes.md'
    assert hit['index'] == 7


def test_add_chunks_requires_matching_lengths(store):
    with pytest.raises(ValueError):
        store.add_chunks([chunk('a')], [])
    assert store.add_chunks([], []) == 0


def test_search_requires_top_k_at_least_one(store):
    with pytest.raises(ValueError):
        store.search([1.0], top_k=0)


def test_stats_reports_chunks_sources_and_dimension(store):
    stats = store.stats()
    assert stats['chunks'] == 0
    assert stats['sources'] == 0
    assert stats['dimension'] == 0

    store.add_chunks([chunk('a', 0, 'a.md'), chunk('b', 0, 'b.md'), chunk('c', 1, 'a.md')],
                     [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    stats = store.stats()
    assert stats['chunks'] == 3
    assert stats['sources'] == 2
    assert stats['dimension'] == 2


def test_clear_removes_everything(store):
    store.add_chunks([chunk('x')], [[1.0]])
    assert store.clear() == 1
    assert store.stats()['chunks'] == 0
    assert store.clear() == 0
