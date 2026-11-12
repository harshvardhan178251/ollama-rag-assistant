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

from rag_assistant.chunker import Chunk, chunk_text


def test_empty_input_returns_no_chunks():
    assert chunk_text('') == []
    assert chunk_text('   \n\n  ') == []


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        chunk_text('hello', size=10)
    with pytest.raises(ValueError):
        chunk_text('hello', size=100, overlap=100)


def test_short_text_yields_one_chunk_in_order():
    chunks = chunk_text('First paragraph.\n\nSecond paragraph.', source='doc.md', size=600, overlap=0)
    assert [c.index for c in chunks] == [0]
    assert chunks[0].source == 'doc.md'
    assert 'First paragraph.' in chunks[0].text and 'Second paragraph.' in chunks[0].text


def test_paragraphs_are_packed_up_to_size():
    text = '\n\n'.join(['a' * 100, 'b' * 100, 'c' * 100])
    chunks = chunk_text(text, size=220, overlap=0)
    # a+b fit (201 chars + separator), c spills into the next chunk
    assert len(chunks) == 2
    assert chunks[0].text == 'a' * 100 + '\n\n' + 'b' * 100
    assert chunks[1].text == 'c' * 100


def test_long_paragraph_is_split_with_overlap():
    para = 'x' * 1000
    chunks = chunk_text(para, size=240, overlap=40)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 240 + 40 for c in chunks)
    # consecutive chunks share exactly `overlap` characters
    assert chunks[0].text[-40:] == chunks[1].text[:40]
    assert chunks[-1].text.endswith('x' * 10)


def test_chunk_index_is_sequential_and_source_is_carried():
    text = '\n\n'.join(['alpha ' * 30, 'beta ' * 30, 'gamma ' * 30])
    chunks = chunk_text(text, source='notes.md', size=120, overlap=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == 'notes.md' for c in chunks)


def test_overlap_repeats_tail_of_previous_chunk():
    text = 'one two three four five six seven eight nine ten\n\n' + 'word ' * 60
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    for i in range(1, len(chunks)):
        assert chunks[i - 1].text[-20:] == chunks[i].text[:20]


def test_chunk_rejects_empty_text():
    with pytest.raises(ValueError):
        Chunk(text='   ', source='x', index=0)
    with pytest.raises(ValueError):
        Chunk(text='ok', source='x', index=-1)
