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

import rag_assistant.cli as cli
from rag_assistant.cli import build_parser, main


def _fake_embed_batch(texts, *, model):
    """Deterministic vectors: apple axis is x, banana axis is y."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        if 'apple' in lowered and 'banana' not in lowered:
            vectors.append([1.0, 0.0])
        elif 'banana' in lowered:
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.5, 0.5])
    return vectors


def _fake_embed(text, *, model):
    return [0.0, 1.0]


def _fake_chat(prompt, *, model, system=None):
    return 'Bananas are yellow [1].'


@pytest.fixture
def docs(tmp_path):
    directory = tmp_path / 'docs'
    directory.mkdir()
    (directory / 'fruit.md').write_text('Apples are red. Bananas are yellow.', encoding='utf-8')
    return directory


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(['--version'])
    assert excinfo.value.code == 0


def test_help_flag_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(['--help'])
    assert excinfo.value.code == 0


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert 'COMMAND' in capsys.readouterr().out


def test_index_and_ask_end_to_end(docs, tmp_path, monkeypatch, capsys):
    db = str(tmp_path / 'test.db')
    monkeypatch.setattr(cli, 'embed_batch', _fake_embed_batch)

    assert main(['--db', db, 'index', str(docs)]) == 0
    out = capsys.readouterr().out
    assert 'indexed' in out

    monkeypatch.setattr(cli, 'embed', _fake_embed)
    monkeypatch.setattr('rag_assistant.ollama.chat', _fake_chat)

    assert main(['--db', db, 'ask', 'what color are bananas?']) == 0
    out = capsys.readouterr().out
    assert 'Bananas are yellow [1].' in out
    assert 'fruit.md' in out


def test_stats_and_clear_on_indexed_store(docs, tmp_path, monkeypatch, capsys):
    db = str(tmp_path / 'test.db')
    monkeypatch.setattr(cli, 'embed_batch', _fake_embed_batch)
    main(['--db', db, 'index', str(docs)])
    capsys.readouterr()

    assert main(['--db', db, 'stats']) == 0
    out = capsys.readouterr().out
    assert 'chunks:' in out and 'sources:' in out

    assert main(['--db', db, 'clear']) == 0
    assert 'removed' in capsys.readouterr().out
    assert main(['--db', db, 'clear']) == 0
    assert 'removed 0' in capsys.readouterr().out


def test_index_fails_for_missing_path(tmp_path, capsys):
    assert main(['--db', str(tmp_path / 'x.db'), 'index', str(tmp_path / 'nope')]) == 1
    assert 'no such file or directory' in capsys.readouterr().err


def test_index_fails_when_nothing_supported(tmp_path, capsys):
    directory = tmp_path / 'empty'
    directory.mkdir()
    (directory / 'data.json').write_text('{}', encoding='utf-8')
    assert main(['--db', str(tmp_path / 'x.db'), 'index', str(directory)]) == 1
    assert 'nothing to index' in capsys.readouterr().err
