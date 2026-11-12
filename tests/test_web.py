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

fastapi = pytest.importorskip('fastapi')
pytest.importorskip('httpx')

from fastapi.testclient import TestClient

from rag_assistant.web import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(store_path=str(tmp_path / 'test.db'))
    return TestClient(app)


def test_index_page_serves_the_chat_ui(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'RAG Assistant' in response.text


def test_stats_endpoint_reports_an_empty_store(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    assert response.json()['chunks'] == 0


def test_ask_rejects_empty_question(client):
    response = client.post('/api/ask', json={'question': '   '})
    assert response.status_code == 400


def test_ask_reports_unavailable_index(client, monkeypatch):
    def fake_answer(*args, **kwargs):
        from rag_assistant.errors import RagAssistantError

        raise RagAssistantError('the knowledge base is empty')

    monkeypatch.setattr('rag_assistant.web.rag_answer', fake_answer)
    response = client.post('/api/ask', json={'question': 'anything?'})
    assert response.status_code == 503
    assert 'empty' in response.json()['detail']
