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

"""Command line interface: ``rag-assistant <command>``.

Commands:

* ``index``  - read a file or directory of .md/.txt documents, chunk, embed
  and store them.
* ``ask``    - answer a question from the indexed documents.
* ``stats``  - show what is in the store.
* ``clear``  - wipe the store.
* ``serve``  - start the local web UI (needs ``pip install ollama-rag-assistant[web]``).
"""

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .chunker import chunk_text
from .errors import RagAssistantError
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, embed, embed_batch
from .rag import answer as rag_answer
from .store import VectorStore

SUPPORTED_SUFFIXES = {'.md', '.markdown', '.txt', '.rst'}


def _read_documents(target: str) -> list:
    """Return ``[(source, text)]`` for every supported file under ``target``."""
    path = Path(target)
    if not path.exists():
        raise RagAssistantError(f'no such file or directory: {target}')
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise RagAssistantError(f'nothing to index under {target} (supported: {", ".join(sorted(SUPPORTED_SUFFIXES))})')
    documents = []
    for file in files:
        try:
            documents.append((str(file), file.read_text(encoding='utf-8', errors='replace')))
        except OSError as exc:
            print(f'warning: skipping {file} ({exc})', file=sys.stderr)
    return documents


def cmd_index(args: argparse.Namespace) -> int:
    store = VectorStore(args.db)
    documents = _read_documents(args.path)
    chunks = []
    for source, text in documents:
        chunks.extend(chunk_text(text, source=source, size=args.chunk_size, overlap=args.overlap))
    if not chunks:
        print('nothing to index: the documents were empty', file=sys.stderr)
        return 1

    print(f'embedding {len(chunks)} chunks from {len(documents)} document(s) via {args.embed_model} ...')
    vectors = embed_batch([c.text for c in chunks], model=args.embed_model)
    written = store.add_chunks(chunks, vectors)
    total = store.stats()['chunks']
    print(f'indexed {written} chunks (store now holds {total})')
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    store = VectorStore(args.db)
    result = rag_answer(
        args.question,
        store=store,
        embed_fn=embed,
        chat_model=args.model,
        embed_model=args.embed_model,
        top_k=args.top_k,
    )
    print(result.answer)
    print()
    print('sources:')
    for source in result.sources:
        print(f'  - {source}')
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = VectorStore(args.db)
    stats = store.stats()
    print(f'db:        {stats["path"]}')
    print(f'chunks:    {stats["chunks"]}')
    print(f'sources:   {stats["sources"]}')
    print(f'vectors:   {stats["dimension"]}-dimensional')
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    store = VectorStore(args.db)
    removed = store.clear()
    print(f'removed {removed} chunks')
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from .web import create_app
    except ImportError:
        print('the web UI needs the optional dependencies: pip install "ollama-rag-assistant[web]"', file=sys.stderr)
        return 1
    app = create_app(store_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='rag-assistant',
        description='Ask questions about your own documents with a fully local, keyless Ollama RAG pipeline.',
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--db', default=os.environ.get('RAG_DB', 'rag-assistant.db'),
                        help='path of the SQLite vector store (default: %(default)s, env RAG_DB)')
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    p = sub.add_parser('index', help='index a file or directory of documents')
    p.add_argument('path', help='file or directory to index')
    p.add_argument('--chunk-size', type=int, default=600)
    p.add_argument('--overlap', type=int, default=80)
    p.add_argument('--embed-model', default=DEFAULT_EMBED_MODEL)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser('ask', help='answer a question from the indexed documents')
    p.add_argument('question')
    p.add_argument('--model', default=DEFAULT_CHAT_MODEL, help='chat model (default: %(default)s)')
    p.add_argument('--embed-model', default=DEFAULT_EMBED_MODEL)
    p.add_argument('--top-k', type=int, default=4)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser('stats', help='show what is in the store')
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser('clear', help='wipe the store')
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser('serve', help='start the local web UI (needs the [web] extra)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except RagAssistantError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    except (KeyboardInterrupt,):
        print('\ninterrupted', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
