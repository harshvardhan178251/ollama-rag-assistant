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

"""Split raw text into overlapping chunks for indexing.

The splitter is structure-aware: whole paragraphs are packed into chunks up to
``size`` characters, any paragraph longer than ``size`` is hard-split into
non-overlapping windows, and - when ``overlap`` is enabled - up to ``overlap``
characters of the previous chunk are repeated at the start of the next one so
retrieval still finds text that straddles a boundary. Overlap is applied
exactly once, by the final pass, so no chunk ever exceeds ``size + overlap``
characters and every consecutive chunk pair shares exactly ``overlap``
characters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit: a piece of one source document."""

    text: str
    source: str
    index: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError('chunk text must not be empty')
        if self.index < 0:
            raise ValueError('chunk index must be >= 0')


def chunk_text(text: str, *, source: str = '', size: int = 600, overlap: int = 80) -> list:
    """Split ``text`` into overlapping chunks.

    Args:
        text: The raw document content.
        source: Name carried into every returned chunk (usually the file path).
        size: Maximum characters per window. Must be >= 50.
        overlap: Characters of the previous chunk repeated at the start of the
            next one. Must be < ``size``.

    Returns:
        A list of :class:`Chunk`, in document order. Empty input yields an
        empty list. Every returned chunk is at most ``size + overlap``
        characters long, and each consecutive pair shares exactly
        ``overlap`` characters.

    Raises:
        ValueError: If ``size`` is too small or ``overlap`` >= ``size``.
    """
    if size < 50:
        raise ValueError('size must be >= 50')
    if overlap >= size:
        raise ValueError('overlap must be smaller than size')
    if not text or not text.strip():
        return []

    pieces: list[str] = []
    current = ''
    for para in (p.strip() for p in text.split('\n\n')):
        if not para:
            continue
        candidate = f'{current}\n\n{para}' if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ''
        if len(para) <= size:
            current = para
            continue
        # Paragraph longer than size: slide non-overlapping windows across it.
        for start in range(0, len(para), size):
            pieces.append(para[start:start + size])
    if current:
        pieces.append(current)

    if overlap > 0 and len(pieces) > 1:
        overlapped = [pieces[0]]
        for piece in pieces[1:]:
            overlapped.append(overlapped[-1][-overlap:] + piece)
        pieces = overlapped

    return [Chunk(text=piece, source=source, index=i) for i, piece in enumerate(pieces)]
