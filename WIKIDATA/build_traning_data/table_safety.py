"""Safety boundary for table methods implemented by foreign runtimes."""

from __future__ import annotations

import re
from typing import Iterable, Optional


_PROCESS_CONTROL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def safe_get_cell(table, row: int, col: int) -> Optional[str]:
    """Return normalized cell text, or None for malformed foreign-runtime data."""
    try:
        cell = table.get_cell(row, col)
        if cell is None:
            return ""
        value = getattr(cell, "value", None)
        text = getattr(value, "text", None)
        if text is None:
            text = str(cell)
        return str(text or "").strip()
    except BaseException as exc:
        if isinstance(exc, _PROCESS_CONTROL_EXCEPTIONS):
            raise
        return None


_CITATION_MENTION = re.compile(
    r"^\[(?:\d+|[a-z]|note\s*\d+)\]$", re.IGNORECASE
)
_SCORE_MENTION = re.compile(
    r"^\s*\d+\s*[-\u2013\u2014]\s*\d+(?:\s*\([a-z]+\))?\s*$",
    re.IGNORECASE,
)


def _alphanumeric_length(text: str) -> int:
    return sum(character.isalnum() for character in text)


def select_primary_link(links: Iterable[object], cell_text: str):
    """Select the unique longest valid text-bearing Wikidata link."""
    text_length = len(cell_text)
    candidates = []
    for link in links:
        qid = getattr(link, "wikidata_id", None)
        start = getattr(link, "start", None)
        end = getattr(link, "end", None)
        if not qid or not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > text_length:
            continue
        mention = cell_text[start:end].strip()
        url = str(getattr(link, "wikipedia_url", "") or "").lower()
        if "#cite_note-" in url or "#cite_ref-" in url:
            continue
        if _CITATION_MENTION.fullmatch(mention) or _SCORE_MENTION.fullmatch(mention):
            continue
        if start != 0 or not any(character.isalpha() for character in mention):
            continue
        mention_size = _alphanumeric_length(mention)
        cell_size = _alphanumeric_length(cell_text)
        if mention_size < 2 or cell_size == 0 or mention_size / cell_size <= 0.5:
            continue
        candidates.append((end - start, str(qid), link))

    if not candidates:
        return None
    max_length = max(length for length, _, _ in candidates)
    longest = [item for item in candidates if item[0] == max_length]
    longest_qids = {qid for _, qid, _ in longest}
    if len(longest_qids) != 1:
        return None
    return longest[0][2]
