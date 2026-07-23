"""High-precision column filters for Wikidata-derived CTA supervision."""

from __future__ import annotations

import re
from typing import List, Mapping, Sequence


_GLOBAL_BLOCKED_TYPE_QIDS = {"Q13406463", "Q20136634"}
_SEASON_HEADER_EVIDENCE = re.compile(
    r"\b(season|year|records?)\b|\b(?:18|19|20|21)\d{2}\b",
    re.IGNORECASE,
)


_LEGACY_QID_BLOCKS = (
    (re.compile(r"\b(soccer|football)\s+(team|club|squad)\b", re.I), {"Q6256"}),
    (re.compile(r"\bcountry\b", re.I), {"Q5"}),
    (re.compile(r"\byear\b", re.I), {"Q5", "Q6256"}),
)

_LABEL_CONFLICTS = (
    (
        re.compile(r"\b(nat\.?|nation|nationality|country)\b", re.I),
        re.compile(
            r"federation|delegation|team|club|governing body|"
            r"sports organi[sz]ation|nation at\b",
            re.I,
        ),
    ),
    (
        re.compile(r"^(gold|silver|bronze)$", re.I),
        re.compile(r"country|state|delegation|team", re.I),
    ),
    (re.compile(r"^round$", re.I), re.compile(r"season", re.I)),
    (re.compile(r"^school$", re.I), re.compile(r"season", re.I)),
    (re.compile(r"\b(team|club)\b", re.I), re.compile(r"season", re.I)),
    (
        re.compile(r"\b(competition|tournament|event)\b", re.I),
        re.compile(r"season", re.I),
    ),
    (
        re.compile(r"\b(opponent|opponents|conference)\b", re.I),
        re.compile(r"season", re.I),
    ),
    (
        re.compile(r"\b(stadium|site|venue)\b", re.I),
        re.compile(r"school", re.I),
    ),
    (
        re.compile(r"\b(cities|towns)\b", re.I),
        re.compile(r"island|region", re.I),
    ),
    (
        re.compile(r"^state$", re.I),
        re.compile(r"colonial power|historical period", re.I),
    ),
)


def normalize_header(header: str) -> str:
    """Remove leading Wikipedia navigation boilerplate from a header."""
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if len(lines) >= 3 and [line.casefold() for line in lines[:3]] == [
        "v", "t", "e"
    ]:
        lines = lines[3:]
    elif lines and re.sub(r"\s+", "", lines[0]).casefold() == "vte":
        lines = lines[1:]
    return "\n".join(lines).strip()


def header_is_informative(header: str) -> bool:
    """Return whether a header contains learnable alphabetic semantics."""
    cleaned = re.sub(r"\[[^]]*\]", "", normalize_header(header))
    return any(character.isalpha() for character in cleaned)


def header_type_is_blocked(
    header: str,
    type_label: str,
    type_qid: str = "",
) -> bool:
    """Reject high-confidence conflicts between a header and inferred type."""
    if type_qid in _GLOBAL_BLOCKED_TYPE_QIDS:
        return True
    if re.search(r"\bseason\b", type_label, re.IGNORECASE):
        if not _SEASON_HEADER_EVIDENCE.search(header):
            return True
    for header_pattern, blocked_qids in _LEGACY_QID_BLOCKS:
        if header_pattern.search(header) and type_qid in blocked_qids:
            return True
    return any(
        header_pattern.search(header) and label_pattern.search(type_label)
        for header_pattern, label_pattern in _LABEL_CONFLICTS
    )


def filter_compatible_types(
    header: str,
    type_qids: Sequence[str],
    type_labels: Mapping[str, str],
) -> List[str]:
    """Filter every positive type through the shared compatibility policy."""
    return [
        qid for qid in type_qids
        if not header_type_is_blocked(
            header,
            type_labels.get(qid, ""),
            qid,
        )
    ]
