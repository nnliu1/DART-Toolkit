"""Pure helpers for constructing multi-positive supervision masks."""
from __future__ import annotations

from typing import List, Optional


def build_positive_mask_rows(
    pos_counts: List[int],
    positive_qid_sets: List[List[str]],
    positive_qids_flat: Optional[List[str]] = None,
) -> List[List[bool]]:
    total = sum(pos_counts)
    if positive_qids_flat is not None and len(positive_qids_flat) != total:
        raise ValueError(
            "sum(pos_counts) must equal len(positive_qids_flat)"
        )
    if len(pos_counts) != len(positive_qid_sets):
        raise ValueError(
            "pos_counts and positive_qid_sets must have one entry per query"
        )
    rows = [[False] * total for _ in pos_counts]
    offset = 0
    for row, count in enumerate(pos_counts):
        for col in range(offset, offset + count):
            rows[row][col] = True
        offset += count
    if positive_qids_flat is not None:
        for row, qids in enumerate(positive_qid_sets):
            qid_set = set(qids)
            for col, qid in enumerate(positive_qids_flat):
                if qid in qid_set:
                    rows[row][col] = True
    return rows
