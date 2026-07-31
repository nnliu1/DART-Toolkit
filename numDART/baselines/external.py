from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .contracts import AnnotationTask, CandidateRecord


class DartOutputAdapter:
    """Map DART output rows to numDART's ontology-independent contract."""

    def normalize(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        task: AnnotationTask,
    ) -> list[CandidateRecord]:
        return _normalize_rows(rows, task=task, adapter="dart")


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    task: AnnotationTask,
    adapter: str,
) -> list[CandidateRecord]:
    grouped_rank: dict[tuple[str, int, int | None], int] = defaultdict(int)
    records = []
    for row in rows:
        table_id = str(row["table_id"])
        source = int(row["column_index"])
        target_value = row.get("target_column_index")
        target = None if target_value is None else int(target_value)
        key = (table_id, source, target)
        grouped_rank[key] += 1
        metadata = dict(row.get("metadata", {}))
        metadata["adapter"] = adapter
        records.append(
            CandidateRecord(
                table_id=table_id,
                task=task,
                source_column=source,
                target_column=target,
                candidate_iri=str(row["candidate"]),
                rank=int(row.get("rank", grouped_rank[key])),
                retrieval_score=float(row["score"]),
                metadata=metadata,
            )
        )
    return records
