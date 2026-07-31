from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class AnnotationTask(str, Enum):
    CTA = "cta"
    CPA = "cpa"


@dataclass(frozen=True)
class ColumnPairRef:
    table_id: str
    source_column: int
    target_column: int

    def __post_init__(self) -> None:
        _validate_table_and_index(self.table_id, self.source_column)
        if self.target_column < 0:
            raise ValueError("target_column must be non-negative")
        if self.source_column == self.target_column:
            raise ValueError("source_column and target_column must differ")


@dataclass(frozen=True)
class NumericSummary:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    percentiles: Mapping[str, float] = field(default_factory=dict)
    unit: str | None = None
    is_discrete: bool | None = None
    is_multimodal: bool | None = None
    outlier_rate: float | None = None


@dataclass(frozen=True)
class ColumnProfileRecord:
    index: int
    name: str
    path: str
    dtype: str
    null_rate: float
    cardinality: int
    uniqueness: float
    numeric: NumericSummary | None = None


@dataclass(frozen=True)
class TableProfileRecord:
    table_id: str
    row_count: int
    column_count: int
    columns: tuple[ColumnProfileRecord, ...]

    def __post_init__(self) -> None:
        if not self.table_id:
            raise ValueError("table_id must not be empty")
        if self.row_count < 0 or self.column_count < 0:
            raise ValueError("table dimensions must be non-negative")
        if self.column_count != len(self.columns):
            raise ValueError("column_count must equal the number of column profiles")


@dataclass(frozen=True)
class CandidateRecord:
    table_id: str
    task: AnnotationTask
    source_column: int
    candidate_iri: str
    rank: int
    retrieval_score: float
    target_column: int | None = None
    reranker_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_table_and_index(self.table_id, self.source_column)
        if not isinstance(self.task, AnnotationTask):
            object.__setattr__(self, "task", AnnotationTask(self.task))
        if not self.candidate_iri:
            raise ValueError("candidate_iri must not be empty")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        _validate_finite("retrieval_score", self.retrieval_score)
        if self.reranker_score is not None:
            _validate_finite("reranker_score", self.reranker_score)
        if self.task is AnnotationTask.CPA and self.target_column is None:
            raise ValueError("target_column is required for CPA candidates")
        if self.task is AnnotationTask.CTA and self.target_column is not None:
            raise ValueError("target_column must be omitted for CTA candidates")
        if self.target_column is not None:
            if self.target_column < 0:
                raise ValueError("target_column must be non-negative")
            if self.target_column == self.source_column:
                raise ValueError("source_column and target_column must differ")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["task"] = self.task.value
        result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRecord:
        return cls(
            table_id=str(value["table_id"]),
            task=AnnotationTask(value["task"]),
            source_column=int(value["source_column"]),
            target_column=(
                None
                if value.get("target_column") is None
                else int(value["target_column"])
            ),
            candidate_iri=str(value["candidate_iri"]),
            rank=int(value["rank"]),
            retrieval_score=float(value["retrieval_score"]),
            reranker_score=(
                None
                if value.get("reranker_score") is None
                else float(value["reranker_score"])
            ),
            metadata=dict(value.get("metadata", {})),
        )


def _validate_table_and_index(table_id: str, column_index: int) -> None:
    if not table_id:
        raise ValueError("table_id must not be empty")
    if column_index < 0:
        raise ValueError("source_column must be non-negative")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

