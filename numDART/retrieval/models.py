"""Small immutable records shared by the CTA retrieval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ColumnManifestRecord:
    """Maps a DART query record back to its source table column."""

    record_id: str
    source_table_id: str
    column_index: int
    column_name: str
    source_archive: str
    source_member: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColumnManifestRecord":
        """Builds a record from decoded JSON."""
        return cls(
            record_id=str(value["record_id"]),
            source_table_id=str(value["source_table_id"]),
            column_index=int(value["column_index"]),
            column_name=str(value["column_name"]),
            source_archive=str(value["source_archive"]),
            source_member=str(value["source_member"]),
        )


@dataclass(frozen=True)
class CandidateExportSummary:
    """Counts produced while normalizing one DART retrieval artifact."""

    query_count: int
    candidate_count: int
    minimum_candidates_per_query: int
    maximum_candidates_per_query: int
