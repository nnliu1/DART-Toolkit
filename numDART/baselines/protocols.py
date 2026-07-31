from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from .contracts import CandidateRecord, ColumnPairRef, TableProfileRecord


class TableProfiler(Protocol):
    def profile_dataframe(
        self, table_id: str, frame: pd.DataFrame
    ) -> TableProfileRecord: ...


class CandidateRetriever(Protocol):
    def retrieve_cta(
        self, profile: TableProfileRecord, top_k: int
    ) -> Sequence[CandidateRecord]: ...

    def retrieve_cpa(
        self,
        profile: TableProfileRecord,
        pairs: Sequence[ColumnPairRef],
        top_k: int,
    ) -> Sequence[CandidateRecord]: ...

