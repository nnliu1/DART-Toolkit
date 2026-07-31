from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from .contracts import (
    ColumnProfileRecord,
    NumericSummary,
    TableProfileRecord,
)


class Td2kgProfilingAdapter:
    """Normalize td2kg profiles so numDART does not depend on td2kg models."""

    def __init__(self, profiling_src: str | Path) -> None:
        self.profiling_src = Path(profiling_src).resolve()
        if not (self.profiling_src / "table_profiling").is_dir():
            raise FileNotFoundError(
                f"table_profiling package not found under {self.profiling_src}"
            )

    def profile_dataframe(
        self, table_id: str, frame: pd.DataFrame
    ) -> TableProfileRecord:
        self._make_importable()
        from table_profiling.column_profiler import profile_all_columns
        from table_profiling.config import EnergyConfig, Level1Config

        external_profiles = profile_all_columns(
            frame,
            level1_cfg=Level1Config(),
            energy_cfg=EnergyConfig(enabled=False),
        )
        columns = tuple(
            self._normalize_column(index, external)
            for index, external in enumerate(external_profiles)
        )
        return TableProfileRecord(
            table_id=table_id,
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=columns,
        )

    def _make_importable(self) -> None:
        source = str(self.profiling_src)
        if source not in sys.path:
            sys.path.insert(0, source)

    @staticmethod
    def _normalize_column(index: int, external: object) -> ColumnProfileRecord:
        scan = external.fast_scan
        numeric = None
        if external.numeric is not None:
            stats = external.numeric.stats
            distribution = external.numeric.distribution
            numeric = NumericSummary(
                minimum=float(stats.min),
                maximum=float(stats.max),
                mean=float(stats.mean),
                standard_deviation=float(stats.std),
                percentiles=dict(stats.percentiles),
                unit=external.numeric.unit,
                is_discrete=bool(distribution.is_discrete),
                is_multimodal=bool(distribution.is_multimodal),
                outlier_rate=float(distribution.outlier_rate),
            )
        return ColumnProfileRecord(
            index=index,
            name=str(external.column_name),
            path=str(scan.path.value),
            dtype=str(scan.dtype_raw),
            null_rate=float(scan.null_rate),
            cardinality=int(scan.cardinality),
            uniqueness=float(scan.uniqueness_score),
            numeric=numeric,
        )

