"""Normalize trusted native DART retrieval output for numDART."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
from typing import Any

from numDART.baselines.contracts import AnnotationTask, CandidateRecord
from numDART.baselines.io import write_candidates_jsonl

from .models import CandidateExportSummary, ColumnManifestRecord


def export_dart_candidates(
    retrieval_pkl: str | Path,
    manifest_path: str | Path,
    ontology_path: str | Path,
    output_path: str | Path,
    run_id: str,
) -> CandidateExportSummary:
    """Exports native DART Top-K results as normalized candidate JSONL.

    The pickle file must be produced by the trusted local DART job. Pickle
    files from untrusted sources must never be passed to this function.

    Args:
        retrieval_pkl: Native DART retrieval pickle.
        manifest_path: Column identity manifest created during preparation.
        ontology_path: DART ontology JSON used for retrieval.
        output_path: Destination candidate JSONL.
        run_id: Stable identifier for the retrieval run.

    Returns:
        Aggregate candidate counts.

    Raises:
        ValueError: If any native result violates the output contract.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    manifest = _load_manifest(Path(manifest_path))
    ontology_iris = _load_ontology_iris(Path(ontology_path))
    results = _load_results(Path(retrieval_pkl))

    candidates: list[CandidateRecord] = []
    counts: list[int] = []
    seen_queries: set[str] = set()
    for result in results:
        record_id = str(result.get("table_id", ""))
        if record_id in seen_queries:
            raise ValueError(f"Duplicate DART result for record_id: {record_id}")
        seen_queries.add(record_id)
        try:
            source = manifest[record_id]
        except KeyError as error:
            raise ValueError(f"Missing manifest record: {record_id}") from error

        uris = result.get("cand_uris")
        labels = result.get("cand_labels")
        scores = result.get("cand_scores")
        if not all(isinstance(values, list) for values in (uris, labels, scores)):
            raise ValueError(f"Missing candidate arrays for record_id: {record_id}")
        if len({len(uris), len(labels), len(scores)}) != 1:
            raise ValueError(f"Unequal candidate array lengths for record_id: {record_id}")
        if len(set(uris)) != len(uris):
            raise ValueError(f"Duplicate candidate URI for record_id: {record_id}")

        numeric_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) for score in numeric_scores):
            raise ValueError(f"Non-finite candidate score for record_id: {record_id}")
        if numeric_scores != sorted(numeric_scores, reverse=True):
            raise ValueError(f"Candidate scores are not descending for record_id: {record_id}")

        for rank, (iri, label, score) in enumerate(
            zip(uris, labels, numeric_scores), start=1
        ):
            if iri not in ontology_iris:
                raise ValueError(f"Candidate IRI is absent from ontology: {iri}")
            candidates.append(
                CandidateRecord(
                    table_id=source.source_table_id,
                    task=AnnotationTask.CTA,
                    source_column=source.column_index,
                    candidate_iri=str(iri),
                    rank=rank,
                    retrieval_score=score,
                    metadata={
                        "adapter": "dart",
                        "candidate_label": str(label),
                        "column_name": source.column_name,
                        "dart_record_id": record_id,
                        "margin": float(result.get("margin", 0.0)),
                        "query_text": str(result.get("column_text", "")),
                        "run_id": run_id,
                    },
                )
            )
        counts.append(len(uris))

    if set(manifest) != seen_queries:
        missing = sorted(set(manifest) - seen_queries)
        raise ValueError(f"DART output is missing {len(missing)} manifest records")
    write_candidates_jsonl(output_path, candidates)
    return CandidateExportSummary(
        query_count=len(results),
        candidate_count=len(candidates),
        minimum_candidates_per_query=min(counts, default=0),
        maximum_candidates_per_query=max(counts, default=0),
    )


def _load_manifest(path: Path) -> dict[str, ColumnManifestRecord]:
    records: dict[str, ColumnManifestRecord] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = ColumnManifestRecord.from_dict(json.loads(line))
            if record.record_id in records:
                raise ValueError(
                    f"Duplicate manifest record_id at line {line_number}: "
                    f"{record.record_id}"
                )
            records[record.record_id] = record
    if not records:
        raise ValueError("Manifest contains no column records")
    return records


def _load_ontology_iris(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    concepts = value.get("concepts") if isinstance(value, dict) else None
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError("Ontology JSON contains no concepts")
    return set(concepts)


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as source:
        value = pickle.load(source)  # noqa: S301 - trusted native DART artifact.
    results = value.get("results") if isinstance(value, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("DART pickle contains no results")
    if any(not isinstance(result, dict) for result in results):
        raise ValueError("DART results must be JSON-like objects")
    return results
