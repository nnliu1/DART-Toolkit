"""Pure hard-negative selection helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Set

from .schema_v2 import normalize_ignored_qids, normalize_positive_qids


def build_exclusion_set(record: Dict) -> Set[str]:
    return set(normalize_positive_qids(record)) | set(normalize_ignored_qids(record))


def effective_retrieval_k(top_k: int, max_rank: int, ontology_size: int) -> int:
    if top_k <= 0 or max_rank <= 0 or ontology_size <= 0:
        raise ValueError("top_k, max_rank, and ontology_size must be positive")
    return min(top_k, max_rank, ontology_size)


def sample_key(record: Dict) -> str:
    return f'{record.get("table_id", "")}___{record.get("col_index", -1)}'


def iter_mining_record_batches(
    records: Iterable[Dict],
    ontology_qids: Set[str],
    batch_size: int,
) -> Iterator[List[Dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    seen_keys = set()
    batch: List[Dict] = []
    for record in records:
        key = sample_key(record)
        if key in seen_keys:
            raise ValueError(f"Duplicate sample key across training shards: {key}")
        seen_keys.add(key)
        if not any(qid in ontology_qids for qid in normalize_positive_qids(record)):
            continue
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def select_hard_negatives(
    *,
    ranked_qids: Sequence[str],
    ranked_scores: Sequence[float],
    record: Dict,
    ontology_qids: Set[str],
    n_hard_negs: int,
    min_rank: int,
    max_rank: int,
    min_score: float,
) -> List[str]:
    excluded = build_exclusion_set(record)
    selected: List[str] = []
    seen = set(excluded)
    for rank, (qid, score) in enumerate(
        zip(ranked_qids[:max_rank], ranked_scores[:max_rank]), start=1
    ):
        if rank < min_rank or score < min_score or qid not in ontology_qids or qid in seen:
            continue
        selected.append(qid)
        seen.add(qid)
        if len(selected) == n_hard_negs:
            return selected
    for qid in record.get("hard_negative_type_qids", []):
        if qid in ontology_qids and qid not in seen:
            selected.append(qid)
            seen.add(qid)
            if len(selected) == n_hard_negs:
                break
    return selected
