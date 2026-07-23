"""Pure-Python label inference and safe-negative helpers for CTA synthesis."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


@dataclass(frozen=True)
class ColumnTypeInference:
    primary_qid: str
    primary_coverage: float
    positive_qids: List[str]
    evidence_qids: List[str]
    support: Dict[str, float]
    entity_counts: Dict[str, int]
    num_typed_entities: int


def _parents(qid: str, class_db) -> List[str]:
    try:
        return list(class_db[qid].parents)
    except KeyError:
        return []


def infer_column_types(
    entity_qids: Sequence[str],
    entity_db,
    majority_threshold: float,
    positive_coverage_threshold: float,
) -> Optional[ColumnTypeInference]:
    """Infer direct P31 positives using unique-entity coverage."""
    counts: Counter = Counter()
    typed_entities = 0

    for entity_qid in dict.fromkeys(entity_qids):
        try:
            entity = entity_db[entity_qid]
        except KeyError:
            continue
        entity_types = {
            stmt.value.as_entity_id_safe()
            for stmt in entity.props.get("P31", [])
        }
        entity_types.discard("")
        if not entity_types:
            continue
        typed_entities += 1
        counts.update(entity_types)

    if not counts or typed_entities == 0:
        return None

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    primary_qid, primary_count = ranked[0]
    primary_coverage = primary_count / typed_entities
    if primary_coverage < majority_threshold:
        return None

    support = {qid: count / typed_entities for qid, count in ranked}
    positive_qids = [
        qid for qid, _ in ranked
        if support[qid] >= positive_coverage_threshold
    ]
    if primary_qid not in positive_qids:
        positive_qids.insert(0, primary_qid)

    return ColumnTypeInference(
        primary_qid=primary_qid,
        primary_coverage=primary_coverage,
        positive_qids=positive_qids,
        evidence_qids=[qid for qid, _ in ranked],
        support={qid: support[qid] for qid in positive_qids},
        entity_counts=dict(ranked),
        num_typed_entities=typed_entities,
    )


def collect_ancestors(
    qids: Iterable[str], class_db, max_depth: int
) -> Set[str]:
    """Return bounded P279 ancestors, excluding the input QIDs."""
    if max_depth <= 0:
        return set()
    roots = set(qids)
    frontier = set(roots)
    ancestors: Set[str] = set()
    for _ in range(max_depth):
        next_frontier: Set[str] = set()
        for qid in frontier:
            next_frontier.update(_parents(qid, class_db))
        next_frontier -= roots
        next_frontier -= ancestors
        if not next_frontier:
            break
        ancestors.update(next_frontier)
        frontier = next_frontier
    return ancestors


class HardNegativeMiner:
    """Mine hierarchy-related negatives while honoring per-query exclusions."""

    def __init__(
        self,
        class_db,
        global_type_pool: List[str],
        num_negatives: int,
        rng: random.Random,
    ) -> None:
        self._class_db = class_db
        self._global_pool = list(global_type_pool)
        self._num_negatives = num_negatives
        self._rng = rng
        self._parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self._type_parents: Dict[str, List[str]] = {}
        nodes = set(global_type_pool)
        for qid in list(nodes):
            nodes.update(_parents(qid, class_db))
        for qid in nodes:
            parents = _parents(qid, class_db)
            self._type_parents[qid] = parents
            if qid in global_type_pool:
                for parent in parents:
                    self._parent_to_children[parent].append(qid)

    def _siblings(self, qid: str) -> Set[str]:
        return {
            child
            for parent in self._type_parents.get(qid, [])
            for child in self._parent_to_children.get(parent, [])
            if child != qid
        }

    def _cousins(self, qid: str) -> Set[str]:
        result: Set[str] = set()
        for parent in self._type_parents.get(qid, []):
            for grandparent in self._type_parents.get(parent, []):
                for uncle in self._parent_to_children.get(grandparent, []):
                    if uncle == parent:
                        continue
                    result.update(self._parent_to_children.get(uncle, []))
        result.discard(qid)
        return result

    def mine(
        self,
        positive_qid: str,
        excluded_qids: Optional[Set[str]] = None,
    ) -> List[str]:
        excluded = set(excluded_qids or ()) | {positive_qid}
        candidates: List[str] = []
        for pool in (
            sorted(self._siblings(positive_qid)),
            sorted(self._cousins(positive_qid)),
        ):
            safe_pool = [
                qid for qid in pool
                if qid not in excluded and qid not in candidates
            ]
            remaining = self._num_negatives - len(candidates)
            if remaining <= 0:
                break
            candidates.extend(
                self._rng.sample(safe_pool, min(remaining, len(safe_pool)))
            )

        if len(candidates) >= self._num_negatives:
            return candidates

        fallback_pool = [
            qid for qid in self._global_pool
            if qid not in excluded and qid not in candidates
        ]
        remaining = self._num_negatives - len(candidates)
        candidates.extend(
            self._rng.sample(fallback_pool, min(remaining, len(fallback_pool)))
        )
        return candidates
