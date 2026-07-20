"""
Build a Wikidata-native training dataset for CTA retrieval model fine-tuning.

Pipeline (following Vu et al. ISWC 2025):
  1. Load linked_relational_tables from kgdata (Wikipedia tables with cells
     already resolved to Wikidata QIDs via hyperlinks).
  2. For each table column, infer the column type via majority-vote over the
     P31 (instance of) values of the linked entities in that column.
  3. Apply a block-list filter to discard examples where the column header is
     semantically incompatible with the inferred type.
  4. For each accepted (column, type) pair, build a triplet record:
       anchor   : column header + sampled cell values  (query side)
       positive : Wikidata type QID                    (ontology side)
       negative : mined hard-negative type QIDs        (ontology side)
  5. Serialize the dataset as JSONL shards ready for contrastive training.

Requirements:
  pip install kgdata pyspark orjson tqdm

kgdata prerequisites (run once before this script):
  kgdata wikidata entities      -d $WD_DIR -o $WD_DB -c
  kgdata wikidata entity_labels -d $WD_DIR -o $WD_DB -c
  kgdata wikidata classes       -d $WD_DIR -o $WD_DB -c
  python -m kgdata.wikipedia.datasets --wp-dir $WP_DIR --wd-dir $WD_DIR \
      -d linked_relational_tables

Usage:
  python build_cta_retrieval_dataset.py \
      --wp-dir  /path/to/wikipedia \
      --wd-dir  /path/to/wikidata \
      --wd-db   /path/to/wikidata_db \
      --out-dir /path/to/output \
      --num-hard-negatives 5 \
      --max-cell-samples   10 \
      --min-linked-cells   3 \
      --majority-threshold 0.5 \
      --max-examples-per-type 500 \
      --max-tables 200 \
      --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import orjson
from tqdm import tqdm

# ---------------------------------------------------------------------------
# kgdata imports
# FIX 1: get_class_db does not exist; correct name is get_wdclass_db
# ---------------------------------------------------------------------------
from kgdata.wikidata.db import get_entity_db, get_entity_label_db, get_wdclass_db
from kgdata.wikipedia.models.linked_html_table import LinkedHTMLTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cta_dataset_builder")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnRecord:
    """Intermediate representation of a single table column."""

    table_id: str
    col_index: int
    header: str
    cell_values: List[str]   # surface-form text of linked cells
    entity_qids: List[str]   # Wikidata QIDs parallel to cell_values


@dataclass
class TrainingExample:
    """One (anchor, positive, hard_negatives) triplet for contrastive training."""

    anchor_header: str
    anchor_cells: List[str]
    positive_type_qid: str
    positive_type_label: str
    hard_negative_type_qids: List[str]
    hard_negative_type_labels: List[str]
    table_id: str
    col_index: int
    majority_ratio: float


# ---------------------------------------------------------------------------
# Block-list filter
# Format: header_regex -> set of incompatible type QIDs
# ---------------------------------------------------------------------------

BLOCK_LIST_PATTERNS: List[Tuple[re.Pattern, Set[str]]] = [
    (re.compile(r"\b(soccer|football)\s+(team|club|squad)\b", re.I), {"Q6256"}),
    (re.compile(r"\bcountry\b", re.I), {"Q5"}),
    (re.compile(r"\byear\b", re.I), {"Q5", "Q6256"}),
]


def header_type_is_blocked(header: str, type_qid: str) -> bool:
    for pattern, blocked_qids in BLOCK_LIST_PATTERNS:
        if pattern.search(header) and type_qid in blocked_qids:
            return True
    return False


# ---------------------------------------------------------------------------
# Hard-negative mining
#
# FIX 2: WDClass has `parents: List[str]` but NO `children` field.
#
# Strategy: build an inverted index (parent -> children) over the observed
# global type pool at startup. Siblings are types in the pool sharing the
# same parent. This avoids any call to a non-existent .children attribute.
# ---------------------------------------------------------------------------


class HardNegativeMiner:
    """
    Mine hard negatives using the P279 hierarchy.

    Because WDClass only exposes `parents` (not children), we construct a
    parent->children inverse index over the global type pool once at init,
    then reuse it for all sibling/cousin lookups.
    """

    def __init__(
        self,
        class_db,
        global_type_pool: List[str],
        num_negatives: int,
        rng: random.Random,
    ) -> None:
        self._class_db = class_db
        self._global_pool = global_type_pool
        self._num_negatives = num_negatives
        self._rng = rng

        # Build parent -> children index from the global type pool
        logger.info(
            "Building parent->children index over %d types …",
            len(global_type_pool),
        )
        self._parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self._type_parents: Dict[str, List[str]] = {}

        for qid in global_type_pool:
            parents = self._fetch_parents(qid)
            self._type_parents[qid] = parents
            for p in parents:
                self._parent_to_children[p].append(qid)

        logger.info(
            "Index built: %d unique parent nodes.", len(self._parent_to_children)
        )

    def _fetch_parents(self, qid: str) -> List[str]:
        """Fetch WDClass.parents which is List[str] of parent QIDs."""
        try:
            cls = self._class_db[qid]
            return list(cls.parents)
        except KeyError:
            return []

    def _siblings(self, qid: str) -> List[str]:
        """Types in the pool that share at least one parent with qid."""
        result: Set[str] = set()
        for parent in self._type_parents.get(qid, []):
            for child in self._parent_to_children.get(parent, []):
                if child != qid:
                    result.add(child)
        return list(result)

    def _cousins(self, qid: str) -> List[str]:
        """Types two P279 hops away from qid, within the pool."""
        result: Set[str] = set()
        for parent in self._type_parents.get(qid, []):
            for grandparent in self._type_parents.get(parent, []):
                for uncle in self._parent_to_children.get(grandparent, []):
                    if uncle != parent:
                        for cousin in self._parent_to_children.get(uncle, []):
                            if cousin != qid:
                                result.add(cousin)
        return list(result)

    def mine(self, positive_qid: str) -> List[str]:
        seen: Set[str] = {positive_qid}
        candidates: List[str] = []

        for s in self._siblings(positive_qid):
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        if len(candidates) < self._num_negatives:
            for c in self._cousins(positive_qid):
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)

        if len(candidates) >= self._num_negatives:
            return self._rng.sample(candidates, self._num_negatives)

        # Fallback: random from global pool
        remaining = self._num_negatives - len(candidates)
        fallback_pool = [q for q in self._global_pool if q not in seen]
        fallback = self._rng.sample(
            fallback_pool, min(remaining, len(fallback_pool))
        )
        return candidates + fallback


# ---------------------------------------------------------------------------
# Column type inference via P31 majority vote
# ---------------------------------------------------------------------------


def infer_column_type(
    entity_qids: List[str],
    entity_db,
    majority_threshold: float,
) -> Optional[Tuple[str, float]]:
    """
    Infer the Wikidata type for a column by majority-voting P31 values.

    FIX 3: WDStatement.value is WDValue; use .as_entity_id_safe() to extract
    the QID string. Returns "" for non-entity values, which we filter out.
    The old code used .as_qid() which does not exist.
    """
    p31_counter: Counter = Counter()

    for qid in entity_qids:
        try:
            entity = entity_db[qid]
            for stmt in entity.props.get("P31", []):
                # as_entity_id_safe() returns "" for non-entity values
                type_qid = stmt.value.as_entity_id_safe()
                if type_qid:
                    p31_counter[type_qid] += 1
        except KeyError:
            continue

    if not p31_counter:
        return None

    total = sum(p31_counter.values())
    top_qid, top_count = p31_counter.most_common(1)[0]
    ratio = top_count / total

    if ratio < majority_threshold:
        return None

    return top_qid, ratio


# ---------------------------------------------------------------------------
# Iterate linked_relational_tables dataset files
#
# FIX 4 (table object access):
#   LinkedHTMLTable fields (confirmed from source):
#     .table  : Table (rsoup compiled type)
#     .links  : Dict[Tuple[int, int], List[WikiLink]]
#
#   Table API (confirmed via dir()):
#     .shape()         -> (n_rows, n_cols)
#     .get_cell(r, c)  -> cell text str
#     .n_rows          -> int
#     .id / .url       -> table identifier
#
#   WikiLink fields:
#     .wikidata_id     : Optional[str]   <- the entity QID
#     .wikipedia_url   : str
#     .start / .end    : int  (char offsets within cell text)
#
#   Header convention:
#     Row 0 is the header row. Data rows are 1..n_rows-1.
#     This matches how to_full_table() skips header rows.
# ---------------------------------------------------------------------------


def _iter_linked_tables(wp_dir: str) -> Iterator[LinkedHTMLTable]:
    """
    Iterate LinkedHTMLTable objects from kgdata output.

    kgdata writes linked_relational_tables as gzipped JSONL files.
    We search the two common output locations.
    """
    search_bases = [
        os.path.join(wp_dir, "linked_relational_tables"),
        os.path.join(wp_dir, "datasets", "linked_relational_tables"),
    ]
    gz_files: List[str] = []
    for base in search_bases:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for fname in files:
                    if fname.endswith(".gz"):
                        gz_files.append(os.path.join(root, fname))

    if not gz_files:
        raise FileNotFoundError(
            f"No linked_relational_tables .gz files found under {wp_dir}.\n"
            "Run first:\n"
            "  python -m kgdata.wikipedia.datasets "
            "--wp-dir $WP_DIR --wd-dir $WD_DIR -d linked_relational_tables"
        )

    logger.info("Found %d linked_relational_tables shard files.", len(gz_files))

    for gz_path in gz_files:
        try:
            with gzip.open(gz_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield LinkedHTMLTable.from_json(line)
                    except Exception as exc:
                        logger.debug("Skipping malformed record in %s: %s", gz_path, exc)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", gz_path, exc)


def iter_column_records(
    wp_dir: str,
    min_linked_cells: int,
    max_tables: Optional[int] = None,
) -> Iterator[ColumnRecord]:
    """
    Extract ColumnRecord objects from LinkedHTMLTable objects.

    For each column:
      - header   : text from row 0 at that column index
      - data rows: rows 1..n_rows-1
      - linked   : (row, col) key exists in .links with non-None wikidata_id
    """
    tables_seen = 0
    for linked_table in _iter_linked_tables(wp_dir):
        if max_tables is not None and tables_seen >= max_tables:
            break
        tables_seen += 1

        table = linked_table.table
        links = linked_table.links

        try:
            n_rows, n_cols = table.shape()
        except Exception:
            continue

        # Need at least one header row + min_linked_cells data rows
        if n_rows < 2:
            continue

        table_id: str = ""
        for attr in ("id", "url"):
            try:
                table_id = str(getattr(table, attr) or "")
                if table_id:
                    break
            except Exception:
                pass

        for col_idx in range(n_cols):
            # Header from row 0
            try:
                header = str(table.get_cell(0, col_idx) or "").strip()
            except Exception:
                continue
            if not header:
                continue

            # Collect linked data rows
            cell_values: List[str] = []
            entity_qids: List[str] = []

            for row_idx in range(1, n_rows):
                wiki_links = links.get((row_idx, col_idx), [])
                # Take first link with a valid wikidata_id
                qid: Optional[str] = None
                for wl in wiki_links:
                    if wl.wikidata_id:
                        qid = wl.wikidata_id
                        break
                if qid is None:
                    continue

                try:
                    cell_text = str(table.get_cell(row_idx, col_idx) or "").strip()
                except Exception:
                    cell_text = ""

                entity_qids.append(qid)
                cell_values.append(cell_text)

            if len(entity_qids) < min_linked_cells:
                continue

            yield ColumnRecord(
                table_id=table_id,
                col_index=col_idx,
                header=header,
                cell_values=cell_values,
                entity_qids=entity_qids,
            )


# ---------------------------------------------------------------------------
# Label resolution
#
# FIX 4: get_entity_label_db returns WDEntityLabel objects.
#   WDEntityLabel.id    : str  (QID)
#   WDEntityLabel.label : str  (English label, plain string, not a dict)
# ---------------------------------------------------------------------------


def _resolve_label(qid: str, label_db) -> Optional[str]:
    """Resolve the English label for a Wikidata QID via WDEntityLabel.label."""
    try:
        entry = label_db[qid]
        return entry.label or None
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------


def _sample_diverse_cells(
    cell_values: List[str],
    max_samples: int,
    rng: random.Random,
) -> List[str]:
    """Deduplicate then sample up to max_samples non-empty cell values."""
    unique = list(dict.fromkeys(v for v in cell_values if v.strip()))
    if len(unique) <= max_samples:
        return unique
    return rng.sample(unique, max_samples)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    wp_dir: str,
    wd_dir: str,
    wd_db: str,
    out_dir: str,
    num_hard_negatives: int,
    max_cell_samples: int,
    min_linked_cells: int,
    majority_threshold: float,
    max_examples_per_type: int,
    seed: int,
    max_tables: Optional[int] = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Load databases
    logger.info("Loading kgdata databases from %s …", wd_db)
    entity_db = get_entity_db(wd_db)
    label_db = get_entity_label_db(wd_db)
    class_db = get_wdclass_db(wd_db)         # FIX 1
    logger.info("Databases loaded.")

    # ------------------------------------------------------------------
    # Pass 1: collect (column, type) pairs
    # ------------------------------------------------------------------
    logger.info("Pass 1: inferring column types …")
    type_to_examples: Dict[
        str, List[Tuple[ColumnRecord, float, List[str]]]
    ] = defaultdict(list)
    col_count = accepted_count = 0

    for col in tqdm(
        iter_column_records(wp_dir, min_linked_cells, max_tables),
        desc="columns",
        unit="col",
    ):
        col_count += 1
        result = infer_column_type(col.entity_qids, entity_db, majority_threshold)
        if result is None:
            continue
        type_qid, ratio = result
        if header_type_is_blocked(col.header, type_qid):
            logger.debug("Blocked: header=%r type=%s", col.header, type_qid)
            continue
        sampled = _sample_diverse_cells(col.cell_values, max_cell_samples, rng)
        type_to_examples[type_qid].append((col, ratio, sampled))
        accepted_count += 1

    logger.info(
        "Pass 1 done.  columns=%d  accepted=%d  unique_types=%d",
        col_count, accepted_count, len(type_to_examples),
    )

    # Cap per-type examples
    logger.info("Capping at %d examples per type …", max_examples_per_type)
    for qid in list(type_to_examples):
        exs = type_to_examples[qid]
        if len(exs) > max_examples_per_type:
            rng.shuffle(exs)
            type_to_examples[qid] = exs[:max_examples_per_type]

    # Build hard-negative miner
    global_pool = list(type_to_examples.keys())
    logger.info("Global type pool: %d types", len(global_pool))
    miner = HardNegativeMiner(
        class_db=class_db,
        global_type_pool=global_pool,
        num_negatives=num_hard_negatives,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Pass 2: write JSONL shards
    # ------------------------------------------------------------------
    logger.info("Pass 2: writing triplets …")
    stats: Dict[str, int] = {"total": 0, "label_not_found": 0}
    shard_idx = 0
    shard_size = 50_000
    buf: List[dict] = []

    def flush(b: List[dict], idx: int) -> None:
        p = out_path / f"train_shard_{idx:04d}.jsonl"
        with open(p, "wb") as f:
            for rec in b:
                f.write(orjson.dumps(rec) + b"\n")
        logger.info("Shard %04d  %d examples  →  %s", idx, len(b), p)

    for type_qid, examples in tqdm(
        type_to_examples.items(), desc="types", unit="type"
    ):
        pos_label = _resolve_label(type_qid, label_db)  # FIX 4
        if pos_label is None:
            stats["label_not_found"] += 1
            continue

        neg_qids = miner.mine(type_qid)
        neg_labels = [_resolve_label(q, label_db) or q for q in neg_qids]

        for col, ratio, sampled in examples:
            rec = asdict(
                TrainingExample(
                    anchor_header=col.header,
                    anchor_cells=sampled,
                    positive_type_qid=type_qid,
                    positive_type_label=pos_label,
                    hard_negative_type_qids=neg_qids,
                    hard_negative_type_labels=neg_labels,
                    table_id=col.table_id,
                    col_index=col.col_index,
                    majority_ratio=round(ratio, 4),
                )
            )
            buf.append(rec)
            stats["total"] += 1
            if len(buf) >= shard_size:
                flush(buf, shard_idx)
                buf = []
                shard_idx += 1

    if buf:
        flush(buf, shard_idx)

    # Write metadata
    meta = {
        "total_examples": stats["total"],
        "unique_types": len(type_to_examples),
        "label_not_found": stats["label_not_found"],
        "num_shards": shard_idx + 1,
        "config": dict(
            num_hard_negatives=num_hard_negatives,
            max_cell_samples=max_cell_samples,
            min_linked_cells=min_linked_cells,
            majority_threshold=majority_threshold,
            max_examples_per_type=max_examples_per_type,
            max_tables=max_tables,
            seed=seed,
        ),
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Done. %s", meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Wikidata-native CTA retrieval training dataset."
    )
    p.add_argument("--wp-dir", required=True)
    p.add_argument("--wd-dir", required=True)
    p.add_argument("--wd-db", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-hard-negatives", type=int, default=5)
    p.add_argument("--max-cell-samples", type=int, default=10)
    p.add_argument("--min-linked-cells", type=int, default=3)
    p.add_argument("--majority-threshold", type=float, default=0.5)
    p.add_argument("--max-examples-per-type", type=int, default=500)
    p.add_argument(
        "--max-tables",
        type=int,
        default=None,
        help="Stop after reading this many tables (default: no limit). "
             "Use a small value (e.g. 200) for a quick debug run.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        wp_dir=args.wp_dir,
        wd_dir=args.wd_dir,
        wd_db=args.wd_db,
        out_dir=args.out_dir,
        num_hard_negatives=args.num_hard_negatives,
        max_cell_samples=args.max_cell_samples,
        min_linked_cells=args.min_linked_cells,
        majority_threshold=args.majority_threshold,
        max_examples_per_type=args.max_examples_per_type,
        max_tables=args.max_tables,
        seed=args.seed,
    )