# Page-Grouped Training and Validation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a concise standalone Python 3.8 script that deterministically divides the full Wikidata CTA dataset into page-disjoint 90/10 train and validation partitions.

**Architecture:** `data/split_training_data.py` contains pure helpers for path resolution, page-key extraction, SHA-256 assignment, streaming sharded output, metadata collection, and validation, plus a thin argparse CLI. Unit tests use temporary JSONL fixtures and invoke both the pure API and standalone command. The original source shards remain untouched.

**Tech Stack:** Python 3.8 standard library, `unittest`, JSONL, SHA-256.

## Global Constraints

- Group by Wikipedia page rather than row, column, table, or input shard.
- Use SHA-256 assignment with validation ratio `0.1` and seed `42`.
- Preserve every source record unchanged and exactly once.
- Preserve deterministic source order inside each output partition.
- Write at most 50,000 records per output shard.
- Keep train and validation in separate directories containing only `train_shard_*.jsonl` files.
- Do not overwrite an existing output directory unless `--overwrite` is given.
- Use Google-style docstrings, Python 3.8 type annotations, concise single-purpose functions, and no third-party dependencies.
- Retain a standalone `main(argv=None) -> int` and `if __name__ == "__main__"` entry point.

---

### Task 1: Deterministic Page Assignment

**Files:**
- Create: `data/split_training_data.py`
- Create: `WIKIDATA/tests/test_training_data_split.py`

**Interfaces:**
- `resolve_input_shards(input_dir: Path) -> List[Path]`
- `page_key(table_id: object) -> Tuple[str, bool]`
- `is_validation_group(group_key: str, val_ratio: float, seed: int) -> bool`

- [ ] Write failing tests proving that only `train_shard_*.jsonl` inputs are resolved, table numbers are removed from valid Wikipedia URLs, malformed IDs are reported, and assignment is deterministic across calls and input order.
- [ ] Run `python -m unittest WIKIDATA.tests.test_training_data_split -v` and confirm RED because the script does not exist.
- [ ] Implement the three small helpers with Google-style docstrings, QID-independent logic, `hashlib.sha256`, and validation that `0 < val_ratio < 1`.
- [ ] Rerun the focused tests and confirm GREEN.

### Task 2: Streaming Split and Atomic Output

**Files:**
- Modify: `data/split_training_data.py`
- Modify: `WIKIDATA/tests/test_training_data_split.py`

**Interfaces:**
- `split_dataset(input_dir: Path, output_dir: Path, val_ratio: float = 0.1, seed: int = 42, shard_size: int = 50000, overwrite: bool = False) -> Dict[str, object]`

- [ ] Write failing fixture tests for exact-once output, page/table disjointness, record preservation, maximum shard size, positive-type seen/unseen statistics, and refusal to overwrite.
- [ ] Confirm RED against the missing `split_dataset`.
- [ ] Implement streaming input and partition writers using a temporary sibling directory. Track page sets, table sets, primary/all-positive type sets, record counts, malformed IDs, and output shard counts without loading 173,976 records into memory.
- [ ] Validate totals, non-empty partitions, page intersection, table intersection, and emitted-record counts before atomically renaming the temporary directory.
- [ ] Write `split_metadata.json` with sorted lists only where auditability requires them and counts elsewhere.
- [ ] Rerun focused and full WIKIDATA tests.

### Task 3: Standalone CLI

**Files:**
- Modify: `data/split_training_data.py`
- Modify: `WIKIDATA/tests/test_training_data_split.py`

**Interfaces:**
- `parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace`
- `main(argv: Optional[List[str]] = None) -> int`

- [ ] Write failing tests for defaults, explicit arguments, `--overwrite`, and a subprocess invocation of `python data/split_training_data.py --help`.
- [ ] Implement the CLI with `--input-dir`, `--output-dir`, `--val-ratio`, `--seed`, `--shard-size`, and `--overwrite`.
- [ ] Print the completed metadata as formatted JSON and return zero.
- [ ] Verify `--help`, focused tests, the full WIKIDATA suite, Python compilation, and `git diff --check`.

### Task 4: Generate and Audit the Full Split

**Files:**
- Generate without committing: `data/training_data/splits/page_90_10_seed42/`

- [ ] Run the standalone script on `data/training_data`.
- [ ] Independently rescan source, train, and validation JSONL files to confirm 173,976 total output records, exact page/table disjointness, required JSON fields, and no duplicate source-record fingerprints lost or added.
- [ ] Inspect `split_metadata.json`, file sizes, shard names, ratios, train/validation type coverage, and validation seen/unseen type counts.
- [ ] Run the complete WIKIDATA test suite again and report the exact commands for local and HPC training.
