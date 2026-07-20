# Recall-Oriented CTA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wikidata CTA synthesis and DART encoder training multi-positive, hierarchy-safe, backward-compatible, and optimized for candidate Recall@10.

**Architecture:** The builder produces a legacy primary positive plus entity-coverage-based positive evidence and an ignored hierarchy set. The dataset normalizes old and new records, the collator builds a deduplicated positive bank and mask, and the model uses multi-positive InfoNCE while preserving query-local explicit hard negatives.

**Tech Stack:** Python 3.8+, PyTorch, Transformers, kgdata in production, pytest with in-memory fakes locally.

## Global Constraints

- Preserve all existing JSONL fields and old-record loading behavior.
- Preserve existing HPC CLI arguments; only add optional arguments with defaults.
- Do not require a Wikidata dump, network, GPU, or OpenAI API for unit tests.
- Never emit an accepted direct positive or bounded ancestor as an explicit hard negative.
- Select the best checkpoint with Recall@10 by default.

---

### Task 1: Entity-Coverage Labels and Safe Static Negatives

**Files:**
- Modify: `WIKIDATA/build_traning_data/kgdata_build_dataset.py`
- Create: `tests/test_kgdata_build_dataset.py`

**Interfaces:**
- Produces: `ColumnTypeInference(primary_qid, primary_coverage, positive_qids, support, entity_counts, num_typed_entities)`.
- Produces: `collect_ancestors(qids, class_db, max_depth) -> Set[str]`.
- Changes: `HardNegativeMiner.mine(positive_qid, excluded_qids=None) -> List[str]`.

- [ ] **Step 1: Write failing entity-coverage tests**

Use fake statements whose `value.as_entity_id_safe()` returns a QID. Assert duplicate entity QIDs count once, multiple P31 values share the entity denominator, ties are QID-deterministic, and a secondary direct type above `positive_coverage_threshold` is retained.

```python
result = infer_column_types(
    ["Q1", "Q1", "Q2", "Q3"], entity_db,
    majority_threshold=2 / 3,
    positive_coverage_threshold=1 / 3,
)
assert result.primary_qid == "T1"
assert result.num_typed_entities == 3
assert result.entity_counts == {"T1": 3, "T2": 1}
assert result.positive_qids == ["T1", "T2"]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_kgdata_build_dataset.py -v`

Expected: collection/import succeeds but fails because `infer_column_types` and `ColumnTypeInference` do not exist.

- [ ] **Step 3: Implement entity-level inference**

Add a frozen dataclass and replace statement-count inference with unique-entity support:

```python
@dataclass(frozen=True)
class ColumnTypeInference:
    primary_qid: str
    primary_coverage: float
    positive_qids: List[str]
    support: Dict[str, float]
    entity_counts: Dict[str, int]
    num_typed_entities: int
```

Sort candidates with `key=lambda item: (-item[1], item[0])`. Keep `infer_column_type` as a compatibility wrapper returning `(primary_qid, primary_coverage)`.

- [ ] **Step 4: Write failing hierarchy and exclusion tests**

Assert bounded traversal stops at the requested depth and `mine(..., excluded_qids={...})` never returns excluded graph or random candidates, even when fewer than the requested count remain.

- [ ] **Step 5: Verify RED and implement safe mining**

Run the focused tests, then implement:

```python
def mine(self, positive_qid: str, excluded_qids: Optional[Set[str]] = None):
    excluded = set(excluded_qids or ()) | {positive_qid}
    # filter siblings, cousins, and fallback through excluded
```

Add `collect_ancestors` with breadth-first bounded parent traversal.

- [ ] **Step 6: Extend output schema and CLI**

Add fields to `TrainingExample`, add `--positive-coverage-threshold` default `0.5` and `--hierarchy-ignore-depth` default `2`, and write `schema_version: 2` to metadata. Resolve labels for all positive QIDs. Mine negatives per example using direct positives plus collected ancestors as exclusions.

- [ ] **Step 7: Run Task 1 tests**

Run: `python -m pytest tests/test_kgdata_build_dataset.py -v`

Expected: all Task 1 tests pass.

- [ ] **Step 8: Commit**

```bash
git add WIKIDATA/build_traning_data/kgdata_build_dataset.py tests/test_kgdata_build_dataset.py
git commit -m "feat: synthesize recall-oriented CTA labels"
```

### Task 2: Backward-Compatible Multi-Positive Dataset and Collator

**Files:**
- Modify: `WIKIDATA/dart_encoder/data_types.py`
- Modify: `WIKIDATA/dart_encoder/dataset.py`
- Create: `tests/test_cta_dataset.py`

**Interfaces:**
- Changes: `TrainSample` includes `positive_qids: List[str]` while retaining `positive_qid`.
- Produces collator keys: `positive_qids`, `positive_mask`, `primary_pos_indices`, plus existing encodings and `neg_counts`.

- [ ] **Step 1: Write failing schema-normalization tests**

Create temporary ontology/train JSONL files. Assert an old record normalizes to `[positive_type_qid]`; a new record deduplicates positives, keeps the primary first, drops missing ontology types, and removes positives from explicit negatives.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_cta_dataset.py -v`

Expected: failures because items do not expose positive IDs or masks.

- [ ] **Step 3: Implement normalized samples**

Add `positive_qids` with a default factory and normalize with stable order:

```python
raw = [d["positive_type_qid"], *d.get("positive_type_qids", [])]
positive_qids = list(dict.fromkeys(q for q in raw if q in self.ontology))
neg_qids = [q for q in neg_qids if q not in set(positive_qids)]
```

Return `positive_qids`, formatted positive texts, primary positive QID, and negative QIDs from `__getitem__`.

- [ ] **Step 4: Write failing collator-mask tests**

Use two queries sharing `T1`, with one also accepting `T2`. Assert the positive bank is `[T1, T2]` and the mask is `[[True, True], [True, False]]` rather than a diagonal identity matrix.

- [ ] **Step 5: Implement the positive bank and mask**

Deduplicate positive QIDs across the batch, tokenize one passage per bank QID, create a boolean `(batch, bank)` mask, and compute the primary-positive index for each query. Preserve `pos_enc` as the bank encoding consumed by the updated model.

- [ ] **Step 6: Run Task 2 tests and commit**

Run: `python -m pytest tests/test_cta_dataset.py -v`

```bash
git add WIKIDATA/dart_encoder/data_types.py WIKIDATA/dart_encoder/dataset.py tests/test_cta_dataset.py
git commit -m "feat: collate multi-positive CTA samples"
```

### Task 3: Multi-Positive InfoNCE

**Files:**
- Modify: `WIKIDATA/dart_encoder/model.py`
- Modify: `WIKIDATA/dart_encoder/train.py`
- Create: `tests/test_multi_positive_loss.py`

**Interfaces:**
- Produces: `multi_positive_cross_entropy(logits, positive_mask) -> torch.Tensor`.
- Changes: `BiEncoder.forward(..., positive_mask, primary_pos_indices, ...)`.

- [ ] **Step 1: Write failing loss tests**

Test finite loss, rejection of a row with no positive, lower loss when either accepted positive logit rises, and shared QID positives across two queries.

```python
logits = torch.tensor([[4.0, 3.0, -2.0]])
mask = torch.tensor([[True, True, False]])
loss = multi_positive_cross_entropy(logits, mask)
assert torch.isfinite(loss)
```

- [ ] **Step 2: Verify RED and implement the loss**

Run the focused test, then implement:

```python
positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
return (torch.logsumexp(logits, dim=1) -
        torch.logsumexp(positive_logits, dim=1)).mean()
```

Raise `ValueError` when mask shape differs from logits or a row has no positive.

- [ ] **Step 3: Update model and training call**

Use the collator's bank encoding for in-batch logits and select each query's primary embedding by `primary_pos_indices` for explicit hard-negative loss. Pass the new tensors to the model from `train.py`.

- [ ] **Step 4: Run focused and regression tests**

Run: `python -m pytest tests/test_multi_positive_loss.py tests/test_cta_dataset.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add WIKIDATA/dart_encoder/model.py WIKIDATA/dart_encoder/train.py tests/test_multi_positive_loss.py
git commit -m "feat: train CTA encoder with multi-positive InfoNCE"
```

### Task 4: Multi-Positive Recall@10 and Checkpoint Selection

**Files:**
- Modify: `WIKIDATA/dart_encoder/model.py`
- Modify: `WIKIDATA/dart_encoder/train.py`
- Create: `tests/test_retrieval_metrics.py`

**Interfaces:**
- Produces: `recall_hits(ranked_qids, acceptable_qids, k_values) -> Dict[int, bool]`.
- Adds CLI: `--selection_k` default `10`.

- [ ] **Step 1: Write failing metric tests**

Assert a sample with acceptable types `{T2, T3}` is a Recall@10 hit when either appears in the first ten, even when the legacy primary is absent. Assert no hit when acceptable types are outside K.

- [ ] **Step 2: Verify RED and implement metric helper**

Run: `python -m pytest tests/test_retrieval_metrics.py -v`

Then implement set-intersection-based hits and make `evaluate_recall` default to `[1, 5, 10, 20]`.

- [ ] **Step 3: Load multi-positive validation samples**

Normalize validation JSONL with the same primary-first fallback used for training. Evaluate any accepted direct positive as correct.

- [ ] **Step 4: Select checkpoints by configurable K**

Add `--selection_k 10`, validate that it is present in evaluated K values, replace `best_recall20` with `best_selection_recall`, and log the selected metric explicitly.

- [ ] **Step 5: Run the complete local suite**

Run: `python -m pytest tests -v`

Expected: all tests pass without kgdata, network, GPU, or Wikidata files.

- [ ] **Step 6: Static compatibility checks**

Run:

```bash
python -m compileall -q WIKIDATA/dart_encoder WIKIDATA/build_traning_data/kgdata_build_dataset.py
python WIKIDATA/build_traning_data/kgdata_build_dataset.py --help
python -m WIKIDATA.dart_encoder.train --help
```

Expected: compilation and help output succeed in an environment with declared dependencies. If kgdata or Transformers is absent locally, record that dependency limitation and retain passing dependency-free unit tests.

- [ ] **Step 7: Commit**

```bash
git add WIKIDATA/dart_encoder/model.py WIKIDATA/dart_encoder/train.py tests/test_retrieval_metrics.py
git commit -m "feat: select CTA checkpoints by multi-positive Recall@10"
```

### Task 5: Final Verification and HPC Handoff Notes

**Files:**
- Modify: `WIKIDATA/slurm/run_all.slurm`
- Create: `WIKIDATA/README.md`

**Interfaces:**
- Documents old/new schema and exact local/HPC commands.

- [ ] **Step 1: Write a failing CLI consistency test**

Add a small source-level test asserting every option used by `run_all.slurm` exists in the builder parser; this must expose the current unsupported `--max-accepted` argument.

- [ ] **Step 2: Verify RED and fix the Slurm command**

Remove unsupported `--max-accepted`, add explicit values for the new builder flags, and keep paths/configuration otherwise unchanged.

- [ ] **Step 3: Document generation and training commands**

Document schema v2, fallback behavior, new CLI flags, Recall@10 selection, local tests, and the fact that full kgdata/GPU execution must occur on HPC.

- [ ] **Step 4: Final verification**

Run: `python -m pytest tests -v`

Run: `git status --short`

Expected: tests pass; only intended documentation or source changes are present before the final commit.

- [ ] **Step 5: Commit**

```bash
git add WIKIDATA/slurm/run_all.slurm WIKIDATA/README.md tests
git commit -m "docs: add HPC handoff for recall-oriented CTA training"
```
