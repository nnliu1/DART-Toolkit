# Recall-Oriented CTA Training Design

## Goal

Improve the Wikidata-derived CTA training pipeline so that the DART retriever maximizes the probability that a semantically acceptable type appears in the candidate set passed to the LLM reranker. Preserve compatibility with existing JSONL datasets, checkpoints, and HPC commands.

## Scope

This change covers four high-impact areas:

1. Entity-level P31 coverage and multi-positive evidence in dataset synthesis.
2. Hierarchy-aware exclusion of false negatives.
3. Multi-positive in-batch contrastive training.
4. Recall@10-oriented validation and checkpoint selection.

It does not add ontology-specific mappings, table-context training, LLM reranker training, model-based data cleaning, or a new HPC execution environment.

## Compatibility

Existing fields remain present and keep their current meanings:

- `positive_type_qid` is the primary direct P31 type.
- `positive_type_label` is its label.
- `hard_negative_type_qids` and `hard_negative_type_labels` remain flat lists.

New records add:

- `positive_type_qids`: all accepted direct P31 positives, ordered by entity coverage and then QID for deterministic ties.
- `positive_type_labels`: labels parallel to `positive_type_qids`.
- `positive_type_support`: mapping from accepted direct-positive QID to entity coverage in `[0, 1]`.
- `ignored_type_qids`: hierarchy-related types that must not be sampled or treated as explicit negatives.
- `type_entity_counts`: mapping from candidate direct P31 QID to the number of unique linked entities supporting it.
- `num_typed_entities`: denominator used for entity coverage.

Readers fall back to `positive_type_qid` when the new fields are absent. Existing CLI flags remain valid. New behavior is enabled by defaults but can be controlled through additional flags.

## Label Inference

Each unique linked entity contributes at most one vote to a direct P31 type. Duplicate entities in a column do not increase support. Statements for one entity are deduplicated before counting.

For candidate type `t`:

```text
direct_coverage(t) = unique typed entities with direct P31 t / unique typed entities with at least one valid P31
```

The primary positive is the candidate with highest entity count, breaking ties deterministically by QID. A direct type enters `positive_type_qids` when its coverage is at least the configured positive coverage threshold. A record is rejected when the primary coverage is below the existing majority threshold.

The primary type therefore remains compatible with the old schema, while the complete direct-positive set provides recall-oriented supervision.

## Hierarchy Semantics

The builder traverses a bounded number of P279 parent hops from every accepted direct positive. These ancestors are added to `ignored_type_qids`, not to equally weighted direct positives. This prevents a reasonable abstraction that may be used by an unseen ontology from becoming a strong negative without forcing the encoder to rank a generic ancestor as highly as the direct type.

All direct positives are also included in the ignored set for negative mining. The primary positive remains excluded as before.

The initial implementation uses bounded parent traversal only. Descendant closure, semantic-equivalence inference, and ontology-specific mappings are outside this phase.

## Safe Negative Mining

Static negative mining keeps the existing sibling/cousin/random strategy but applies a per-example exclusion set containing:

- every direct positive;
- every bounded ancestor of a direct positive;
- the primary positive;
- duplicate candidates.

The miner accepts an exclusion set and never returns an excluded QID. If graph candidates are insufficient, random fallback samples only from the remaining global pool. Negative lists may contain fewer than the requested number when the safe pool is too small; unsafe negatives are never inserted merely to meet a fixed count.

Each column receives its own negative-mining call because its accepted positive and ignored sets may differ, even when the primary positive is shared with another column.

## Dataset and Collation

`TrainSample` carries `positive_qids` in addition to the legacy primary QID. The dataset loader normalizes both old and new records into a non-empty ordered positive list whose first item is the primary positive.

For every item, the dataset emits:

- one query text;
- all accepted positive type texts and QIDs;
- explicit hard-negative texts;
- explicit hard-negative QIDs.

The collator deduplicates positive passages across the batch by QID and returns a boolean query-to-positive mask. Multiple queries may share the same positive type, and one query may have multiple positives.

## Training Objective

The in-batch objective becomes multi-positive InfoNCE. For query `i`, every candidate passage whose QID belongs to that query's accepted direct-positive set is positive. The numerator is the log-sum-exp over all positive logits; the denominator is the log-sum-exp over all candidate logits. Identical or alternate positive QIDs are never treated as negatives.

The explicit hard-negative objective remains query-local. It compares the query against one selected primary-positive embedding and its explicit safe negatives. This phase does not assign graded relevance weights to ancestors.

Old records naturally produce a one-positive mask and retain single-positive behavior, except that duplicate types across queries are no longer false negatives.

## Evaluation

Evaluation supports multiple acceptable direct positives. A query counts as a hit at K when any accepted positive QID is in the first K retrieved ontology terms.

Metrics include Recall@1, Recall@5, Recall@10, and Recall@20. Checkpoint selection uses Recall@10 because the LLM reranker consumes ten candidates by default. Recall@20 remains diagnostic for the earlier retrieval stage.

Validation records without `positive_type_qids` fall back to `positive_type_qid`.

## CLI and Metadata

The builder adds explicit parameters for:

- direct-positive coverage threshold;
- hierarchy ignore depth.

Training adds a checkpoint-selection K parameter defaulting to 10. Metadata records all new synthesis settings and schema version. Existing arguments and old JSONL inputs remain supported.

## Testing

Tests use small in-memory fake entity, label, and class databases. They do not require kgdata, a Wikidata dump, a GPU, or network access.

Required behaviors are:

1. Duplicate entities contribute one vote per direct P31 type.
2. Multiple direct P31 statements from one entity do not distort the entity denominator.
3. Multi-positive selection is deterministic and threshold-based.
4. Bounded ancestors enter the ignored set.
5. Static negative mining never returns excluded QIDs.
6. Old and new JSONL schemas normalize to valid positive lists.
7. Duplicate positive QIDs across queries are positive, not in-batch negatives.
8. Multi-positive loss is finite and favors any accepted positive.
9. Multi-positive Recall@10 succeeds when any accepted positive is retrieved.

## Success Criteria

- Existing training data and CLI invocations remain loadable.
- New builder output contains the backward-compatible primary positive plus multi-positive evidence.
- No accepted direct positive or bounded ancestor is emitted as an explicit hard negative.
- In-batch loss does not penalize equal positive QIDs across samples.
- Validation reports Recall@10 and uses it for best-checkpoint selection.
- All new unit tests pass locally without HPC-only resources.
