# DART v1 training-data compatibility report

## Result

- Wikidata training-data quality: **PASS**
- Records checked: **173,976**
- Valid records: **173,976**
- Unique primary Wikidata types: **7,704**
- Unique-type metadata reconciliation: **explained_by_label_not_found**

## Interpretation

This report evaluates only the Wikidata retrieval training data. DBpedia is an unseen ontology for later evaluation and is deliberately excluded from all training-quality pass/fail criteria.

Metadata reports 7724 types, while emitted records contain 7704 primary types. The difference of 20 equals `label_not_found=20`, so it is classified as explained by label-resolution filtering rather than shard loss.

## Quality failures

- None.

## Operational note

The current `CTADataset` opens one `train_path`, while the full dataset contains 4 shards. This is a loader/launch issue, not a defect in the Wikidata records.

## Smoke artifacts

- `smoke/train_smoke.jsonl`: deterministic, type-diverse training sample.
- `smoke/ontology_training_smoke.jsonl`: label-derived ontology for loader contract testing only; not a formal training ontology.
- `smoke/model_smoke_test.ps1`: dependency-gated command for a future model-level smoke test.
