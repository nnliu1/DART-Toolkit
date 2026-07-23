# DART encoder schema-v2 HPC runbook

The encoder now accepts a JSONL file, a directory containing
`train_shard_*.jsonl`, or a glob. Schema-v2 multi-positive records and legacy
single-positive records are both supported.

## Required inputs

- Training shards, for example:
  `$WS/cta_retrieval_dataset_v9_1_full/train_shard_*.jsonl`
- A Wikidata type ontology JSONL whose `qid` values cover the training
  positives and candidate negatives. Each line follows:
  `{"qid":"Q5","label":"human","description":"","parents":[]}`
- A base encoder or a local Hugging Face checkpoint.

DBpedia is not used as the training ontology. It remains an unseen ontology for
downstream evaluation.

## Train

Run from the repository root so relative package imports resolve:

```bash
python -m WIKIDATA.dart_encoder.train \
  --train_path "$WS/cta_retrieval_dataset_v9_1_full" \
  --ontology_path "$WS/wikidata_type_ontology.jsonl" \
  --output_dir "$WS/dart_schema_v2_model" \
  --model_name intfloat/multilingual-e5-base \
  --epochs 6 \
  --batch_size 32 \
  --min_majority_ratio 0.7 \
  --min_type_count 3 \
  --max_cells 10 \
  --max_length 192 \
  --random_cell_sample \
  --gradient_checkpointing
```

Use a held-out table-level validation JSONL with `--val_path` when available.
Do not randomly split columns from the same table across train and validation.

## Mine model hard negatives

```bash
python -m WIKIDATA.dart_encoder.mine_hard_negative \
  --model_path "$WS/dart_schema_v2_model/best_model" \
  --train_path "$WS/cta_retrieval_dataset_v9_1_full" \
  --ontology_path "$WS/wikidata_type_ontology.jsonl" \
  --output_path "$WS/dart_schema_v2_model/mined_hard_negatives.json" \
  --top_k 50 \
  --max_rank 50 \
  --n_hard_negs 5 \
  --batch_size 256 \
  --max_length 256
```

Mining excludes every `positive_type_qid`, every member of
`positive_type_qids`, and every `ignored_type_qid`. Similarities are computed
one query batch at a time; the full query-by-type matrix is never retained.

## Retrain with mined negatives

Add:

```bash
--hard_neg_path "$WS/dart_schema_v2_model/mined_hard_negatives.json"
```

to the training command.

## Preflight

```bash
python -m py_compile WIKIDATA/dart_encoder/*.py
python -m unittest discover -s WIKIDATA/tests -p 'test_*.py' -v
```

On the HPC environment, all tests including the PyTorch numerical tests must
run without skips before launching the full job.
