# Page-Grouped Training and Validation Split Design

## Objective

Create a deterministic 90/10 training and validation split from the 173,976
Wikidata CTA retrieval records without leaking columns or tables from the same
Wikipedia page across the two partitions.

## Inputs

The source is the unchanged directory:

```text
C:\Users\dg3485\DART-Toolkit\data\training_data
```

Only files matching `train_shard_*.jsonl` are input records. Ontology,
metadata, checksum, and missing-report files in the same directory are not
treated as training records.

## Grouping and Assignment

The split unit is a Wikipedia page. For a record such as:

```text
https://en.wikipedia.org/wiki/Example?table_no=4
```

the group key is:

```text
https://en.wikipedia.org/wiki/Example
```

All records with the same page key are assigned together. Empty or malformed
`table_id` values use the complete `table_id` string as their group key and are
reported in metadata.

Assignment uses SHA-256 over the UTF-8 string `42:<page-key>`. The first eight
digest bytes are interpreted as an unsigned integer. A group enters validation
when that value modulo 10 is zero; otherwise it enters training. This gives an
approximately 90/10 split that is stable across Python versions, machines, and
input shard order.

The split does not force every validation type to appear in training. The
metadata explicitly reports validation positive types that are seen and unseen
in training. This allows checkpoint selection to be analysed separately for
ordinary page generalization and harder type generalization.

## Outputs

The source shards remain untouched. Outputs are written beneath:

```text
data/training_data/splits/page_90_10_seed42/
```

with this layout:

```text
train/
  train_shard_0000.jsonl
  train_shard_0001.jsonl
  ...
val/
  train_shard_0000.jsonl
split_metadata.json
```

Both train and validation use `train_shard_*.jsonl` filenames so the existing
directory resolver can read either directory. Output shards contain at most
50,000 records and preserve the source-record order within each partition.

Files are built in a temporary sibling directory and moved into the final
location only after all validations pass. An existing final output directory
is not overwritten unless the caller explicitly supplies `--overwrite`.

## Validation

The splitter must verify:

- every non-empty source line is emitted exactly once;
- train and validation record totals sum to 173,976 for the current dataset;
- train and validation page sets are disjoint;
- train and validation `table_id` sets are disjoint;
- every output line is a JSON object;
- positive QID sets and all other record fields are unchanged;
- neither partition is empty;
- rerunning with the same inputs, ratio, and seed produces the same page
  assignment.

`split_metadata.json` records:

- source paths and SHA-256 configuration;
- source, train, and validation record counts;
- train and validation page and table counts;
- primary and all-positive type counts per partition;
- validation positive types seen and unseen in training;
- empty/malformed `table_id` count;
- output shard counts;
- ratio, seed, and shard-size configuration.

## Command

The local command will be:

```text
python data/split_training_data.py \
  --input-dir data/training_data \
  --output-dir data/training_data/splits/page_90_10_seed42 \
  --val-ratio 0.1 \
  --seed 42
```

Training then uses:

```text
--train_path data/training_data/splits/page_90_10_seed42/train
--val_path data/training_data/splits/page_90_10_seed42/val
```

## Implementation Quality

The implementation remains a single independently runnable standard-library
script at `data/split_training_data.py`. Its data-processing operations are
small pure functions that can be imported by unit tests without executing the
CLI.

The code uses:

- Google-style module, class, and function docstrings;
- Python type annotations compatible with Python 3.8;
- descriptive names and short single-purpose functions;
- `argparse` and an explicit `main(argv=None) -> int`;
- an `if __name__ == "__main__"` standalone entry point;
- no third-party runtime dependencies;
- no unrelated abstraction or framework code.
