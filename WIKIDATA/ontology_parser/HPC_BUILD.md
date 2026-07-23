# Build the Wikidata Ontology Artifact on HAICORE

The build is fully offline. It reads the CTA training shards and the matching
Wikidata `kgdata` RocksDB databases from the HPC workspace. It does not query
Wikidata SPARQL or an external API.

## Expected HPC inputs

```text
$WS/cta_retrieval_dataset_v9_1_full/train_shard_*.jsonl
$WS/wikidata_db/entities.db
$WS/wikidata_db/entity_labels.db
$WS/wikidata_db/classes.db
```

The job writes:

```text
$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.jsonl
$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.meta.json
$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.missing.jsonl
```

## 1. Transfer the files

If the HPC checkout tracks the same Git repository, push or otherwise transfer
the commits and update the checkout. Only these runtime files are required:

```text
WIKIDATA/ontology_parser/wikidata_artifact.py
WIKIDATA/ontology_parser/build_wikidata_artifact.py
WIKIDATA/ontology_parser/validate_wikidata_artifact.py
WIKIDATA/slurm/build_wikidata_ontology.slurm
```

When copying from another checkout already available on HPC:

```bash
REPO=/home/iai/dg3485/CTA/DART-Toolkit

cp /path/to/source/WIKIDATA/ontology_parser/wikidata_artifact.py \
  "$REPO/WIKIDATA/ontology_parser/"
cp /path/to/source/WIKIDATA/ontology_parser/build_wikidata_artifact.py \
  "$REPO/WIKIDATA/ontology_parser/"
cp /path/to/source/WIKIDATA/ontology_parser/validate_wikidata_artifact.py \
  "$REPO/WIKIDATA/ontology_parser/"
cp /path/to/source/WIKIDATA/slurm/build_wikidata_ontology.slurm \
  "$REPO/WIKIDATA/slurm/"
```

Do not copy or move the RocksDB directories. The job reads them directly from
the workspace with `read_only=True`.

## 2. Run the preflight

```bash
cd /home/iai/dg3485/CTA/DART-Toolkit

source /home/iai/dg3485/miniconda3/etc/profile.d/conda.sh
conda activate cta1_env

python -m WIKIDATA.ontology_parser.build_wikidata_artifact --help
python -m WIKIDATA.ontology_parser.validate_wikidata_artifact --help

export WS=$(ws_find myspace)
test -d "$WS/cta_retrieval_dataset_v9_1_full"
test -d "$WS/wikidata_db/entities.db"
test -d "$WS/wikidata_db/entity_labels.db"
test -d "$WS/wikidata_db/classes.db"
```

All commands must exit without an error.

## 3. Submit the job

The Slurm output directory must exist before submission:

```bash
cd /home/iai/dg3485/CTA/DART-Toolkit
mkdir -p log
sbatch WIKIDATA/slurm/build_wikidata_ontology.slurm
```

The job is CPU-only and requests four CPUs, 64 GB RAM, and four hours.

## 4. Monitor the job

Replace `<JOB_ID>` with the ID printed by `sbatch`:

```bash
squeue -j <JOB_ID>
tail -f "log/wikidata_ontology_<JOB_ID>.out"
```

The successful log ends with:

```text
Wikidata ontology artifact build and validation completed.
```

## 5. Inspect the artifacts

```bash
export WS=$(ws_find myspace)
OUT="$WS/cta_retrieval_dataset_v9_1_full"

ls -lh \
  "$OUT/wikidata_type_ontology.jsonl" \
  "$OUT/wikidata_type_ontology.meta.json" \
  "$OUT/wikidata_type_ontology.missing.jsonl"

head -n 1 "$OUT/wikidata_type_ontology.jsonl" | python -m json.tool
cat "$OUT/wikidata_type_ontology.meta.json"
head -n 10 "$OUT/wikidata_type_ontology.missing.jsonl"
```

The metadata should report 9,745 candidate records for the current full
training dataset. A non-empty missing report is not automatically a failure:
it records types for which optional descriptions, aliases, or class metadata
were unavailable in the snapshot.

## 6. Rerun validation without rebuilding

```bash
cd /home/iai/dg3485/CTA/DART-Toolkit
source /home/iai/dg3485/miniconda3/etc/profile.d/conda.sh
conda activate cta1_env
export WS=$(ws_find myspace)

python -m WIKIDATA.ontology_parser.validate_wikidata_artifact \
  --train-path "$WS/cta_retrieval_dataset_v9_1_full" \
  --ontology \
    "$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.jsonl" \
  --max-examples 10
```

The validator exits nonzero if any training candidate is missing, QIDs are
duplicated or unsorted, a required field has the wrong type, a parent entry is
malformed, or an examples list exceeds the configured limit.
