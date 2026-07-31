# numDART CTA retrieval

This package wraps the unmodified DART CTA retriever. Local preparation
creates DART-compatible column and ontology files. GPU retrieval runs only
through the SLURM script, and export converts DART's pickle into stable JSONL.

## Prepare inputs

Install the CPU-side dependency and prepare the ignored run directory:

```bash
python -m pip install -r numDART/requirements.txt
python -m numDART.retrieval.cli prepare \
  --config numDART/config/cta_retrieval.example.json
python -m numDART.retrieval.cli validate \
  --config numDART/config/cta_retrieval.example.json \
  --stage prepared
```

The example reads the real DBpedia table archive and ontology under
`numDART/data/`. Relative paths are resolved from the configuration file.
Before the HPC run, update `dart_repo` and `model_path` or set the equivalent
submission variables below. `model_path` must point to an unpacked Hugging
Face checkpoint directory, not `model.zip`.

## Submit on HPC

The tracked SLURM file deliberately omits cluster-specific account, partition,
module, and virtual-environment setup. Supply those according to the target
cluster, then submit:

```bash
export NUMDART_REPO=/hpc/project/DART-Toolkit
export DART_REPO=/hpc/project/DART
export DART_MODEL_PATH=/hpc/project/checkpoints/dart
export RUN_DIR=/hpc/project/DART-Toolkit/numDART/outputs/cta-dbpedia-baseline
export CONFIG_PATH=/hpc/project/DART-Toolkit/numDART/config/cta_retrieval.example.json
export PYTHON_BIN=/hpc/project/envs/dart/bin/python

sbatch numDART/slurm/run_cta_retrieval.slurm
```

Optional variables are `TOP_K`, `MAX_CELLS`, `MAX_PARENTS`, and `BATCH_SIZE`.
Their defaults are 20, 10, 3, and 128. Keep them consistent with the JSON
configuration so complete-output validation uses the same Top-K.

The SLURM job calls only DART's native `src/cta/run_retrieve.py`; it does not
call DART's LLM reranking pipeline.

## Artifacts

Each ignored `numDART/outputs/<run_id>/` directory contains:

- `columns/*.json`: native DART column inputs;
- `manifest.jsonl`: original table and column identities;
- `ontology.json`: runtime ontology classes;
- `preparation_summary.json`: preparation counts;
- `retrieval.pkl`: native DART output created on HPC;
- `candidates.jsonl`: normalized CTA candidates;
- `export_summary.json` and `validation_summary.json`: integrity checks.

Each candidate line follows the shared contract:

```json
{"table_id":"Test/table","task":"cta","source_column":4,"candidate_iri":"http://dbpedia.org/ontology/Quantity","rank":1,"retrieval_score":0.81,"target_column":null,"reranker_score":null,"metadata":{"adapter":"dart","candidate_label":"Quantity","dart_record_id":"...","run_id":"cta-dbpedia-baseline"}}
```

If GPU retrieval succeeds but export fails, preserve `retrieval.pkl`, correct
the configuration or manifest issue, and rerun only:

```bash
python -m numDART.retrieval.cli export --config CONFIG.json
python -m numDART.retrieval.cli validate --config CONFIG.json
```

Pickle is executable serialization. Export only a `retrieval.pkl` produced by
the trusted DART job; never load a pickle obtained from an untrusted source.
