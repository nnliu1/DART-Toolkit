# numDART CTA Retrieval Design

## Goal

Run the unmodified DART CTA retriever on HPC and persist deterministic Top-K ontology-class candidates for each real table column. This milestone ends at candidate generation; it does not include CPA, graph reranking, LLM processing, or profiling-enhanced retrieval.

## Design principle

numDART wraps DART rather than copying or modifying it. DART remains the retrieval backend at `src/cta/run_retrieve.py`; numDART owns data conversion, job configuration, output normalization, and validation. This preserves an auditable original-DART baseline for later comparison with numeric-profile-aware variants.

## Data flow

1. Read real tables from the configured ZIP archive without extracting persistent copies.
2. Convert every selected column into one DART-compatible JSON record containing a stable record ID, table ID, column index, header, sampled cells, and optional gold CTA metadata.
3. Convert an OWL or TTL ontology into DART's `{"concepts": ...}` JSON representation. Only ontology classes enter this CTA index.
4. Write a manifest connecting each DART record ID to its original table and column.
5. Submit a SLURM job that invokes the repository's unmodified `src/cta/run_retrieve.py` with the generated column directory, ontology JSON, checkpoint, Top-K, batch size, and output PKL.
6. Convert DART's PKL output into numDART `CandidateRecord` JSONL, recovering column identity from the manifest.
7. Validate candidate count, rank order, finite scores, ontology membership, and coverage.

## Components

### CTA input builder

The builder reads gzipped JSON-lines members inside the table archive. It produces one JSON file per column using the fields expected by DART:

- `table_id`: stable unique record identifier used by DART;
- `pk_col_header_raw` and `pk_col_header_clean`: original and normalized header;
- `cell_samples_raw` and `cell_samples_clean`: deterministic non-null cell samples;
- `gt_uri` and `gt_ontology`: included only when gold annotations are available.

The manifest separately records `record_id`, `source_table_id`, `column_index`, and `column_name`. Separation is necessary because the current DART output preserves `table_id` but not an explicit column index.

Sampling follows source row order, removes null/empty values, converts structured cells to stable JSON text, and takes at most the configured number of cells. No synthetic rows or columns are generated.

### Ontology builder

The builder parses OWL/TTL with RDFLib and exports each named OWL/RDFS class as:

```json
{
  "uri": {
    "label": "...",
    "description": "...",
    "parents": ["..."]
  }
}
```

Labels prefer English `rdfs:label`, then any available label, then the URI local name. Descriptions prefer English `rdfs:comment`. Parents are direct named `rdfs:subClassOf` classes and are represented by labels. Blank nodes and ontology properties are excluded from the CTA candidate index.

### Native DART job wrapper

The SLURM script does not contain model logic. It validates required paths and invokes:

```text
python DART/src/cta/run_retrieve.py
  --model_path ...
  --ontology_path ...
  --data_dir ...
  --output_pkl ...
  --top_k ...
  --max_cells ...
  --max_parents ...
  --batch_size ...
  --device cuda
```

Machine-specific checkpoint, environment, account, partition, and storage paths are supplied through command-line arguments or environment variables rather than hard-coded into Python.

### Candidate exporter

For each DART result and each aligned candidate, the exporter writes one JSONL record with:

- `table_id`: original table identifier;
- `task`: `cta`;
- `source_column`: zero-based original column index;
- `candidate_iri`;
- `rank`;
- `retrieval_score`;
- metadata containing candidate label, DART record ID, query text, score margin, and run configuration identifier.

Gold labels and ranks are stored in a separate ignored diagnostic artifact, not mixed into the candidate contract required by downstream reranking.

## Configuration and artifacts

A single run configuration identifies the table archive, ontology source, DART repository, checkpoint, work directory, output directory, Top-K, sample count, batch size, and random seed. Generated data and tests remain ignored by Git. Production source, configuration examples, and the SLURM template remain tracked.

Each run produces:

- DART column JSON directory;
- DART ontology JSON;
- input manifest JSONL;
- native `retrieval.pkl`;
- normalized `candidates.jsonl`;
- validation summary JSON;
- optional gold-based Recall@K report when annotations are available.

All generated artifacts are written below `numDART/outputs/<run_id>/`.

## Failure handling

Preprocessing fails before submission for duplicate record IDs, empty ontologies, missing table members, or zero generated columns. Export fails for a missing manifest entry, malformed PKL, unequal candidate arrays, non-finite scores, unknown candidate IRIs, duplicate ranks, or fewer than the requested Top-K candidates when the ontology contains at least Top-K classes. Failures are explicit; incomplete outputs are not silently accepted.

## Verification

Local tests use small fixtures and never load the DART checkpoint. They cover deterministic table conversion, ontology conversion, manifest identity, PKL normalization, validation failures, and command construction. A local real-data smoke check reads at least one table from the existing DBpedia or Schema.org archive. The HPC acceptance run processes a small real-table subset before the complete archive.

## Scope boundaries

- CTA only; no CPA candidates.
- Original DART query and passage formatting only.
- No numeric profiling in the baseline query.
- No modification of DART model or checkpoint.
- No graph reranking or LLM calls.
- No synthetic or Wikidata-derived columns in the core evaluation input.

The next experiment will add numeric profiling as an isolated query/passage intervention and compare its candidate Recall@K against this baseline.
