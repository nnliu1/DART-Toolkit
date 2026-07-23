# Wikidata Ontology Artifact Design

## Objective

Build the ontology artifact required by DART retrieval training from the same
HPC-hosted Wikidata snapshot and training dataset used to synthesize the CTA
examples. The build must not depend on Wikidata SPARQL or other network
services.

## Inputs

- Training shards:
  `$WS/cta_retrieval_dataset_v9_1_full/train_shard_*.jsonl`
- Wikidata entity database:
  `$WS/wikidata_db/entities.db`
- Wikidata English label database:
  `$WS/wikidata_db/entity_labels.db`
- Wikidata class database:
  `$WS/wikidata_db/classes.db`

Every RocksDB database is opened through its explicit subdirectory with
`read_only=True`. The workspace database root is never opened as a RocksDB
database.

## Candidate Universe

The artifact contains one record for every QID occurring in:

- `positive_type_qids`;
- the legacy fallback field `positive_type_qid`; or
- `hard_negative_type_qids`.

Ancestor QIDs are recorded in the `ancestors` field but do not expand the
candidate universe. This keeps the ontology index aligned with the terms
actually used during training.

## Record Schema

Each JSONL record has this shape:

```json
{
  "qid": "Q100151658",
  "label": "failure mode",
  "description": "specific manner or way by which a failure occurs",
  "aliases": ["failure type", "fault mode"],
  "parents": [
    {"qid": "Q21146257", "label": "type"}
  ],
  "ancestors": ["Q21146257", "Q16889133"],
  "children": [],
  "examples": []
}
```

English label, description, and aliases come from the entity databases.
Missing optional text becomes an empty string or empty list. A missing label
falls back to the QID and is reported.

Direct parents come from the class object's `parents` field. Parent labels are
resolved with the same robust label resolver used for candidate terms.
Ancestors are produced by deterministic breadth-first traversal with cycle
detection and a configurable maximum depth, defaulting to 32.

`children` contains direct children that are also members of the artifact
candidate universe. It is constructed by reversing candidate-to-parent edges;
the builder does not scan all Wikidata classes merely to populate an unused
field.

`examples` contains up to 10 unique, non-empty `anchor_cells` observed in
records where the QID is a positive type. Examples preserve deterministic
first-seen order and are never collected from negative assignments.

## Compatibility

The HPC environment uses `kgdata 5.6.1`. The builder imports
`get_class_db`, with a compatibility fallback to `get_wdclass_db` for older
installations. Label and entity values may be strings, mappings, language
objects, or model objects; extraction normalizes these representations without
requiring the local development environment to install `kgdata`.

The script supports Python 3.8 and uses no network access. Local unit tests use
small in-memory fake databases.

## Outputs

The Slurm job writes:

- `$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.jsonl`
- `$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.meta.json`
- `$WS/cta_retrieval_dataset_v9_1_full/wikidata_type_ontology.missing.jsonl`

The metadata file records the configuration, counts, coverage statistics, and
input paths. The missing report records QIDs with missing entities, labels,
descriptions, aliases, or class metadata. Output is written through temporary
files and atomically replaced only after a successful build.

## Failure Handling

The build fails before writing final artifacts when:

- no training shards are found;
- no candidate QIDs are discovered;
- a required RocksDB directory is missing; or
- the output schema cannot be validated.

Individual missing Wikidata records do not abort the full build. They produce a
valid fallback record and a structured missing-report entry.

## Verification

Unit tests cover candidate extraction, multi-positive example assignment,
metadata normalization, parent-label resolution, cycle-safe ancestor
traversal, artifact-scoped children, deterministic output, and missing-data
reporting.

The Slurm job performs a preflight `--help`, builds the artifact, and runs a
post-build validator that checks JSON parsing, unique QIDs, required keys,
parent structure, and candidate coverage against all training shards.
