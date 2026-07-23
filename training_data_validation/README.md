# DART CTA training-data validation

This folder validates the full Wikidata CTA retrieval dataset without Torch,
Transformers, a GPU, or network access. Source data is opened read-only.

## Run the full validation

From `C:\Users\dg3485\DART-Toolkit`:

```powershell
python .\training_data_validation\validate_and_build_smoke.py `
  --training-dir .\data\training_data `
  --model-zip .\dart\model.zip `
  --encoder-dir .\WIKIDATA\dart_encoder `
  --output-dir .\training_data_validation `
  --smoke-size 256 `
  --seed 42
```

## Outputs

- `reports/dataset_validation.json`: complete machine-readable result.
- `reports/compatibility_report.md`: concise interpretation and blockers.
- `reports/type_frequency.csv`: primary Wikidata type distribution.
- `smoke/train_smoke.jsonl`: deterministic type-diverse sample.
- `smoke/ontology_training_smoke.jsonl`: generated only to test the current
  loader contract. It is not a formal ontology for training or evaluation.
- `smoke/model_smoke_test.ps1`: checks model dependencies when available.

## Tests

```powershell
cd .\training_data_validation
python -m unittest discover -s tests -v
```

DBpedia is treated as an unseen ontology for later evaluation and is not used
in Wikidata training-quality pass/fail decisions. The current DART loader accepts
one JSONL file; this validator reports that operational limitation separately.
