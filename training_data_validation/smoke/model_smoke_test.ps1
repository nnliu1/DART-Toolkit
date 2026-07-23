$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Install torch and transformers before running the model smoke test." }
Write-Host "Dependencies are available. Use train.py with smoke/train_smoke.jsonl and smoke/ontology_training_smoke.jsonl for a one-step contract smoke run."
