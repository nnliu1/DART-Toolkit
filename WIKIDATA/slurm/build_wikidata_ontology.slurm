#!/bin/bash
#SBATCH --job-name=wikidata_ontology
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:0

set -euo pipefail

REPO=/home/iai/dg3485/CTA/DART-Toolkit
PYTHON=/home/iai/dg3485/miniconda3/envs/cta1_env/bin/python

source /home/iai/dg3485/miniconda3/etc/profile.d/conda.sh
conda activate cta1_env

export WS=$(ws_find myspace)
TRAIN_DATA="$WS/cta_retrieval_dataset_v9_1_full"
WD_DB="$WS/wikidata_db"
ONTOLOGY="$TRAIN_DATA/wikidata_type_ontology.jsonl"

cd "$REPO"

echo "Starting Wikidata ontology artifact build..."
date
echo "Python:        $PYTHON"
echo "Repository:    $REPO"
echo "Training data: $TRAIN_DATA"
echo "Wikidata DB:   $WD_DB"
echo "Ontology:      $ONTOLOGY"

test -d "$TRAIN_DATA"
test -d "$WD_DB/entities.db"
test -d "$WD_DB/entity_labels.db"
test -d "$WD_DB/classes.db"

"$PYTHON" -m WIKIDATA.ontology_parser.build_wikidata_artifact --help

"$PYTHON" -m WIKIDATA.ontology_parser.build_wikidata_artifact \
    --train-path "$TRAIN_DATA" \
    --wd-db "$WD_DB" \
    --output "$ONTOLOGY" \
    --max-examples 10 \
    --max-ancestor-depth 32

"$PYTHON" -m WIKIDATA.ontology_parser.validate_wikidata_artifact \
    --train-path "$TRAIN_DATA" \
    --ontology "$ONTOLOGY" \
    --max-examples 10

echo "Wikidata ontology artifact build and validation completed."
ls -lh \
    "$ONTOLOGY" \
    "$TRAIN_DATA/wikidata_type_ontology.meta.json" \
    "$TRAIN_DATA/wikidata_type_ontology.missing.jsonl"
date
