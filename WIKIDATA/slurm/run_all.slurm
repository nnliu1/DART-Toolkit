#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:0

set -eo pipefail

PYTHON=/home/iai/dg3485/miniconda3/envs/cta1_env/bin/python
source /home/iai/dg3485/miniconda3/etc/profile.d/conda.sh
conda activate cta1_env

export WS=$(ws_find myspace)

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-17.0.18.0.8-1.el9.x86_64
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PYSPARK_SUBMIT_ARGS="--driver-memory 32g pyspark-shell"

export WD_DIR=$WS/wikidata/20250320
export WP_DIR=$WS/wikipedia/20250320
export DBP_DIR=$WS/dbpedia  
export SPARK_LOCAL_DIRS=$TMPDIR


echo "WD_DIR:  $WD_DIR"
echo "TMPDIR:  $TMPDIR"

echo "Building CTA retrieval dataset..."

# $PYTHON $WS/code/22_kgdata_build_dataset.py \
#     --wp-dir  $WS/wikipedia/20250320 \
#     --wd-dir  $WS/wikidata/20250320 \
#     --wd-db   $WS/wikidata_db \
#     --out-dir $WS/cta_retrieval_dataset_debug \
#     --num-hard-negatives    5 \
#     --max-cell-samples     10 \
#     --min-linked-cells      3 \
#     --majority-threshold  0.7 \
#     --max-tables 500000 \
#     --max-accepted 250000 \
#     --seed 42

$PYTHON "$WS/code/kgdata_build_dataset.py" \
    --wp-dir "$WP_DIR" \
    --wd-dir "$WD_DIR" \
    --wd-db "$WS/wikidata_db" \
    --out-dir "$WS/cta_retrieval_dataset_v9_1_full" \
    --num-hard-negatives 5 \
    --max-cell-samples 10 \
    --min-linked-cells 3 \
    --majority-threshold 0.7 \
    --positive-coverage-threshold 0.5 \
    --hierarchy-ignore-depth 2 \
    --max-examples-per-type 200 \
    --seed 42

echo "Done."

