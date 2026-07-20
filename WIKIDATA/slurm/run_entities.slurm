#!/bin/bash
#SBATCH --job-name=wd_entites
#SBATCH --cpus-per-task=4
#SBATCH --mem=124G
#SBATCH --time=24:00:00
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

echo "Building Wikidata entities..."

# kgdata wikidata entities -d $WS/wikidata -o $WS/wikidata_db -c
# $PYTHON -m kgdata.wikidata entities \
#     -o $WS/wikidata_db \
#     -c
# echo "Done: entities"
# $PYTHON -m kgdata.wikidata entity_labels \
#     -o $WS/wikidata_db \
#     -c
# echo "Done: entities label"
# $PYTHON -m kgdata.wikidata entity_redirections \
#     -o $WS/wikidata_db \
#     -c

# echo "Done: entities redirections"
$PYTHON -m kgdata.wikidata classes \
    -o $WS/wikidata_db \
    -c


echo "Done: class"