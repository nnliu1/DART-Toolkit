#!/bin/bash
#SBATCH --job-name=wd_entites
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export PATH=$JAVA_HOME/bin:$PATH
export PYSPARK_SUBMIT_ARGS="--driver-memory 32g pyspark-shell"
export WD_DIR=$WS/wikidata/20250320
export WP_DIR=$WS/wikipedia/20250320
export DBP_DIR=$WS/dbpedia  

export SPARK_LOCAL_DIRS=$TMPDIR
# export KGDATA_FORCE_DISABLE_CHECK_SIGNATURE=1

echo "WD_DIR:  $WD_DIR"
echo "WP_DIR:  $WP_DIR"
echo "TMPDIR:  $TMPDIR"

# echo "Building wikipedia..."
# echo "[step3a] Building html_articles ..."
# $PYTHON -m kgdata.wikipedia.datasets -d html_articles
 
# # Step 3b: extract tables from HTML articles
# echo "[step3b] Building html_tables ..."
# $PYTHON -m kgdata.wikipedia.datasets -d html_tables
 
# Step 3c: filter to relational tables only
echo "[step3c] Building relational_tables ..."
$PYTHON -m kgdata.wikipedia.datasets -d relational_tables
 
# Step 3d: link table cells to Wikidata entities via hyperlinks
echo "[step3d] Building linked_relational_tables ..."
$PYTHON -m kgdata.wikipedia.datasets -d linked_relational_tables
echo "Done: class"