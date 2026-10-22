#!/usr/bin/env bash
# Reproduce every experiment in the paper.
# Usage: bash scripts/run_all.sh data/reddit
set -euo pipefail
DATA=${1:?"data directory with train/dev/test CSVs required"}

echo ">> Classical ML (title and post)"
for F in title post; do python -m mentalnlp.baselines_ml --data-dir "$DATA" --field "$F"; done

echo ">> Deep learning baselines"
for M in cnn lstm bilstm; do
  for F in title post; do python -m mentalnlp.train --data-dir "$DATA" --model "$M" --field "$F"; done
done

echo ">> Pretrained language models"
for M in bert-base-uncased albert-base-v2 roberta-base; do
  for F in title post; do python -m mentalnlp.train_plm --data-dir "$DATA" --model "$M" --field "$F"; done
done

echo ">> Proposed Transformer: single texts and all late-fusion strategies"
for F in title post; do python -m mentalnlp.train --data-dir "$DATA" --model transformer --field "$F"; done
for FUS in concat avg wavg max min; do
  python -m mentalnlp.train --data-dir "$DATA" --model transformer --field both --fusion "$FUS"
done

echo "Done. Results in outputs/"
