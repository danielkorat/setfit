#!/usr/bin/env bash
# Probe queue: one process per (model, shots, dataset), so memory is freed between runs. Extra args go to run.sh.
#   MODELS="BAAI/bge-base-en-v1.5 sentence-transformers/all-mpnet-base-v2" SHOTS="8 0" ./queue_probe.sh --device mps
cd "$(dirname "$0")"
MODELS=${MODELS:-"BAAI/bge-base-en-v1.5 sentence-transformers/all-mpnet-base-v2"}
SHOTS=${SHOTS:-"8 0"}
DATASETS=${DATASETS:-"sst2 ag_news enron_spam banking77"}
for shots in $SHOTS; do
  for model in $MODELS; do
    tag=$(basename "$model")
    for ds in $DATASETS; do
      echo "$(date +%H:%M:%S) START $tag shots=$shots $ds"
      ./run.sh --model "$model" --datasets "$ds" --shots "$shots" --seeds 0 --n_test 100 \
        --batch_size 16 --max_steps 600 --latency 50 --out "out_probe_$tag" "$@" >> "out_probe_$tag.log" 2>&1
      status=$?
      echo "$(date +%H:%M:%S) END   $tag shots=$shots $ds exit=$status"
    done
  done
done
echo "$(date +%H:%M:%S) QUEUE_DONE"
