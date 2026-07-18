#!/bin/bash
# run_ladder.sh — serial driver for one GPU. usage: run_ladder.sh <gpu> <tag> <job1> <job2> ...
# job format: <model>|<data_file>|<out_file>
set -u
GPU=$1; TAG=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/data/lab/hf_cache
cd /data/lab/CareJudge
mkdir -p outputs outputs/logs
echo "[$TAG] === GPU$GPU sequence start $(date '+%Y-%m-%d %H:%M:%S') ==="
for job in "$@"; do
  IFS='|' read -r model data out <<< "$job"
  logname="outputs/logs/${TAG}_$(basename "$out" .jsonl).log"
  echo "[$TAG] $(date '+%H:%M:%S') START $model | $data -> $out"
  python3 scripts/collect_6call.py --input "$data" --out "$out" --model "$model" \
    --max-new-tokens 80 --dtype bfloat16 --device cuda --log-every 50 > "$logname" 2>&1
  rc=$?
  echo "[$TAG] $(date '+%H:%M:%S') DONE $model | $data (exit $rc, log $logname)"
done
echo "[$TAG] === ALL DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
