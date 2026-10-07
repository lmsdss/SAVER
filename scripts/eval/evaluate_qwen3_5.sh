#!/usr/bin/env bash
# Original Qwen3.5 baseline: ordinary prompts and answer parsing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
export EVAL_TASKS="${EVAL_TASKS:-charades activitynet_tvg nextgqa mvbench mmvu_val_mc longvideobench_val_v video_mmmu videomme mvp_mini}"
read -r -a baseline_tasks <<< "$EVAL_TASKS"
for task in "${baseline_tasks[@]}"; do
  if [[ "$task" == *boxed* ]]; then
    echo "[ERROR] This baseline launcher requires non-boxed tasks: $task" >&2
    exit 1
  fi
done
exec bash "$SCRIPT_DIR/evaluate.sh"
