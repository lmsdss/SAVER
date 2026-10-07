#!/usr/bin/env bash
# Fixed-FPS SAVER evaluation; every invocation gets its own result directory.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODEL_PATH="${MODEL_PATH:-lmsdss/SAVER-2B}"
export QWEN_VL_VIDEO_FPS="${FPS:-${QWEN_VL_VIDEO_FPS:-0.1}}"
export LMMS_USE_LOCAL_QWEN_VL_UTILS=1
export PYTHONPATH="$REPO_ROOT/saver:$REPO_ROOT:${PYTHONPATH:-}"
export LMMS_DATA_ROOT="${LMMS_DATA_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/eval/${MODEL_PATH//\//_}/fps-${QWEN_VL_VIDEO_FPS}/${RUN_ID}}"
read -r -a tasks <<< "${EVAL_TASKS:-charades_boxed activitynet_tvg_boxed nextgqa_boxed mvbench_boxed mmvu_val_mc_boxed longvideobench_val_v_boxed video_mmmu_boxed videomme_boxed mvp_mini_boxed}"
mkdir -p "$OUTPUT_PATH"
export SAVER_EVAL_MODEL="$MODEL_PATH" SAVER_EVAL_OUTPUT="$OUTPUT_PATH" SAVER_EVAL_TASKS="${tasks[*]}"
python - <<'PY'
import json, os
from pathlib import Path
config=dict(model=os.environ['SAVER_EVAL_MODEL'], fps=float(os.environ['QWEN_VL_VIDEO_FPS']), tasks=os.environ['SAVER_EVAL_TASKS'].split(), max_frames=256, max_new_tokens=1024, enable_thinking=False, video_total_pixels=128000*32*32, limit=os.environ.get('EVAL_LIMIT'), dry_run=os.environ.get('DRY_RUN')=='1')
(Path(os.environ['SAVER_EVAL_OUTPUT'])/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
PY
extra=()
[[ -z "${EVAL_LIMIT:-}" ]] || extra+=(--limit "$EVAL_LIMIT")
for task in "${tasks[@]}"; do
  export LMMS_CURRENT_TASK="$task"
  export LMMS_VIDEO_FRAME_STATS_TXT="$OUTPUT_PATH/video_frame_stats_${task}.txt"
  CMD=(accelerate launch --num_processes "$NPROC_PER_NODE" --num_machines 1
    -m lmms_eval.__main__ --model qwen3_5
    --model_args "pretrained=${MODEL_PATH},video_min_pixels=16384,video_max_pixels=786432,video_total_pixels=131072000,max_frames=256,image_min_pixels=131072,image_max_pixels=16777216,fps=${QWEN_VL_VIDEO_FPS},enable_thinking=False,adaptive_video_fps=false"
    --gen_kwargs "max_new_tokens=1024"
    --tasks "$task" --batch_size 1 --log_samples --output_path "$OUTPUT_PATH/$task" "${extra[@]}")
  printf '%q ' "${CMD[@]}"; printf '\n'
  [[ "${DRY_RUN:-0}" == 1 ]] || "${CMD[@]}"
done
printf 'Results: %s\n' "$OUTPUT_PATH"
