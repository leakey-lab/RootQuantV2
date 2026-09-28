#!/usr/bin/env bash
# run_infer_all.sh — evaluate every checkpoint under RootQuantV2/runs/checkpoints
# (<name>/best.pt, where tools/download_checkpoints.py puts them) on the MIXED
# test set with infer_dual.py, producing BOTH vanilla and TTA predictions split
# per species (maize / soy), then aggregate metrics to a single CSV.
#
# Needs the per-species split layout (see "Data layout" in the README).
# Checkpoints are spread round-robin over GPUS; each GPU evaluates its share one
# after another.
#
#   GPUS=0,1,2,3 BATCH=32 bash run_infer_all.sh
set -uo pipefail

# Paths derived from this script's location, so the repo can live anywhere.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARENT_DIR="${PARENT_DIR:-$(dirname "${PROJECT_DIR}")}"   # `python -m RootQuantV2.*` runs from here
cd "$PARENT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

GPUS="${GPUS:-0}"                   # comma-separated device ids
BATCH="${BATCH:-32}"                # per-job batch size; lower it if a job runs out of memory
NUM_WORKERS="${NUM_WORKERS:-8}"

# Dataset locations. See "Data layout" in the README; every one is overridable.
DATA_ROOT="${ROOTQUANT_DATA_DIR:?set ROOTQUANT_DATA_DIR to the directory holding train/val/test_data.csv and images/}"
MIXED_CSV="${MIXED_CSV:-${DATA_ROOT}/test_data.csv}"
DATA_PATH="${DATA_PATH:-${ROOTQUANT_IMAGES:-${DATA_ROOT}/images}}"
SOY_CSV="${SOY_CSV:-${ROOTQUANT_SOY_DIR:-${DATA_ROOT}/soy}/test_data.csv}"
MAIZE_CSV="${MAIZE_CSV:-${ROOTQUANT_MAIZE_DIR:-${DATA_ROOT}/maize}/test_data.csv}"

OUT_DIR="${PROJECT_DIR}/runs/inference_eval"
PRED_DIR="${OUT_DIR}/preds"
LOG_DIR="${PROJECT_DIR}/logs"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/runs/checkpoints}"
mkdir -p "$PRED_DIR" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

shopt -s nullglob
CKPTS=( "${CKPT_DIR}"/*/best.pt )
shopt -u nullglob
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[run_infer_all] no checkpoints found under ${CKPT_DIR}/*/best.pt"
  echo "                (download some with: python tools/download_checkpoints.py --all)"
  exit 1
fi
IFS=',' read -r -a GPU_LIST <<<"${GPUS}"

run_one () {
  local ckpt="$1" gpu="$2"
  local name; name="$(basename "$(dirname "$ckpt")")"
  local log="${LOG_DIR}/infer_${name}_${STAMP}.log"
  echo "[launch] $name on GPU $gpu (batch $BATCH) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m RootQuantV2.infer_dual \
    --checkpoint "$ckpt" --model_name "$name" \
    --mixed_csv "$MIXED_CSV" --data_path "$DATA_PATH" \
    --soy_csv "$SOY_CSV" --maize_csv "$MAIZE_CSV" \
    --output_csv "${PRED_DIR}/${name}.csv" \
    --batch_size "$BATCH" --num_workers "$NUM_WORKERS" \
    > "$log" 2>&1 || echo "[fail] $name (see $log)"
}

echo "[run_infer_all] start ${STAMP}: ${#CKPTS[@]} checkpoint(s) on GPU(s) ${GPUS}"

pids=()
for g in "${!GPU_LIST[@]}"; do
  (
    for i in "${!CKPTS[@]}"; do
      (( i % ${#GPU_LIST[@]} == g )) || continue
      run_one "${CKPTS[$i]}" "${GPU_LIST[$g]}"
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "[run_infer_all] all inference jobs done"

python -m RootQuantV2.aggregate_metrics \
  --preds_dir "$PRED_DIR" \
  --out_csv "${OUT_DIR}/metrics_summary.csv" \
  2>&1 | tee "${LOG_DIR}/aggregate_${STAMP}.log"

echo "[run_infer_all] DONE. preds -> ${PRED_DIR}/  summary -> ${OUT_DIR}/metrics_summary.csv"
