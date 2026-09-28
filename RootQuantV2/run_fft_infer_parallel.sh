#!/usr/bin/env bash
# run_fft_infer_parallel.sh — evaluate ONE checkpoint on the MIXED test set by
# data-sharding it across several GPUs (infer_dual.py has no DDP). Splits the
# mixed CSV into one contiguous shard per GPU, runs one infer_dual per GPU,
# concatenates the per-image CSVs into preds/<NAME>.csv, then re-aggregates every
# model in preds/ (the same directory run_infer_all.sh writes to).
#
# Usage (NAME = a directory under RootQuantV2/runs/checkpoints/ holding best.pt):
#   NAME=rootquant-v2-soybean-to-maize-fft GPUS=0,1,2,3 BATCH=32 \
#     bash run_fft_infer_parallel.sh
#
# Needs the per-species split layout (see "Data layout" in the README).
set -uo pipefail

# Paths derived from this script's location, so the repo can live anywhere.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARENT_DIR="${PARENT_DIR:-$(dirname "${PROJECT_DIR}")}"   # `python -m RootQuantV2.*` runs from here
cd "$PARENT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

NAME="${NAME:-rootquant-v2-weights}"
GPUS="${GPUS:-0}"                   # comma-separated; one shard per GPU
BATCH="${BATCH:-32}"                # per-shard batch size; lower it if a shard runs out of memory
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT="${CKPT:-${PROJECT_DIR}/runs/checkpoints/${NAME}/best.pt}"
# Dataset locations. See "Data layout" in the README; every one is overridable.
DATA_ROOT="${ROOTQUANT_DATA_DIR:?set ROOTQUANT_DATA_DIR to the directory holding train/val/test_data.csv and images/}"
MIXED_CSV="${MIXED_CSV:-${DATA_ROOT}/test_data.csv}"
DATA_PATH="${DATA_PATH:-${ROOTQUANT_IMAGES:-${DATA_ROOT}/images}}"
SOY_CSV="${SOY_CSV:-${ROOTQUANT_SOY_DIR:-${DATA_ROOT}/soy}/test_data.csv}"
MAIZE_CSV="${MAIZE_CSV:-${ROOTQUANT_MAIZE_DIR:-${DATA_ROOT}/maize}/test_data.csv}"

OUT_DIR="${PROJECT_DIR}/runs/inference_eval"
PRED_DIR="${OUT_DIR}/preds"
SHARD_DIR="${OUT_DIR}/_shards_${NAME}"  # temp; NOT globbed by aggregate (it globs preds/*.csv)
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$PRED_DIR" "$SHARD_DIR" "$LOG_DIR"
rm -f "${SHARD_DIR}"/*.csv
STAMP="$(date +%Y%m%d_%H%M%S)"
IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
NSHARDS=${#GPU_LIST[@]}

[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }

echo "[shard-infer] sharding ${MIXED_CSV} into ${NSHARDS} parts -> ${SHARD_DIR}"
python - "$MIXED_CSV" "$SHARD_DIR" "$NSHARDS" <<'PY'
import sys, math, pandas as pd
csv, outdir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(csv)
size = math.ceil(len(df) / n)
total = 0
for i in range(n):
    part = df.iloc[i*size:(i+1)*size]
    part.to_csv(f"{outdir}/shard{i}.csv", index=False)
    total += len(part)
    print(f"  shard{i}: {len(part)} rows")
assert total == len(df), (total, len(df))
print(f"  total {total} rows == {len(df)} (ok)")
PY

echo "[shard-infer] launching ${NSHARDS} infer_dual jobs @ batch ${BATCH} (one per GPU)"
pids=()
for i in $(seq 0 $((NSHARDS-1))); do
  log="${LOG_DIR}/infer_${NAME}_shard${i}_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" python -m RootQuantV2.infer_dual \
    --checkpoint "$CKPT" --model_name "${NAME}" \
    --mixed_csv "${SHARD_DIR}/shard${i}.csv" --data_path "$DATA_PATH" \
    --soy_csv "$SOY_CSV" --maize_csv "$MAIZE_CSV" \
    --output_csv "${SHARD_DIR}/out_shard${i}.csv" \
    --batch_size "$BATCH" --num_workers "$NUM_WORKERS" > "$log" 2>&1 &
  pids+=($!)
  echo "  GPU ${GPU_LIST[$i]} -> pid $!  log=$log"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[shard-infer] all shard jobs finished (rc=${rc})"
if [[ "$rc" -ne 0 ]]; then
  echo "[shard-infer] FATAL: a shard job failed — inspect logs/infer_${NAME}_shard*_${STAMP}.log"
  grep -iE "error|oom|out of memory|traceback" "${LOG_DIR}"/infer_"${NAME}"_shard*_"${STAMP}".log | tail -20
  exit 1
fi

echo "[shard-infer] concatenating shard outputs -> ${PRED_DIR}/${NAME}.csv"
python - "$SHARD_DIR" "$NSHARDS" "${PRED_DIR}/${NAME}.csv" "$MIXED_CSV" <<'PY'
import sys, pandas as pd
shard_dir, n, out, mixed = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
parts = [pd.read_csv(f"{shard_dir}/out_shard{i}.csv") for i in range(n)]
df = pd.concat(parts, ignore_index=True)
exp = len(pd.read_csv(mixed))
assert len(df) == exp, f"row mismatch: concat={len(df)} expected={exp}"
df.to_csv(out, index=False)
print(f"  wrote {out}: {len(df)} rows (== {exp})")
PY

echo "[shard-infer] re-aggregating all models in ${PRED_DIR}"
python -m RootQuantV2.aggregate_metrics \
  --preds_dir "$PRED_DIR" \
  --out_csv "${OUT_DIR}/metrics_summary.csv" 2>&1 | tee "${LOG_DIR}/aggregate_${NAME}_${STAMP}.log"

rm -f "${SHARD_DIR}"/*.csv
rmdir "${SHARD_DIR}" 2>/dev/null || true
echo "[shard-infer] DONE. summary -> ${OUT_DIR}/metrics_summary.csv"
