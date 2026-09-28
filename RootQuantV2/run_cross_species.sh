#!/usr/bin/env bash
# =============================================================================
# run_cross_species.sh — soy -> maize cross-species generalization pipeline.
#
# Establishes which fine-tuning strategy yields the most generalizable backbone:
#   Phase 1  ZST  : train the model on SOYBEAN only            (SOY_GPUS)
#   Phase 2  FCFT : warm-start from soy, freeze all but the head, FT on MAIZE
#            FFT  : warm-start from soy, full FT on MAIZE
#                   (FCFT then FFT, sequentially, each on all of FT_GPUS)
#   Phase 3  eval : score every checkpoint (soy->soy in-domain, soy->maize ZST,
#                   FCFT->maize, FFT->maize) and print an R2 summary.
#
# Data (correctness): SOYBEAN training uses the CLEAN soy-only split
# ($ROOTQUANT_SOY_DIR) — NOT the merged mixed-species CSV, which contains all
# maize-train rows and would leak maize into the "soy" backbone. All images
# resolve from the shared image directory. z-score stats are refit per dataset.
#
# Defaults reproduce the released rootquant-v2-soybean-only / -soybean-to-maize-*
# checkpoints: v3 profile with tile-shuffle OFF (TILE_SHUFFLE=0), batch 8/GPU on
# 3 GPUs.
#
# Usage:
#   bash run_cross_species.sh                 # full pipeline, defaults below
#   SOY_EPOCHS=2 FT_EPOCHS=2 bash run_cross_species.sh   # smoke (tiny)
#
# Everything below is env-overridable.
# =============================================================================
set -uo pipefail

# Paths derived from this script's location, so the repo can live anywhere.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARENT_DIR="${PARENT_DIR:-$(dirname "${PROJECT_DIR}")}"
cd "${PARENT_DIR}"

# ── Experiment config ────────────────────────────────────────────────────
PROFILE="${PROFILE:-dinov3_dora_mona_768_v3}"
SOY_EPOCHS="${SOY_EPOCHS:-30}"     # 80k soy images need >=30 epochs to converge; best.pt caps the peak
FT_EPOCHS="${FT_EPOCHS:-60}"        # maize is small (~9k); more epochs, best.pt caps the peak
BATCH="${BATCH:-8}"                 # per-GPU; 896px
GRAD_CKPT="${GRAD_CKPT:-0}"         # 0=OFF (faster; measured 36.8GB peak @ B8/896, fits 80GB); 1=ON
TILE_SHUFFLE="${TILE_SHUFFLE:-0}"   # 0=OFF (as the released cross-species runs); 1=ON (profile default)

# Maize runs are SEQUENTIAL (FCFT then FFT), each on FT_GPUS — the dataset is
# small (~9k), so running them one after the other is quick.
SOY_GPUS="${SOY_GPUS:-0,1,2}"
FT_GPUS="${FT_GPUS:-0,1,2}"
EVAL_GPU="${EVAL_GPU:-0}"

# ── Data paths ───────────────────────────────────────────────────────────
# See "Data layout" in the README. ROOTQUANT_SOY_DIR / ROOTQUANT_MAIZE_DIR hold
# the per-species split CSVs; images resolve from the shared image directory.
# ROOTQUANT_DATA_DIR is only needed for whichever of the three is not set.
DATA_ROOT="${ROOTQUANT_DATA_DIR:-}"
if [[ -z "${DATA_ROOT}" && ( -z "${ROOTQUANT_SOY_DIR:-}" || -z "${ROOTQUANT_MAIZE_DIR:-}" || -z "${ROOTQUANT_IMAGES:-}" ) ]]; then
  echo "[xspecies] set ROOTQUANT_DATA_DIR, or all of ROOTQUANT_SOY_DIR, ROOTQUANT_MAIZE_DIR and ROOTQUANT_IMAGES."
  exit 1
fi
SOY_DIR="${ROOTQUANT_SOY_DIR:-${DATA_ROOT}/soy}"
MAIZE_DIR="${ROOTQUANT_MAIZE_DIR:-${DATA_ROOT}/maize}"

IMAGES_NFS="${ROOTQUANT_IMAGES:-${DATA_ROOT}/images}"
SHM_DIR="${SHM_DIR:-/dev/shm/rq_images}"
STAGE_SHM="${STAGE_SHM:-1}"

SOY_TRAIN="${SOY_TRAIN:-${SOY_DIR}/train_data.csv}"
SOY_VAL="${SOY_VAL:-${SOY_DIR}/val_data.csv}"
SOY_TEST="${SOY_TEST:-${SOY_DIR}/test_data.csv}"
MAIZE_TRAIN="${MAIZE_TRAIN:-${MAIZE_DIR}/train_data.csv}"
MAIZE_VAL="${MAIZE_VAL:-${MAIZE_DIR}/val_data.csv}"
MAIZE_TEST="${MAIZE_TEST:-${MAIZE_DIR}/test_data.csv}"

# ── Run names / outputs ──────────────────────────────────────────────────
RUN_SOY="${RUN_SOY:-xspecies_soy_v3}"
RUN_FCFT="${RUN_FCFT:-xspecies_fcft_v3}"
RUN_FFT="${RUN_FFT:-xspecies_fft_v3}"
CKPT_DIR="${PROJECT_DIR}/runs/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/runs/xspecies_results"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# ── Environment ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${DINOV3_CHECKPOINT_PATH:-}" ]]; then
  export DINOV3_CHECKPOINT_PATH="${PROJECT_DIR}/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
fi
echo "[xspecies] DINOV3_CHECKPOINT_PATH=${DINOV3_CHECKPOINT_PATH}"

# ── Stage images to /dev/shm (one-time; page-cache shared across ranks) ───
DATA_PATH="${IMAGES_NFS}"
if [[ "${STAGE_SHM}" == "1" ]]; then
  if [[ -f "${SHM_DIR}/.stage_complete" ]]; then
    DATA_PATH="${SHM_DIR}"
    echo "[xspecies] using staged images at ${SHM_DIR}"
  elif [[ -d "${IMAGES_NFS}" ]]; then
    echo "[xspecies] staging ${IMAGES_NFS} -> ${SHM_DIR} (one-time copy) ..."
    mkdir -p "${SHM_DIR}"
    if rsync -a "${IMAGES_NFS}/" "${SHM_DIR}/"; then
      touch "${SHM_DIR}/.stage_complete"
      DATA_PATH="${SHM_DIR}"
      echo "[xspecies] staged to ${SHM_DIR}"
    else
      echo "[xspecies] WARNING: staging failed; using NFS ${IMAGES_NFS}"
    fi
  fi
fi
echo "[xspecies] DATA_PATH=${DATA_PATH}"

STAMP="$(date +%Y%m%d_%H%M%S)"

# ── helper: run one training job (foreground) ─────────────────────────────
train_job () {
  local gpus="$1"; shift
  local nproc; nproc="$(awk -F',' '{print NF}' <<<"${gpus}")"
  local port=$(( 29500 + RANDOM % 1000 ))
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun \
    --nproc_per_node="${nproc}" --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${port}" \
    -m RootQuantV2.train "$@"
}

common_train_args=(--profile "${PROFILE}" --data_path "${DATA_PATH}"
                   --per_gpu_batch "${BATCH}" --grad_checkpointing "${GRAD_CKPT}")
if [[ "${TILE_SHUFFLE}" == "1" ]]; then
  common_train_args+=(--use_tile_shuffle)
else
  common_train_args+=(--no-use_tile_shuffle)
fi

# ═════════════════════════════ Phase 1: SOY (ZST) ════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " PHASE 1 (ZST): soybean training — GPUs ${SOY_GPUS}, ${SOY_EPOCHS} epochs"
echo "════════════════════════════════════════════════════════════════"
SOY_LOG="${LOG_DIR}/xspecies_soy_${STAMP}.log"
train_job "${SOY_GPUS}" \
  "${common_train_args[@]}" \
  --run_name "${RUN_SOY}" --num_epochs "${SOY_EPOCHS}" \
  --train_csv "${SOY_TRAIN}" --val_csv "${SOY_VAL}" --test_csv "${SOY_TEST}" \
  2>&1 | tee "${SOY_LOG}"
SOY_RC=${PIPESTATUS[0]}

SOY_BEST="${CKPT_DIR}/${RUN_SOY}/best.pt"
if [[ "${SOY_RC}" -ne 0 || ! -f "${SOY_BEST}" ]]; then
  echo "[xspecies] FATAL: soy phase failed (rc=${SOY_RC}) or best.pt missing at ${SOY_BEST}. Aborting."
  exit 1
fi
echo "[xspecies] soy done. checkpoint: ${SOY_BEST}"

# ══════════════════════ Phase 2: FCFT then FFT on MAIZE ══════════════════
# Sequential — each run uses all of FT_GPUS.
FCFT_LOG="${LOG_DIR}/xspecies_fcft_${STAMP}.log"
FFT_LOG="${LOG_DIR}/xspecies_fft_${STAMP}.log"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " PHASE 2a (FCFT): freeze all but head, FT on maize — GPUs ${FT_GPUS}, ${FT_EPOCHS} epochs"
echo "════════════════════════════════════════════════════════════════"
train_job "${FT_GPUS}" \
  "${common_train_args[@]}" \
  --run_name "${RUN_FCFT}" --num_epochs "${FT_EPOCHS}" \
  --train_csv "${MAIZE_TRAIN}" --val_csv "${MAIZE_VAL}" --test_csv "${MAIZE_TEST}" \
  --init_from "${SOY_BEST}" --finetune_mode fc_only \
  2>&1 | tee "${FCFT_LOG}"
FCFT_RC=${PIPESTATUS[0]}
echo "[xspecies] FCFT finished (rc=${FCFT_RC})"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " PHASE 2b (FFT): full fine-tune on maize — GPUs ${FT_GPUS}, ${FT_EPOCHS} epochs"
echo "════════════════════════════════════════════════════════════════"
train_job "${FT_GPUS}" \
  "${common_train_args[@]}" \
  --run_name "${RUN_FFT}" --num_epochs "${FT_EPOCHS}" \
  --train_csv "${MAIZE_TRAIN}" --val_csv "${MAIZE_VAL}" --test_csv "${MAIZE_TEST}" \
  --init_from "${SOY_BEST}" --finetune_mode full \
  2>&1 | tee "${FFT_LOG}"
FFT_RC=${PIPESTATUS[0]}
echo "[xspecies] FFT finished (rc=${FFT_RC})"

# ═══════════════════════════ Phase 3: EVAL ═══════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " PHASE 3: evaluation (GPU ${EVAL_GPU})"
echo "════════════════════════════════════════════════════════════════"
eval_ckpt () {
  local ckpt="$1" csv="$2" tag="$3"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[xspecies] SKIP eval ${tag}: checkpoint missing (${ckpt})"
    return
  fi
  echo "---- eval ${tag} : $(basename "$(dirname "${ckpt}")")/$(basename "${ckpt}") on $(basename "${csv}") ----"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python -m RootQuantV2.infer \
    --checkpoint "${ckpt}" --csv "${csv}" --data_path "${DATA_PATH}" \
    --output_csv "${RESULTS_DIR}/${tag}_preds.csv" 2>&1 | tee "${RESULTS_DIR}/${tag}.txt"
}

eval_ckpt "${SOY_BEST}"                  "${SOY_TEST}"   "zst_soy_on_soy"
eval_ckpt "${SOY_BEST}"                  "${MAIZE_TEST}" "zst_soy_on_maize"
eval_ckpt "${CKPT_DIR}/${RUN_FCFT}/best.pt" "${MAIZE_TEST}" "fcft_on_maize"
eval_ckpt "${CKPT_DIR}/${RUN_FFT}/best.pt"  "${MAIZE_TEST}" "fft_on_maize"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SUMMARY  (R2 on present rows; full metrics in ${RESULTS_DIR})"
echo "════════════════════════════════════════════════════════════════"
for tag in zst_soy_on_soy zst_soy_on_maize fcft_on_maize fft_on_maize; do
  f="${RESULTS_DIR}/${tag}.txt"
  if [[ -f "${f}" ]]; then
    echo "• ${tag}"
    grep -E "n_total|length|area" "${f}" | sed 's/^/    /'
  fi
done
echo ""
echo "[xspecies] PIPELINE COMPLETE. rc: soy=${SOY_RC} fcft=${FCFT_RC} fft=${FFT_RC}"
