#!/usr/bin/env bash
# =============================================================================
# run_maize_ft.sh — maize fine-tuning ONLY (resumes the cross-species pipeline
# at Phase 2, reusing an already-trained soy checkpoint).
#
# Use this when the soy (Phase 1 / ZST) run has already converged. It skips soy
# entirely and warm-starts both maize runs from that checkpoint:
#   FCFT : warm-start from soy best.pt, freeze all but the regression readout
#   FFT  : warm-start from soy best.pt, full fine-tune
#   eval : score soy->soy, soy->maize (ZST), FCFT->maize, FFT->maize
#
# FCFT and FFT run SEQUENTIALLY on FT_GPUS. Invocation matches
# run_cross_species.sh phases 2-3.
#
# By default it warm-starts from the downloaded soybean-only checkpoint,
# RootQuantV2/runs/checkpoints/rootquant-v2-soybean-only/best.pt; set
# RUN_SOY=xspecies_soy_v3 to use one trained by run_cross_species.sh.
#
# Usage:
#   bash run_maize_ft.sh
#   FT_EPOCHS=2 bash run_maize_ft.sh        # smoke
# =============================================================================
set -uo pipefail

# Paths derived from this script's location, so the repo can live anywhere.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARENT_DIR="${PARENT_DIR:-$(dirname "${PROJECT_DIR}")}"
cd "${PARENT_DIR}"

# ── Experiment config (must match the soy run's profile) ─────────────────
PROFILE="${PROFILE:-dinov3_dora_mona_768_v3}"
FT_EPOCHS="${FT_EPOCHS:-60}"
BATCH="${BATCH:-8}"
GRAD_CKPT="${GRAD_CKPT:-0}"
TILE_SHUFFLE="${TILE_SHUFFLE:-0}"   # 0=OFF (as the released cross-species runs); 1=ON (profile default)

# Each run needs ~37 GB per GPU at batch 8 / 896px. Sequential.
FT_GPUS="${FT_GPUS:-0,1,2}"
EVAL_GPU="${EVAL_GPU:-0}"

# ── Data paths ───────────────────────────────────────────────────────────
# See "Data layout" in the README; every path below is env-overridable.
# ROOTQUANT_DATA_DIR is only needed for whichever of the three is not set.
DATA_ROOT="${ROOTQUANT_DATA_DIR:-}"
if [[ -z "${DATA_ROOT}" && ( -z "${ROOTQUANT_SOY_DIR:-}" || -z "${ROOTQUANT_MAIZE_DIR:-}" || -z "${ROOTQUANT_IMAGES:-}" ) ]]; then
  echo "[maize] set ROOTQUANT_DATA_DIR, or all of ROOTQUANT_SOY_DIR, ROOTQUANT_MAIZE_DIR and ROOTQUANT_IMAGES."
  exit 1
fi
SOY_DIR="${ROOTQUANT_SOY_DIR:-${DATA_ROOT}/soy}"
MAIZE_DIR="${ROOTQUANT_MAIZE_DIR:-${DATA_ROOT}/maize}"

IMAGES_NFS="${ROOTQUANT_IMAGES:-${DATA_ROOT}/images}"
SHM_DIR="${SHM_DIR:-/dev/shm/rq_images}"

MAIZE_TRAIN="${MAIZE_TRAIN:-${MAIZE_DIR}/train_data.csv}"
MAIZE_VAL="${MAIZE_VAL:-${MAIZE_DIR}/val_data.csv}"
MAIZE_TEST="${MAIZE_TEST:-${MAIZE_DIR}/test_data.csv}"
SOY_TEST="${SOY_TEST:-${SOY_DIR}/test_data.csv}"

# ── Run names / outputs ──────────────────────────────────────────────────
RUN_SOY="${RUN_SOY:-rootquant-v2-soybean-only}"   # downloaded checkpoint (tools/download_checkpoints.py)
RUN_FCFT="${RUN_FCFT:-xspecies_fcft_v3}"
RUN_FFT="${RUN_FFT:-xspecies_fft_v3}"
CKPT_DIR="${PROJECT_DIR}/runs/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/runs/xspecies_results"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

SOY_BEST="${CKPT_DIR}/${RUN_SOY}/best.pt"
if [[ ! -f "${SOY_BEST}" ]]; then
  echo "[maize] FATAL: soy checkpoint missing at ${SOY_BEST}. Download it with"
  echo "        python tools/download_checkpoints.py rootquant-v2-soybean-only"
  echo "        or set RUN_SOY to the run name of your own soy checkpoint. Aborting."
  exit 1
fi

# ── Environment ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${DINOV3_CHECKPOINT_PATH:-}" ]]; then
  export DINOV3_CHECKPOINT_PATH="${PROJECT_DIR}/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
fi
echo "[maize] DINOV3_CHECKPOINT_PATH=${DINOV3_CHECKPOINT_PATH}"

# ── Data path (images already staged to /dev/shm by the soy run) ──────────
DATA_PATH="${IMAGES_NFS}"
if [[ -f "${SHM_DIR}/.stage_complete" ]]; then
  DATA_PATH="${SHM_DIR}"
  echo "[maize] using staged images at ${SHM_DIR}"
else
  echo "[maize] WARNING: ${SHM_DIR} not staged; using NFS ${IMAGES_NFS}"
fi
echo "[maize] DATA_PATH=${DATA_PATH}"
echo "[maize] warm-start from ${SOY_BEST}"

STAMP="$(date +%Y%m%d_%H%M%S)"
FCFT_LOG="${LOG_DIR}/maize_fcft_${STAMP}.log"
FFT_LOG="${LOG_DIR}/maize_fft_${STAMP}.log"

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

# ══════════════════════ Phase 2a: FCFT on MAIZE ══════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " FCFT: freeze all but head, FT on maize — GPUs ${FT_GPUS}, ${FT_EPOCHS} epochs"
echo "════════════════════════════════════════════════════════════════"
train_job "${FT_GPUS}" \
  "${common_train_args[@]}" \
  --run_name "${RUN_FCFT}" --num_epochs "${FT_EPOCHS}" \
  --train_csv "${MAIZE_TRAIN}" --val_csv "${MAIZE_VAL}" --test_csv "${MAIZE_TEST}" \
  --init_from "${SOY_BEST}" --finetune_mode fc_only \
  2>&1 | tee "${FCFT_LOG}"
FCFT_RC=${PIPESTATUS[0]}
echo "[maize] FCFT finished (rc=${FCFT_RC})"

# ══════════════════════ Phase 2b: FFT on MAIZE ═══════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " FFT: full fine-tune on maize — GPUs ${FT_GPUS}, ${FT_EPOCHS} epochs"
echo "════════════════════════════════════════════════════════════════"
train_job "${FT_GPUS}" \
  "${common_train_args[@]}" \
  --run_name "${RUN_FFT}" --num_epochs "${FT_EPOCHS}" \
  --train_csv "${MAIZE_TRAIN}" --val_csv "${MAIZE_VAL}" --test_csv "${MAIZE_TEST}" \
  --init_from "${SOY_BEST}" --finetune_mode full \
  2>&1 | tee "${FFT_LOG}"
FFT_RC=${PIPESTATUS[0]}
echo "[maize] FFT finished (rc=${FFT_RC})"

# ═══════════════════════════ Phase 3: EVAL ═══════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " EVAL (GPU ${EVAL_GPU})"
echo "════════════════════════════════════════════════════════════════"
eval_ckpt () {
  local ckpt="$1" csv="$2" tag="$3"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[maize] SKIP eval ${tag}: checkpoint missing (${ckpt})"
    return
  fi
  echo "---- eval ${tag} : $(basename "$(dirname "${ckpt}")")/$(basename "${ckpt}") on $(basename "${csv}") ----"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python -m RootQuantV2.infer \
    --checkpoint "${ckpt}" --csv "${csv}" --data_path "${DATA_PATH}" \
    --output_csv "${RESULTS_DIR}/${tag}_preds.csv" 2>&1 | tee "${RESULTS_DIR}/${tag}.txt"
}

eval_ckpt "${SOY_BEST}"                     "${SOY_TEST}"   "zst_soy_on_soy"
eval_ckpt "${SOY_BEST}"                     "${MAIZE_TEST}" "zst_soy_on_maize"
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
echo "[maize] PIPELINE COMPLETE. rc: fcft=${FCFT_RC} fft=${FFT_RC}"
