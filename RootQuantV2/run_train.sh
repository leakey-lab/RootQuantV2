#!/usr/bin/env bash
# Launch RootQuantV2 DinoV3 training inside a tmux session.
#
# All training arguments live in the CONFIG block below. Edit them here
# (or override via env var before invoking, e.g. `BATCH=8 bash run_train.sh`).
# Anything left empty is NOT passed to train.py, so the value from
# RootQuantV2/config.py is used.
#
# Usage:
#   GPUS=0,1,2,3 bash run_train.sh
#
# Requires tmux, and rsync when STAGE_SHM=1 (the default). Dataset paths come
# from ROOTQUANT_DATA_DIR / ROOTQUANT_IMAGES or TRAIN_CSV / VAL_CSV / TEST_CSV /
# DATA_PATH; they are resolved here and passed to train.py as flags.
#
# Tmux helpers:
#   tmux ls
#   tmux attach -t <session>
#   tmux kill-session -t <session>

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Launcher config
# ──────────────────────────────────────────────────────────────────────
# GPUS is prompted interactively if not already set in the environment.
GPUS="${GPUS:-}"
SESSION="${SESSION:-}"                # tmux session name (auto if empty)

# ──────────────────────────────────────────────────────────────────────
# train.py arguments. Leave empty to fall back to RootQuantV2/config.py.
# ──────────────────────────────────────────────────────────────────────
# DEFAULTS = the released headline run (rootquant-v2-weights):
#   profile dinov3_dora_mona_768_v3 →
#     896px · DoRA-attn (r=32) + Mona conv on MLP · readout=both (extensive
#     density ⊕ global pool) · softplus GeM · EMA · D4 TTA at eval · fp32 with
#     TF32 matmuls · zscore targets · huber(δ=3) R²-aligned loss · task_weights
#     (1, 1.5) · tile-shuffle (grids 2/4/8 at p 0.3/0.2/0.1).
#   + 20 epochs · per-GPU batch 4, on 4 GPUs (GPUS=0,1,2,3). See "Training" in
#     the README for the exact history of the released run.
# Every value below is env-overridable (e.g. `BATCH=8 bash run_train.sh`). Set
# PROFILE=dinov3_dora_mona_768_v2 for the 768px variant, or dinov3_dora for the
# 640px DoRA-only baseline.
PROFILE="${PROFILE:-dinov3_dora_mona_768_v3}"
ABLATE="${ABLATE:-}"                  # e.g. present-only, 768-only, learned-pool

# Data paths
TRAIN_CSV="${TRAIN_CSV:-}"
VAL_CSV="${VAL_CSV:-}"
TEST_CSV="${TEST_CSV:-}"
DATA_PATH="${DATA_PATH:-}"

# Output
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_NAME="${RUN_NAME:-}"

# Training hyperparameters
NUM_EPOCHS="${NUM_EPOCHS:-20}"        # headline run; EMA + best.pt capture the peak
# per_gpu_batch. Effective batch = BATCH x num_gpus x GRAD_ACCUM_STEPS.
# Default 4 = the headline run. BATCH= (empty) uses the profile's own
# per_gpu_batch from config.py (8 at 896px, 8 at 768px, 16 at 640px). fp32 896px
# with grad-checkpointing OFF peaks around 37 GB per GPU at batch 8.
BATCH="${BATCH-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR_DORA="${LR_DORA:-}"
LR_HEAD="${LR_HEAD:-}"
LR_MONA="${LR_MONA:-}"
HEAD_DROPOUT="${HEAD_DROPOUT:-}"
W_EMPTY="${W_EMPTY:-}"                # 0 = present-only loss ablation
W_PRESENT="${W_PRESENT:-}"
TARGET_SIZE="${TARGET_SIZE:-}"        # override letterbox canvas (e.g. 768)
POOL="${POOL:-}"                      # e.g. cls_attn_gem
AMP="${AMP:-}"                        # fp32 (default) or bf16

# Tile-shuffle augmentation (train-only). Empty = the profile default (on for
# the 768/896px profiles, off for dinov3_dora); 1 forces it on, 0 forces it off.
# Grids and per-grid probs default to config.py if left empty. For independent
# per-grid firing, give one prob per grid (e.g. GRIDS=2,4,8 P=0.2,0.2,0.2).
TILE_SHUFFLE="${TILE_SHUFFLE:-}"               # 1 = on, 0 = off, empty = profile
TILE_SHUFFLE_GRIDS="${TILE_SHUFFLE_GRIDS:-}"   # e.g. 2,4,8
TILE_SHUFFLE_P="${TILE_SHUFFLE_P:-}"           # e.g. 0.2,0.2,0.2 (per grid)

# Misc
SEED="${SEED:-}"
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"            # warm-start adapter / pooler / head weights from a checkpoint
FINETUNE_MODE="${FINETUNE_MODE:-}"    # fc_only | full  (cross-dataset FT; needs INIT_FROM)
GRAD_CKPT="${GRAD_CKPT:-}"            # 0 = checkpointing OFF (faster, more VRAM); 1 = ON

# /dev/shm image staging. The dataset does a cold open() per __getitem__ every
# epoch; on a network FS that is millions of small-file round-trips. Staging the
# (small) image corpus to local tmpfs makes every read a page-cache hit shared
# across all ranks+workers — typically the single largest wall-clock win.
STAGE_SHM="${STAGE_SHM:-1}"           # 0 to disable staging (read straight from the source FS)
SHM_DIR="${SHM_DIR:-/dev/shm/rq_images}"
# Source image directory to stage FROM. Defaults to $ROOTQUANT_IMAGES, else
# $ROOTQUANT_DATA_DIR/images; it is also the DATA_PATH used when not staging.
NFS_IMAGES="${NFS_IMAGES:-${ROOTQUANT_IMAGES:-${ROOTQUANT_DATA_DIR:+${ROOTQUANT_DATA_DIR}/images}}}"
RESTAGE="${RESTAGE:-}"               # 1 to force a re-copy even if already staged

# ──────────────────────────────────────────────────────────────────────
# Paths — derived from this script's location, so the repo can live anywhere.
#   PROJECT_DIR = <repo>/RootQuantV2   (the python package)
#   PARENT_DIR  = <repo>               (`python -m RootQuantV2.*` runs from here)
# ──────────────────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PARENT_DIR="${PARENT_DIR:-$(dirname "${PROJECT_DIR}")}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# ──────────────────────────────────────────────────────────────────────
# Prompt for GPUs (always required, even if other args fall through to config.py)
# ──────────────────────────────────────────────────────────────────────
if [[ -z "${GPUS}" ]]; then
  read -rp "GPU IDs (comma-separated, e.g. 0,1,3): " GPUS
fi
if [[ -z "${GPUS}" ]]; then
  echo "[run_train] No GPUs provided. Aborting."
  exit 1
fi
NPROC=$(awk -F',' '{print NF}' <<<"${GPUS}")

if [[ -n "${BATCH}" ]] && { ! [[ "${BATCH}" =~ ^[0-9]+$ ]] || [[ "${BATCH}" -le 0 ]]; }; then
  echo "[run_train] BATCH must be a positive integer. Got: '${BATCH}'."
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[run_train] tmux not found. Install it, or launch train.py directly:"
  echo "             python -m torch.distributed.run --nproc_per_node=N -m RootQuantV2.train ..."
  exit 1
fi
PY="$(command -v python || command -v python3 || true)"
if [[ -z "${PY}" ]]; then
  echo "[run_train] no python on PATH; activate the environment first."
  exit 1
fi

if [[ -z "${SESSION}" ]]; then
  _prof_tag="${PROFILE:-cfg}"
  SESSION="rqv2_${_prof_tag}_bs${BATCH:-cfg}_$(tr ',' '_' <<<"${GPUS}")"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[run_train] tmux session '${SESSION}' already exists. Kill it first:"
  echo "             tmux kill-session -t ${SESSION}"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Stage images to /dev/shm (local tmpfs) so __getitem__ hits page cache
# instead of a cold NFS open every epoch. Idempotent via a sentinel file.
# Must run BEFORE the argv build below so DATA_PATH is forwarded. Any failure
# degrades gracefully to the NFS path (do not abort under `set -e`).
# ──────────────────────────────────────────────────────────────────────
if [[ "${STAGE_SHM}" == "1" && -z "${DATA_PATH}" ]]; then
  if [[ -f "${SHM_DIR}/.stage_complete" && -z "${RESTAGE}" ]]; then
    DATA_PATH="${SHM_DIR}"
    echo "[run_train] using staged images at ${SHM_DIR}"
  elif [[ -d "${NFS_IMAGES}" ]]; then
    echo "[run_train] staging images ${NFS_IMAGES} -> ${SHM_DIR} (one-time copy) ..."
    mkdir -p "${SHM_DIR}"
    if rsync -a "${NFS_IMAGES}/" "${SHM_DIR}/"; then
      touch "${SHM_DIR}/.stage_complete"
      DATA_PATH="${SHM_DIR}"
      echo "[run_train] staged to ${SHM_DIR}"
    else
      echo "[run_train] WARNING: staging failed; falling back to NFS ${NFS_IMAGES}"
      DATA_PATH="${NFS_IMAGES}"
    fi
  else
    echo "[run_train] WARNING: image directory '${NFS_IMAGES}' not found; not staging"
  fi
fi

# Resolve the dataset here and pass it to train.py as flags: a tmux server that
# is already running does not inherit this shell's environment, so train.py
# cannot rely on ROOTQUANT_DATA_DIR / ROOTQUANT_IMAGES inside the session.
if [[ -n "${ROOTQUANT_DATA_DIR:-}" ]]; then
  TRAIN_CSV="${TRAIN_CSV:-${ROOTQUANT_DATA_DIR}/train_data.csv}"
  VAL_CSV="${VAL_CSV:-${ROOTQUANT_DATA_DIR}/val_data.csv}"
  TEST_CSV="${TEST_CSV:-${ROOTQUANT_DATA_DIR}/test_data.csv}"
fi
DATA_PATH="${DATA_PATH:-${NFS_IMAGES}}"
for _var in TRAIN_CSV VAL_CSV TEST_CSV DATA_PATH; do
  if [[ -z "${!_var}" || ! -e "${!_var}" ]]; then
    echo "[run_train] ${_var}='${!_var}' not found. Set ROOTQUANT_DATA_DIR (see"
    echo "             'Data layout' in the README) or ${_var} directly."
    exit 1
  fi
done

# ──────────────────────────────────────────────────────────────────────
# Build the train.py argv. Only non-empty vars are forwarded; the rest
# fall through to config.py defaults.
# ──────────────────────────────────────────────────────────────────────
ARGS=()
[[ -n "${PROFILE}"          ]] && ARGS+=(--profile "${PROFILE}")
[[ -n "${ABLATE}"           ]] && ARGS+=(--ablate "${ABLATE}")
[[ -n "${TRAIN_CSV}"        ]] && ARGS+=(--train_csv "${TRAIN_CSV}")
[[ -n "${VAL_CSV}"          ]] && ARGS+=(--val_csv "${VAL_CSV}")
[[ -n "${TEST_CSV}"         ]] && ARGS+=(--test_csv "${TEST_CSV}")
[[ -n "${DATA_PATH}"        ]] && ARGS+=(--data_path "${DATA_PATH}")
[[ -n "${OUTPUT_ROOT}"      ]] && ARGS+=(--output_root "${OUTPUT_ROOT}")
[[ -n "${RUN_NAME}"         ]] && ARGS+=(--run_name "${RUN_NAME}")
[[ -n "${NUM_EPOCHS}"       ]] && ARGS+=(--num_epochs "${NUM_EPOCHS}")
[[ -n "${BATCH}"            ]] && ARGS+=(--per_gpu_batch "${BATCH}")
[[ -n "${GRAD_ACCUM_STEPS}" ]] && ARGS+=(--grad_accum_steps "${GRAD_ACCUM_STEPS}")
[[ -n "${LR_DORA}"          ]] && ARGS+=(--lr_dora "${LR_DORA}")
[[ -n "${LR_HEAD}"          ]] && ARGS+=(--lr_head "${LR_HEAD}")
[[ -n "${LR_MONA}"          ]] && ARGS+=(--lr_mona "${LR_MONA}")
[[ -n "${HEAD_DROPOUT}"     ]] && ARGS+=(--head_dropout "${HEAD_DROPOUT}")
[[ -n "${W_EMPTY}"          ]] && ARGS+=(--w_empty "${W_EMPTY}")
[[ -n "${W_PRESENT}"        ]] && ARGS+=(--w_present "${W_PRESENT}")
[[ -n "${TARGET_SIZE}"      ]] && ARGS+=(--target_size "${TARGET_SIZE}")
[[ -n "${POOL}"             ]] && ARGS+=(--pool "${POOL}")
[[ -n "${AMP}"              ]] && ARGS+=(--amp "${AMP}")
[[ "${TILE_SHUFFLE}" == "1" ]] && ARGS+=(--use_tile_shuffle)
[[ "${TILE_SHUFFLE}" == "0" ]] && ARGS+=(--no-use_tile_shuffle)
[[ -n "${TILE_SHUFFLE_GRIDS}" ]] && ARGS+=(--tile_shuffle_grids "${TILE_SHUFFLE_GRIDS}")
[[ -n "${TILE_SHUFFLE_P}"   ]] && ARGS+=(--tile_shuffle_p "${TILE_SHUFFLE_P}")
[[ -n "${GRAD_CKPT}"        ]] && ARGS+=(--grad_checkpointing "${GRAD_CKPT}")
[[ -n "${SEED}"             ]] && ARGS+=(--seed "${SEED}")
[[ -n "${RESUME}"           ]] && ARGS+=(--resume "${RESUME}")
[[ -n "${INIT_FROM}"        ]] && ARGS+=(--init_from "${INIT_FROM}")
[[ -n "${FINETUNE_MODE}"    ]] && ARGS+=(--finetune_mode "${FINETUNE_MODE}")

# Quote each arg for safe embedding in the tmux send-keys string.
ARGS_STR=""
for a in "${ARGS[@]:-}"; do
  ARGS_STR+=" $(printf '%q' "${a}")"
done

PORT=$(( 29500 + RANDOM % 1000 ))
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${PROFILE:-dinov3_dora}_bs${BATCH:-cfg}_gpus${GPUS//,/}_${STAMP}.log"

# The backbone loader (backbone/dinov3.py) reads DINOV3_CHECKPOINT_PATH when set,
# else RootQuantV2/dinov3/<file>. Check here so a missing file fails before tmux
# starts; the variable itself is forwarded into the session below.
_DEFAULT_DINOV3_CKPT="${PROJECT_DIR}/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
if [[ -n "${DINOV3_CHECKPOINT_PATH:-}" && ! -f "${DINOV3_CHECKPOINT_PATH}" ]]; then
  echo "[run_train] DINOV3_CHECKPOINT_PATH='${DINOV3_CHECKPOINT_PATH}' is not a file."
  exit 1
fi
if [[ -z "${DINOV3_CHECKPOINT_PATH:-}" && ! -f "${_DEFAULT_DINOV3_CKPT}" ]]; then
  echo "[run_train] DINOv3 backbone not found. Put dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  echo "             in ${PROJECT_DIR}/dinov3/ or set DINOV3_CHECKPOINT_PATH"
  echo "             (see RootQuantV2/dinov3/README.md)."
  exit 1
fi

DINOV3_CKPT_ENV=""
if [[ -n "${DINOV3_CHECKPOINT_PATH:-}" ]]; then
  DINOV3_CKPT_ENV="DINOV3_CHECKPOINT_PATH=$(printf '%q' "${DINOV3_CHECKPOINT_PATH}") "
fi

# expandable_segments cuts allocator fragmentation (helps keep the peak from
# creeping past the limit on a shared card). Override with EXPANDABLE_SEGMENTS=0
# to disable it (the allocator note claims a small speed edge with it off, at the
# cost of more fragmentation near the memory limit).
EXPANDABLE_SEGMENTS="${EXPANDABLE_SEGMENTS:-1}"
if [[ "${EXPANDABLE_SEGMENTS}" == "0" ]]; then
  ALLOC_CONF_ENV=""
  echo "[run_train] expandable_segments DISABLED (PYTORCH_CUDA_ALLOC_CONF unset)"
else
  ALLOC_CONF_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
fi
# Absolute python path: the tmux session may not have this shell's venv on PATH.
CMD="cd $(printf '%q' "${PARENT_DIR}") && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
${ALLOC_CONF_ENV}\
MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT} \
CUDA_VISIBLE_DEVICES=${GPUS} \
${DINOV3_CKPT_ENV}\
$(printf '%q' "${PY}") -m torch.distributed.run --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=${PORT} \
  -m RootQuantV2.train${ARGS_STR} 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

tmux new-session -d -s "${SESSION}" -c "${PARENT_DIR}"
tmux send-keys -t "${SESSION}" "${CMD}" ENTER

echo "[run_train] launched."
echo "  session : ${SESSION}"
echo "  gpus    : ${GPUS}  (nproc_per_node=${NPROC})"
echo "  batch   : ${BATCH:-<from config.py>}"
echo "  port    : ${PORT}"
echo "  log     : ${LOG_FILE}"
echo "  args    :${ARGS_STR:-  <all from config.py>}"
echo
echo "  attach  : tmux attach -t ${SESSION}"
echo "  kill    : tmux kill-session -t ${SESSION}"
