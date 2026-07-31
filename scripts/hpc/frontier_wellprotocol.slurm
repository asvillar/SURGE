#!/bin/bash
# TheWell-protocol scaling sweep — OLCF Frontier (4x MI250X = 8 GCDs/node).
#
# Frontier exposes each MI250X as two GCDs, so ONE node runs the entire
# 8-point factorial concurrently — the whole sweep is ~1 node-day.
#
# EDIT BEFORE FIRST USE: -A project, and the environment block below.
#
#   sbatch scripts/hpc/frontier_wellprotocol.slurm
#
# ROCm caveats handled here: per-process MIOpen cache directories (concurrent
# ranks corrupt a shared one) and ROCR_VISIBLE_DEVICES for GCD pinning.
# Verify `python -c "import torch; print(torch.cuda.is_available())"` on a
# compute node before trusting a long submission — the ROCm PyTorch build,
# h5py and the_well need checking on this toolchain.

#SBATCH -A CHANGEME
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J well-sweep
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --requeue

set -uo pipefail

SURGE_ROOT="${SURGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$SURGE_ROOT" || exit 1
echo "SURGE_ROOT: $SURGE_ROOT"

# --- environment (EDIT) ----------------------------------------------------
module load PrgEnv-gnu rocm 2>/dev/null || echo "WARN: module load failed"
# source "$SURGE_ROOT/.venv-rocm/bin/activate"
export PYTHONPATH="$SURGE_ROOT:${PYTHONPATH:-}"

# --- data: Frontier compute nodes cannot see $HOME — stage on Lustre -------
export WELL_DATA_ROOT="${WELL_DATA_ROOT:-$MEMBERWORK/$SLURM_JOB_ACCOUNT/thewell/datasets}"
if [ ! -d "$WELL_DATA_ROOT/turbulent_radiative_layer_2D/data/train" ]; then
  echo "ERROR: dataset not staged at $WELL_DATA_ROOT (6 GB, copy once)"
  exit 1
fi

OUT_DIR="${OUT_DIR:-$MEMBERWORK/$SLURM_JOB_ACCOUNT/well_sweep}"
mkdir -p "$OUT_DIR" logs

TIME_LIMIT_S=$(( 11*3600 + 40*60 ))
export OMP_NUM_THREADS=7

source "$SURGE_ROOT/scripts/hpc/grid.sh"

# --- all 8 configs, one per GCD -------------------------------------------
echo "[node] running ${#CONFIGS_LONGEST_FIRST[@]} configs on 8 GCDs"
pids=()
for i in "${!CONFIGS_LONGEST_FIRST[@]}"; do
  # each rank needs its own MIOpen kernel cache or they clobber each other
  export MIOPEN_USER_DB_PATH="/tmp/miopen-$SLURM_JOB_ID-$i"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  mkdir -p "$MIOPEN_USER_DB_PATH"
  ROCR_VISIBLE_DEVICES=$i run_config "${CONFIGS_LONGEST_FIRST[$i]}" "$i" \
    "$TIME_LIMIT_S" "$OUT_DIR" &
  pids+=($!)
done

incomplete=0
for pid in "${pids[@]}"; do
  wait "$pid" || incomplete=1
done

MAX_CHAIN="${MAX_CHAIN:-4}"
CHAIN="${CHAIN:-1}"
if [ "$incomplete" = "1" ]; then
  if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
    echo "[chain] resubmitting ($((CHAIN+1))/$MAX_CHAIN)"
    CHAIN=$((CHAIN+1)) OUT_DIR="$OUT_DIR" \
      sbatch "$SURGE_ROOT/scripts/hpc/frontier_wellprotocol.slurm"
  else
    echo "[chain] MAX_CHAIN reached with work outstanding — inspect results."
  fi
else
  echo "[done] all 8 configs completed"
fi

grep -H '"best"' "$OUT_DIR"/*/result.json 2>/dev/null || true
