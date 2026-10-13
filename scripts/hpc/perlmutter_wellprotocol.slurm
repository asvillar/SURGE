#!/bin/bash
# TheWell-protocol scaling sweep — NERSC Perlmutter (4x A100 per node).
#
# Runs 4 configurations concurrently, one per GPU. Two nodes cover the full
# 8-point factorial; a single node covers half (submit twice with HALF=1 / 2).
#
# EDIT BEFORE FIRST USE: -A account, and the environment block below.
#
#   sbatch scripts/hpc/perlmutter_wellprotocol.slurm        # configs 1-4
#   HALF=2 sbatch scripts/hpc/perlmutter_wellprotocol.slurm # configs 5-8
#
# Wall clock is 11:30 against Perlmutter's 12 h regular-QOS limit; the driver
# checkpoints before the wall and exits 42, and this script resubmits itself
# (bounded by MAX_CHAIN) so long configs finish across several windows.

#SBATCH -A CHANGEME_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 32
#SBATCH -t 11:30:00
#SBATCH -J well-sweep
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --requeue

set -uo pipefail

# --- repo root: Slurm copies the script to spool, so never trust $0 --------
SURGE_ROOT="${SURGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$SURGE_ROOT" || exit 1
echo "SURGE_ROOT: $SURGE_ROOT"

# --- environment (EDIT) ----------------------------------------------------
module load pytorch/2.3.1 2>/dev/null || echo "WARN: module load failed"
# ...or activate your own env instead:
# source "$SURGE_ROOT/.venv/bin/activate"
export PYTHONPATH="$SURGE_ROOT:${PYTHONPATH:-}"

# --- data: stage to node-local or $SCRATCH, never read from $HOME ----------
export WELL_DATA_ROOT="${WELL_DATA_ROOT:-$SCRATCH/thewell/datasets}"
if [ ! -d "$WELL_DATA_ROOT/turbulent_radiative_layer_2D/data/train" ]; then
  echo "ERROR: dataset not staged at $WELL_DATA_ROOT"
  echo "  the split is 6 GB — copy it once with:"
  echo "  mkdir -p \$SCRATCH/thewell && cp -r ~/.surge/data/thewell/datasets \$SCRATCH/thewell/"
  exit 1
fi

OUT_DIR="${OUT_DIR:-$SCRATCH/well_sweep}"
mkdir -p "$OUT_DIR" logs

# 11:30 booked, leave 20 min for the final checkpoint + eval
TIME_LIMIT_S=$(( 11*3600 + 10*60 ))

# dataloader workers: 32 cores / 4 concurrent jobs, leave one for the driver
export OMP_NUM_THREADS=7

source "$SURGE_ROOT/scripts/hpc/grid.sh"

# --- pick this node's 4 configs -------------------------------------------
HALF="${HALF:-1}"
if [ "$HALF" = "1" ]; then
  MY_CONFIGS=("${CONFIGS_LONGEST_FIRST[@]:0:4}")
else
  MY_CONFIGS=("${CONFIGS_LONGEST_FIRST[@]:4:4}")
fi

echo "[node] running ${#MY_CONFIGS[@]} configs, half=$HALF, limit=${TIME_LIMIT_S}s"
pids=()
for i in "${!MY_CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i run_config "${MY_CONFIGS[$i]}" "$i" \
    "$TIME_LIMIT_S" "$OUT_DIR" &
  pids+=($!)
done

incomplete=0
for pid in "${pids[@]}"; do
  wait "$pid" || incomplete=1
done

# --- chain another window if anything paused on the wall clock ------------
MAX_CHAIN="${MAX_CHAIN:-4}"
CHAIN="${CHAIN:-1}"
if [ "$incomplete" = "1" ]; then
  if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
    echo "[chain] configs paused at the wall — resubmitting ($((CHAIN+1))/$MAX_CHAIN)"
    CHAIN=$((CHAIN+1)) HALF="$HALF" OUT_DIR="$OUT_DIR" \
      sbatch "$SURGE_ROOT/scripts/hpc/perlmutter_wellprotocol.slurm"
  else
    echo "[chain] MAX_CHAIN=$MAX_CHAIN reached with work outstanding — "
    echo "        inspect $OUT_DIR/*/result.json before resubmitting."
  fi
else
  echo "[done] all configs on half=$HALF completed"
fi

grep -H '"best"' "$OUT_DIR"/*/result.json 2>/dev/null || true
