# Shared sweep definition — sourced by both submit scripts so Perlmutter and
# Frontier run the SAME experiment. Edit here, nowhere else.
#
# A 2x2x2 factorial over the three confounded axes:
#   DATA      1400  (laptop study)  vs  6984 (full train split)
#   CAPACITY  3.4M  (base=40 d=3)   vs  19.8M (base=64 d=4, published class)
#   EPOCHS    60    (laptop study)  vs  500   (published 12h-H100 class)
#
# A_control reproduces the 0.486 laptop result on GPU (sanity check that the
# port changed nothing); H_full is the published-scale target. The six points
# between them give main effects AND interactions — which is the whole reason
# to run a factorial instead of just scaling everything at once.
#
# Format: tag|n_train|base_channels|depth|epochs

CONFIGS=(
  "A_control|1400|40|3|60"
  "B_data|6984|40|3|60"
  "C_capacity|1400|64|4|60"
  "D_epochs|1400|40|3|500"
  "E_data_capacity|6984|64|4|60"
  "F_data_epochs|6984|40|3|500"
  "G_capacity_epochs|1400|64|4|500"
  "H_full|6984|64|4|500"
)

# Order matters: the two longest configs (H, F) go on GPU 0 and 1 so they
# start immediately rather than queueing behind short work.
CONFIGS_LONGEST_FIRST=(
  "H_full|6984|64|4|500"
  "F_data_epochs|6984|40|3|500"
  "G_capacity_epochs|1400|64|4|500"
  "E_data_capacity|6984|64|4|60"
  "D_epochs|1400|40|3|500"
  "B_data|6984|40|3|60"
  "C_capacity|1400|64|4|60"
  "A_control|1400|40|3|60"
)

run_config() {
  # run_config <config-string> <gpu-index> <time-limit-s> <out-dir>
  local cfg="$1" gpu="$2" tlimit="$3" outdir="$4"
  IFS='|' read -r tag ntrain base depth epochs <<< "$cfg"
  echo "[launch] gpu=$gpu $tag n_train=$ntrain base=$base depth=$depth epochs=$epochs"
  python "$SURGE_ROOT/scripts/hpc/wellprotocol_sweep.py" \
    --tag "$tag" \
    --n-train "$ntrain" \
    --base-channels "$base" \
    --depth "$depth" \
    --epochs "$epochs" \
    --out-dir "$outdir" \
    --time-limit-s "$tlimit" \
    --resume \
    > "$outdir/${tag}.log" 2>&1
  echo "$tag exited $?"
}
