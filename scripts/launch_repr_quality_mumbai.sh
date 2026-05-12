#!/bin/bash
# Representation-quality probes launcher for the 6 weekend+production checkpoints
# on the Mumbai p5.48xlarge (8 x H100).  Each checkpoint runs on its own GPU
# so the whole sweep finishes in ~one wall-clock run of the slowest probe pass.
#
# Usage:
#   ./launch_repr_quality_mumbai.sh launch a       # Tier A only
#   ./launch_repr_quality_mumbai.sh launch ab      # Tier A + B
#   ./launch_repr_quality_mumbai.sh status         # Show per-GPU status
#
# Outputs go to ${OUT_ROOT}/<label>_tier_<tier>.csv .

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
RUN_ROOT=/opt/dlami/nvme/eeg/runs
OUT_ROOT=/opt/dlami/nvme/eeg/runs/repr_quality
LOG_ROOT=/home/ubuntu/transfer_logs

# (gpu_index, model_dir, label) tuples
CKPTS=(
  "0|${RUN_ROOT}/toto2_eeg_exp36_triple_aux|exp36_triple_aux_may7"
  "1|${RUN_ROOT}/toto2_eeg_exp48_mr_mpl_long|exp48_mr_mpl_long_60k"
  "2|${RUN_ROOT}/toto2_eeg_exp48_mr_mpl_long2|exp48_long2_150k_noCAR"
  "3|${RUN_ROOT}/toto2_eeg_exp50_rest_car_long|exp50_long_150k_CAR"
  "4|${RUN_ROOT}/toto2_eeg_exp50_long_seed55|exp50_long_seed55_80k"
  "5|${RUN_ROOT}/toto2_eeg_exp51_chain_long|exp51_chain_long_80k"
)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 missing"; exit 1; }

run_one() {
  local gpu=$1
  local mdir=$2
  local label=$3
  local tier=$4

  local out="${OUT_ROOT}/${label}_tier_${tier}.csv"
  local log="${LOG_ROOT}/repr_quality_${label}_tier_${tier}.log"
  local pidf="${LOG_ROOT}/repr_quality_gpu${gpu}.pid"

  if [[ ! -d "$mdir" || ! -f "$mdir/config.json" || ! -f "$mdir/model.safetensors" ]]; then
    echo "  SKIP $label : missing $mdir" | tee -a "$log"
    return
  fi

  echo "  [GPU $gpu] $label  tier=$tier"
  nohup bash -c "
    set -u
    cd $REPO/toto2
    export PYTHONPATH=$REPO/toto2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo '=== [GPU $gpu / $label / $tier] start ==='
    date -u +%FT%TZ
    CUDA_VISIBLE_DEVICES=$gpu \
      $PY -u -m toto2.eval.repr_quality \
        --checkpoint '$mdir' \
        --label '$label' \
        --tier '$tier' \
        --output '$out' \
        --verbose \
        > '$log' 2>&1
    echo '=== [GPU $gpu / $label / $tier] exit=\$? ==='
    date -u +%FT%TZ
  " >> "$log" 2>&1 &
  local pid=$!
  echo $pid > "$pidf"
  disown
  echo "    PID=$pid log=$log"
}

case "${1:-launch}" in
  launch)
    tier="${2:-a}"
    echo "====================================================================="
    echo "Launching representation-quality probes (tier=$tier) on Mumbai 8xH100"
    echo "Output root: $OUT_ROOT"
    echo "====================================================================="
    for entry in "${CKPTS[@]}"; do
      IFS='|' read -r gpu mdir label <<< "$entry"
      run_one "$gpu" "$mdir" "$label" "$tier"
    done
    ;;
  status)
    echo "=== repr_quality probe status ==="
    for entry in "${CKPTS[@]}"; do
      IFS='|' read -r gpu mdir label <<< "$entry"
      pidf="${LOG_ROOT}/repr_quality_gpu${gpu}.pid"
      [[ -f "$pidf" ]] || { echo "  [GPU $gpu / $label] (no pid file)"; continue; }
      pid=$(cat "$pidf")
      out_a="${OUT_ROOT}/${label}_tier_a.csv"
      out_ab="${OUT_ROOT}/${label}_tier_ab.csv"
      out_b="${OUT_ROOT}/${label}_tier_b.csv"
      rows=0
      for csv in "$out_a" "$out_ab" "$out_b"; do
        if [[ -f "$csv" ]]; then
          r=$(tail -n +2 "$csv" 2>/dev/null | wc -l | tr -d ' ')
          rows=$((rows + r))
        fi
      done
      if kill -0 "$pid" 2>/dev/null; then
        etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        echo "  [GPU $gpu / $label] RUNNING pid=$pid elapsed=$etime rows=$rows"
      else
        echo "  [GPU $gpu / $label] DONE pid=$pid rows=$rows"
      fi
    done
    ;;
  *)
    echo "Usage: $0 {launch [a|b|ab]|status}" ; exit 2 ;;
esac
