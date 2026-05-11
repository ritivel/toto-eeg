#!/bin/bash
# Eval launcher for weekend exp50_long_seed55 reproducibility check on Oregon p4d.24xlarge.
# 1 model x 8 datasets, one dataset per A100 (in parallel).

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
RUN_ROOT=/opt/dlami/nvme/eeg/runs

CKPT_EXP50_SEED55="${RUN_ROOT}/checkpoints/toto2_eeg_exp50_long_seed55/epoch=21-step=80000-val_loss=0.2382-train_mr_mpl_step=2.562e-01.ckpt"

DATASETS=(arithmetic_zyma2019 bcic2a bcic2020_3 chbmit faced isruc_sleep mdd_mumtaz2016 physionet)
STRATS="frozen ridge_probe lora"
HEAD="linear_head"
N_SEEDS=3

OUT="${RUN_ROOT}/eval_exp50_seed55_fast"
mkdir -p "$OUT" /home/ubuntu/transfer_logs

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }

[[ -f "$CKPT_EXP50_SEED55" ]] || { echo "FATAL: ckpt not found: $CKPT_EXP50_SEED55"; exit 1; }

run_one() {
  local gpu=$1
  local ds=$2

  local pid_file="/home/ubuntu/transfer_logs/eval_or_gpu${gpu}.pid"
  local log="${OUT}/${ds}.log"
  local csv="${OUT}/${ds}.csv"

  echo "[GPU $gpu / $ds] launching"

  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$REPO/toto2" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" -u -m toto2.eval.run_benchmark_fast \
      --checkpoint "$CKPT_EXP50_SEED55" \
      --datasets $ds \
      --strategies $STRATS \
      --heads $HEAD \
      --n-seeds $N_SEEDS \
      --device cuda \
      --output "$csv" \
      > "$log" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  disown
  echo "  PID=$pid log=$log"
}

case "${1:-launch}" in
  launch)
    echo "==================================================================="
    echo "Launching exp50_long_seed55 eval on 8 A100s (1 model, 8 datasets parallel)"
    echo "  ckpt: $CKPT_EXP50_SEED55"
    echo "==================================================================="
    for i in "${!DATASETS[@]}"; do
      run_one "$i" "${DATASETS[$i]}"
    done
    ;;
  status)
    echo "=== Oregon eval status ==="
    for gpu in 0 1 2 3 4 5 6 7; do
      ds="${DATASETS[$gpu]}"
      pid_file="/home/ubuntu/transfer_logs/eval_or_gpu${gpu}.pid"
      [[ -f "$pid_file" ]] || { echo "  [GPU $gpu / $ds] (no pid file)"; continue; }
      pid=$(cat "$pid_file")
      csv="${OUT}/${ds}.csv"
      rows=0
      [[ -f "$csv" ]] && rows=$(tail -n +2 "$csv" 2>/dev/null | wc -l | tr -d ' ')
      if kill -0 "$pid" 2>/dev/null; then
        etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        echo "  [GPU $gpu / $ds] RUNNING pid=$pid elapsed=$etime  $rows/9"
      else
        echo "  [GPU $gpu / $ds] DONE pid=$pid  $rows/9"
      fi
    done
    ;;
  *)
    echo "Usage: $0 {launch|status}"
    ;;
esac
