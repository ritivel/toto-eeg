#!/bin/bash
# Eval launcher for exp50_long_seed55 on Oregon p4d.24xlarge.
# v2: explicit dataset arg per GPU so we can stage launches as HF cache lands.
#
# Usage:
#   ./launch_weekend_eval_oregon_v2.sh launch_one <gpu> <dataset>
#   ./launch_weekend_eval_oregon_v2.sh status

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
RUN_ROOT=/opt/dlami/nvme/eeg/runs

CKPT_EXP50_SEED55="${RUN_ROOT}/checkpoints/toto2_eeg_exp50_long_seed55/epoch=21-step=80000-val_loss=0.2382-train_mr_mpl_step=2.562e-01.ckpt"

STRATS="frozen ridge_probe lora"
HEAD="linear_head"
N_SEEDS=3

OUT="${RUN_ROOT}/eval_exp50_seed55_fast"
mkdir -p "$OUT" /home/ubuntu/transfer_logs

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }
[[ -f "$CKPT_EXP50_SEED55" ]] || { echo "FATAL: ckpt not found: $CKPT_EXP50_SEED55"; exit 1; }

launch_one() {
  local gpu=$1
  local ds=$2

  local pid_file="/home/ubuntu/transfer_logs/eval_or_gpu${gpu}.pid"
  local log="${OUT}/${ds}.log"
  local csv="${OUT}/${ds}.csv"

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid=$(cat "$pid_file")
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "[GPU $gpu] already running PID $existing_pid; skipping"
      return 0
    fi
  fi

  echo "[GPU $gpu / $ds] launching"

  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$REPO/toto2" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" -u -m toto2.eval.run_benchmark_fast \
      --checkpoint "$CKPT_EXP50_SEED55" \
      --datasets "$ds" \
      --strategies $STRATS \
      --heads $HEAD \
      --n-seeds $N_SEEDS \
      --device cuda \
      --output "$csv" \
      > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "  PID=$pid log=$log csv=$csv"
  disown
}

case "${1:-help}" in
  launch_one)
    [[ $# -ge 3 ]] || { echo "Usage: $0 launch_one <gpu> <dataset>"; exit 1; }
    launch_one "$2" "$3"
    ;;
  status)
    echo "=== Oregon eval status ==="
    for pid_file in /home/ubuntu/transfer_logs/eval_or_gpu*.pid; do
      [[ -f "$pid_file" ]] || continue
      gpu=$(basename "$pid_file" .pid | sed 's/eval_or_gpu//')
      pid=$(cat "$pid_file")
      # find the ds from the running process or from the most recent log
      ds=$(ls -t "$OUT"/*.log 2>/dev/null | head -1 | xargs -I{} basename {} .log)
      # actually grep for ds via the log that was opened by this pid
      log_for_pid=$(ls -t "$OUT"/*.log 2>/dev/null | while read l; do
        ds_l=$(basename "$l" .log); csv="$OUT/${ds_l}.csv"
        # crude: show all
        echo "$ds_l"
      done | head -1)
      csv_count=0
      if kill -0 "$pid" 2>/dev/null; then
        echo "  [GPU $gpu] RUNNING pid=$pid (check log under $OUT)"
      else
        echo "  [GPU $gpu] DONE pid=$pid"
      fi
    done
    echo
    echo "=== CSV row counts ==="
    for csv in "$OUT"/*.csv; do
      [[ -f "$csv" ]] || continue
      ds=$(basename "$csv" .csv)
      rows=$(tail -n +2 "$csv" 2>/dev/null | wc -l | tr -d ' ')
      printf "  %-22s %d/9\n" "$ds" "$rows"
    done
    ;;
  *)
    echo "Usage: $0 {launch_one <gpu> <dataset>|status}"
    ;;
esac
