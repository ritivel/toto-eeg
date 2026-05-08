#!/bin/bash
# --------------------------------------------------------------------
# OPTIMIZED downstream eval launcher (run_benchmark_fast.py).
#
# Speedups vs. the stock launcher (launch_eval_exp47_exp48.sh):
#   * preload=True       -> dataset loaded into RAM once (no per-batch IO)
#   * num_workers=8      -> parallel data loading
#   * batch_size=256     -> 4x GPU utilization (was 64)
#   * max_epochs=30      -> avoid the 50-epoch flat tail
#   * patience=5         -> earlier early-stopping
#   * incremental CSV    -> see results land row-by-row
#   * shared exca cache  -> kill-and-restart is essentially free
#
# Usage:
#   ./launch_eval_fast.sh exp47 <gpu> <dataset>
#   ./launch_eval_fast.sh exp48 <gpu> <dataset>
#   ./launch_eval_fast.sh exp36-lora <gpu> <dataset>
#   ./launch_eval_fast.sh status
#
# To launch a matched pair (exp47 + exp48 same dataset on different GPUs):
#   ./launch_eval_fast.sh exp47 0 chbmit
#   ./launch_eval_fast.sh exp48 1 chbmit
# --------------------------------------------------------------------

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
RUN_ROOT=/opt/dlami/nvme/eeg/runs

CKPT_EXP47="${RUN_ROOT}/checkpoints/toto2_eeg_exp47_mr_mpl_triple/epoch=6-step=30000-val_loss=0.3005-train_mr_mpl_step=4.761e-01.ckpt"
CKPT_EXP48="${RUN_ROOT}/checkpoints/toto2_eeg_exp48_mr_mpl_long/epoch=15-step=57500-val_loss=0.2875-train_mr_mpl_step=2.893e-01.ckpt"
CKPT_EXP36="${RUN_ROOT}/checkpoints/toto2_eeg_exp36_triple_aux/epoch=6-step=30000-val_loss=0.1384-train_jepa_loss_step=7.435e-03.ckpt"

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }

launch_one() {
  local exp_tag=$1
  local gpu=$2
  local dataset=$3
  local ckpt=$4
  local strategies=$5

  local out_dir="${RUN_ROOT}/eval_${exp_tag}_fast"
  mkdir -p "$out_dir"

  local pid_file="$out_dir/${dataset}.pid"
  local log_file="$out_dir/${dataset}.log"
  local csv_file="$out_dir/${dataset}.csv"

  if [[ -f "$pid_file" ]]; then
    local existing
    existing=$(cat "$pid_file")
    if kill -0 "$existing" 2>/dev/null; then
      echo "[$exp_tag/$dataset] already running PID $existing"
      return 0
    fi
  fi

  echo "[$exp_tag/$dataset] launching on GPU $gpu (strategies: $strategies)"
  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$REPO/toto2:${PYTHONPATH:-}" \
    nohup "$PY" -u -m toto2.eval.run_benchmark_fast \
      --checkpoint "$ckpt" \
      --datasets $dataset \
      --strategies $strategies \
      --heads linear_head \
      --n-seeds 3 \
      --device cuda \
      --output "$csv_file" \
      > "$log_file" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  disown
  echo "  PID=$pid log=$log_file csv=$csv_file"
}

case "${1:-help}" in
  exp47)
    [[ $# -ge 3 ]] || { echo "Usage: $0 exp47 <gpu> <dataset>"; exit 1; }
    launch_one exp47 "$2" "$3" "$CKPT_EXP47" "frozen ridge_probe lora"
    ;;
  exp48)
    [[ $# -ge 3 ]] || { echo "Usage: $0 exp48 <gpu> <dataset>"; exit 1; }
    launch_one exp48 "$2" "$3" "$CKPT_EXP48" "frozen ridge_probe lora"
    ;;
  exp36-lora)
    [[ $# -ge 3 ]] || { echo "Usage: $0 exp36-lora <gpu> <dataset>"; exit 1; }
    launch_one exp36-lora "$2" "$3" "$CKPT_EXP36" "lora"
    ;;
  status)
    for tag in exp47 exp48 exp36-lora; do
      out_dir="${RUN_ROOT}/eval_${tag}_fast"
      [[ -d "$out_dir" ]] || continue
      echo "=== $tag (fast) ==="
      for pid_file in "$out_dir"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        ds=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        csv="$out_dir/${ds}.csv"
        n_rows=0
        if [[ -f "$csv" ]]; then
          n_rows=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
        fi
        if kill -0 "$pid" 2>/dev/null; then
          etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
          echo "  [$ds] PID $pid running ($etime, $n_rows rows so far)"
        else
          echo "  [$ds] PID $pid done ($n_rows rows in CSV)"
        fi
      done
    done
    ;;
  help|*)
    cat <<HELP
Usage: $0 {exp47|exp48|exp36-lora|status} [<gpu> <dataset>]

Launches a single (checkpoint, dataset) job using the OPTIMIZED
run_benchmark_fast.py.

Datasets: arithmetic_zyma2019 bcic2a bcic2020_3 chbmit faced
          isruc_sleep mdd_mumtaz2016 physionet
HELP
    ;;
esac
