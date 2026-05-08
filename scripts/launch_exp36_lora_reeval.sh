#!/bin/bash
# --------------------------------------------------------------------
# exp36 LoRA re-evaluation with the PEFT-fix in place (commit 7693767).
#
# The original exp36 eval (May 7) crashed on LoRA for 5 of 8 datasets
# with `ValueError: Target module DepthModuleList(...)` because PEFT 0.x
# rejected `uu.Linear` (Datadog's u-muP scaled linear).  Fix landed
# AFTER exp36 was eval'd, so its LoRA arm is empty / NaN in the CSVs.
#
# This script re-runs LoRA only (frozen + full + ridge already in CSV)
# so we can compare exp47 / exp48 LoRA results to a real exp36 LoRA
# baseline, not "no data".  Output goes to a fresh
# eval_exp36_lora_only/ directory; a separate merge step combines them
# with the existing eval_exp36_full/ CSVs for the final comparison.
#
# Usage:
#   ./launch_exp36_lora_reeval.sh group_a   # GPUs 0-3, 4 lighter datasets
#   ./launch_exp36_lora_reeval.sh group_b   # GPUs 4-7, 4 heavier datasets
#   ./launch_exp36_lora_reeval.sh all       # all 8 GPUs in parallel
#   ./launch_exp36_lora_reeval.sh status    # tail logs + show pids
# --------------------------------------------------------------------

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
CKPT=/opt/dlami/nvme/eeg/runs/checkpoints/toto2_eeg_exp36_triple_aux/epoch=6-step=30000-val_loss=0.1384-train_jepa_loss_step=7.435e-03.ckpt
OUT=/opt/dlami/nvme/eeg/runs/eval_exp36_lora_only

# Same dataset split as the exp47/exp48 launcher
GROUP_A=(arithmetic_zyma2019 bcic2a bcic2020_3 mdd_mumtaz2016)
GROUP_B=(physionet chbmit faced isruc_sleep)

mkdir -p "$OUT"
cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }

[[ -f "$CKPT" ]] || { echo "FATAL: exp36 checkpoint not found at $CKPT"; exit 1; }

run_one() {
  local gpu=$1
  local ds=$2
  local pid_file="$OUT/${ds}.pid"
  local log_file="$OUT/${ds}.log"
  local csv_file="$OUT/${ds}.csv"

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid=$(cat "$pid_file")
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "  [$ds] already running PID $existing_pid; skipping"
      return 0
    fi
  fi

  echo "  [$ds] launching on GPU $gpu (LoRA only, 3 seeds)"
  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$REPO/toto2:${PYTHONPATH:-}" \
    nohup "$PY" -u -m toto2.eval.run_benchmark \
      --checkpoint "$CKPT" \
      --datasets $ds \
      --strategies lora \
      --heads linear_head \
      --n-seeds 3 \
      --device cuda \
      --output "$csv_file" \
      > "$log_file" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  disown
  echo "    PID=$pid"
}

case "${1:-help}" in
  group_a)
    echo "==================================================================="
    echo "exp36 LoRA re-eval — group A on GPUs 0-3"
    echo "==================================================================="
    for i in "${!GROUP_A[@]}"; do
      run_one "$i" "${GROUP_A[$i]}"
    done
    ;;

  group_b)
    echo "==================================================================="
    echo "exp36 LoRA re-eval — group B on GPUs 4-7"
    echo "==================================================================="
    for i in "${!GROUP_B[@]}"; do
      run_one "$((i+4))" "${GROUP_B[$i]}"
    done
    ;;

  all)
    echo "==================================================================="
    echo "exp36 LoRA re-eval — ALL 8 datasets on all 8 GPUs"
    echo "==================================================================="
    for i in "${!GROUP_A[@]}"; do
      run_one "$i" "${GROUP_A[$i]}"
    done
    for i in "${!GROUP_B[@]}"; do
      run_one "$((i+4))" "${GROUP_B[$i]}"
    done
    ;;

  status)
    echo "=== exp36 LoRA re-eval status ==="
    for pid_file in "$OUT"/*.pid; do
      [[ -f "$pid_file" ]] || continue
      ds=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      csv_file="$OUT/${ds}.csv"
      if kill -0 "$pid" 2>/dev/null; then
        etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        echo "  [$ds] PID $pid running ($etime)"
      else
        if [[ -f "$csv_file" ]]; then
          n=$(tail -n +2 "$csv_file" | wc -l | tr -d ' ')
          echo "  [$ds] DONE ($n rows)"
        else
          echo "  [$ds] PID $pid NOT running, NO csv (likely failed)"
        fi
      fi
    done
    ;;

  help|*)
    cat <<HELP
Usage: $0 {group_a|group_b|all|status}

  group_a   GPUs 0-3, 4 lighter datasets:  ${GROUP_A[*]}
  group_b   GPUs 4-7, 4 heavier datasets:  ${GROUP_B[*]}
  all       All 8 GPUs, all 8 datasets in parallel
  status    Show running PIDs + DONE markers

Strategy: LoRA only (other strategies already in eval_exp36_full/)
Head: linear_head, Seeds: 3
Output: $OUT/<dataset>.csv
HELP
    ;;
esac
