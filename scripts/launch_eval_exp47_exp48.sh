#!/bin/bash
# --------------------------------------------------------------------
# Downstream eval launcher for exp47 (MR-MPL + triple aux) and exp48
# (pure MR-MPL long).  Mirrors the exp36 eval methodology EXACTLY:
#   * 8 datasets (the 8 used in exp36 / launch_full_eval.sh)
#   * 4 strategies: frozen, ridge_probe, lora, full_finetune
#   * 1 head: linear_head
#   * 3 seeds
# --------------------------------------------------------------------

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
CKPT_ROOT=/opt/dlami/nvme/eeg/runs/checkpoints

CKPT_EXP47="${CKPT_ROOT}/toto2_eeg_exp47_mr_mpl_triple/epoch=6-step=30000-val_loss=0.3005-train_mr_mpl_step=4.761e-01.ckpt"
CKPT_EXP48_BEST=""  # filled in dynamically when exp48 completes

# Datasets and which 4 to launch on each GPU half
GROUP_A=(arithmetic_zyma2019 bcic2a bcic2020_3 mdd_mumtaz2016)  # GPUs 0-3 (lighter / smaller)
GROUP_B=(physionet chbmit faced isruc_sleep)                    # GPUs 4-7 (heavier)

STRATS="frozen ridge_probe lora"   # full_finetune dropped per user (May 8); too slow vs the signal it gives
HEAD="linear_head"
N_SEEDS=3

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }

run_one() {
  local exp=$1
  local ckpt=$2
  local gpu=$3
  local ds=$4

  local out_dir=/opt/dlami/nvme/eeg/runs/eval_${exp}_full
  mkdir -p "$out_dir"

  local pid_file="$out_dir/${ds}.pid"
  local log_file="$out_dir/${ds}.log"
  local csv_file="$out_dir/${ds}.csv"

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid=$(cat "$pid_file")
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "  [$exp/$ds] already running PID $existing_pid; skipping"
      return 0
    fi
  fi

  echo "  [$exp/$ds] launching on GPU $gpu (PID file: $pid_file)"
  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$REPO/toto2:${PYTHONPATH:-}" \
    nohup "$PY" -u -m toto2.eval.run_benchmark \
      --checkpoint "$ckpt" \
      --datasets $ds \
      --strategies $STRATS \
      --heads $HEAD \
      --n-seeds $N_SEEDS \
      --device cuda \
      --output "$csv_file" \
      > "$log_file" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  disown
  echo "    PID=$pid log=$log_file"
}

case "${1:-help}" in
  exp47_group_a)
    echo "==================================================================="
    echo "Launching exp47 eval — GROUP A (GPUs 0-3, 4 lighter datasets)"
    echo "  checkpoint: $CKPT_EXP47"
    echo "==================================================================="
    [[ -f "$CKPT_EXP47" ]] || { echo "FATAL: checkpoint not found"; exit 1; }
    for i in "${!GROUP_A[@]}"; do
      run_one exp47 "$CKPT_EXP47" "$i" "${GROUP_A[$i]}"
    done
    ;;

  exp47_group_b)
    echo "==================================================================="
    echo "Launching exp47 eval — GROUP B (GPUs 4-7, 4 heavier datasets)"
    echo "  checkpoint: $CKPT_EXP47"
    echo "==================================================================="
    [[ -f "$CKPT_EXP47" ]] || { echo "FATAL: checkpoint not found"; exit 1; }
    for i in "${!GROUP_B[@]}"; do
      run_one exp47 "$CKPT_EXP47" "$((i+4))" "${GROUP_B[$i]}"
    done
    ;;

  exp48_all)
    # Find the best exp48 checkpoint by val_loss
    CKPT_EXP48_BEST=$(ls -t "$CKPT_ROOT/toto2_eeg_exp48_mr_mpl_long/"*.ckpt 2>/dev/null \
      | grep -v last \
      | sort -t= -k4 -n \
      | head -1)
    [[ -f "$CKPT_EXP48_BEST" ]] || { echo "FATAL: no exp48 checkpoint found"; exit 1; }
    echo "==================================================================="
    echo "Launching exp48 eval — ALL 8 datasets on all 8 GPUs"
    echo "  checkpoint: $CKPT_EXP48_BEST"
    echo "==================================================================="
    for i in "${!GROUP_A[@]}"; do
      run_one exp48 "$CKPT_EXP48_BEST" "$i" "${GROUP_A[$i]}"
    done
    for i in "${!GROUP_B[@]}"; do
      run_one exp48 "$CKPT_EXP48_BEST" "$((i+4))" "${GROUP_B[$i]}"
    done
    ;;

  exp48_group_a_on_4567)
    # Same as exp48_all but ONLY group A datasets, mapped to GPUs 4-7.
    # Used when exp47 group A is still occupying GPUs 0-3.
    CKPT_EXP48_BEST=$(ls -t "$CKPT_ROOT/toto2_eeg_exp48_mr_mpl_long/"*.ckpt 2>/dev/null \
      | grep -v last \
      | sort -t= -k4 -n \
      | head -1)
    [[ -f "$CKPT_EXP48_BEST" ]] || { echo "FATAL: no exp48 checkpoint found"; exit 1; }
    echo "==================================================================="
    echo "Launching exp48 eval — group A on GPUs 4-7"
    echo "  checkpoint: $CKPT_EXP48_BEST"
    echo "==================================================================="
    for i in "${!GROUP_A[@]}"; do
      run_one exp48 "$CKPT_EXP48_BEST" "$((i+4))" "${GROUP_A[$i]}"
    done
    ;;

  exp48_group_b_on_0123)
    # exp48 group B datasets on GPUs 0-3.
    CKPT_EXP48_BEST=$(ls -t "$CKPT_ROOT/toto2_eeg_exp48_mr_mpl_long/"*.ckpt 2>/dev/null \
      | grep -v last \
      | sort -t= -k4 -n \
      | head -1)
    [[ -f "$CKPT_EXP48_BEST" ]] || { echo "FATAL: no exp48 checkpoint found"; exit 1; }
    echo "==================================================================="
    echo "Launching exp48 eval — group B on GPUs 0-3"
    echo "  checkpoint: $CKPT_EXP48_BEST"
    echo "==================================================================="
    for i in "${!GROUP_B[@]}"; do
      run_one exp48 "$CKPT_EXP48_BEST" "$i" "${GROUP_B[$i]}"
    done
    ;;

  status)
    for exp in exp47 exp48; do
      out_dir=/opt/dlami/nvme/eeg/runs/eval_${exp}_full
      [[ -d "$out_dir" ]] || continue
      echo "=== $exp eval status ==="
      for pid_file in "$out_dir"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        ds=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        log_file="$out_dir/${ds}.log"
        csv_file="$out_dir/${ds}.csv"
        if kill -0 "$pid" 2>/dev/null; then
          etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
          echo "  [$ds] PID $pid running (elapsed $etime)"
        else
          if [[ -f "$csv_file" ]]; then
            n_results=$(tail -n +2 "$csv_file" | wc -l | tr -d ' ')
            echo "  [$ds] DONE ($n_results result rows)"
          else
            echo "  [$ds] PID $pid NOT running, NO csv (likely failed)"
            echo "    last 3 log lines:"
            tail -3 "$log_file" 2>/dev/null | sed 's/^/      /'
          fi
        fi
      done
      echo
    done
    ;;

  help|*)
    cat <<HELP
Usage: $0 {exp47_group_a|exp47_group_b|exp48_all|status}

  exp47_group_a   Launch exp47 eval on GPUs 0-3 with 4 lighter datasets
                  (arithmetic_zyma2019, bcic2a, bcic2020_3, mdd_mumtaz2016)

  exp47_group_b   Launch exp47 eval on GPUs 4-7 with 4 heavier datasets
                  (physionet, chbmit, faced, isruc_sleep)

  exp48_all       Launch exp48 eval on ALL 8 GPUs.  Picks the best
                  (lowest val_loss) checkpoint automatically.

  status          Show running eval PIDs + results for both experiments.

Datasets:
  GROUP_A: ${GROUP_A[*]}
  GROUP_B: ${GROUP_B[*]}

Strategies (ALL 4): $STRATS
Head: $HEAD,  Seeds: $N_SEEDS
HELP
    ;;
esac
