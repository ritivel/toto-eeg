#!/bin/bash
# --------------------------------------------------------------------
# Overnight launcher: parallel 24h scaling A/B
#   * exp50_long  on GPUs 0-3 (CAR + MR-MPL @ 150k steps, ~23.4 h)
#   * exp48_long2 on GPUs 4-7 (pure MR-MPL @ 150k steps, ~23.4 h)
#
# Both runs are PURE SCALING of existing winners — no new ideas, just
# more compute on the most promising recipe (exp50, with CAR) plus a
# matched-step baseline (exp48, no CAR).  Side-by-side at step 150k
# gives a definitive answer to "does CAR help at scale?".
#
# Compute budget: 8 H100s * ~24h = ~192 GPU-hours total.
# Wall-clock: max(exp48_long2, exp50_long) ≈ 23.4h on 4xH100 each.
#
# Workflow
# --------
#   ./launch_overnight_exp48_exp50_long.sh launch    # start both runs
#   ./launch_overnight_exp48_exp50_long.sh status    # tail logs / show pids
#   ./launch_overnight_exp48_exp50_long.sh kill_all  # stop both runs
#
# Smoke gate: SKIPPED — both recipes are scaled-up versions of recipes
# whose smokes have already passed (exp48 baseline + exp50 short).
# Launching directly to full because:
#   1. exp48 architecture + pure MR-MPL has run cleanly for 60k steps
#      (val_loss=0.288 final, no instability).
#   2. exp50 short ran 14k steps cleanly with CAR enabled and
#      already showed val_pinball=0.189 at step 6731.
# Risk of late-training instability is low; stable LR is the same.
# --------------------------------------------------------------------

set -u

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/toto-eeg}"
PY="${PY:-/home/ubuntu/eegModel/.venv/bin/python}"
DATASET_BUILDER="${DATASET_BUILDER:-toto2.scripts.examples.eeg_builder:build_datasets}"
LOG_ROOT="${LOG_ROOT:-/opt/dlami/nvme/eeg/runs/launch_logs}"

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/toto2" || { echo "FATAL: $REPO_ROOT/toto2 not found"; exit 1; }

run_one() {
  local name=$1
  local config=$2
  local visible_devices=$3

  local log_file="$LOG_ROOT/${name}.log"
  local pid_file="$LOG_ROOT/${name}.pid"

  echo "==================================================================="
  echo "Launching $name"
  echo "  config:    $config"
  echo "  GPUs:      $visible_devices"
  echo "  log:       $log_file"
  echo "==================================================================="

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid=$(cat "$pid_file")
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "WARNING: $name already running with PID $existing_pid; refusing to relaunch."
      echo "         Kill the running job (kill $existing_pid) before relaunching."
      return 1
    fi
  fi

  CUDA_VISIBLE_DEVICES="$visible_devices" \
    PYTHONPATH="$REPO_ROOT/toto2:${PYTHONPATH:-}" \
    nohup "$PY" -u -m toto2.scripts.train_toto2 \
      --config "$config" \
      --dataset-builder "$DATASET_BUILDER" \
      > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  disown
  echo "  PID: $pid"
  echo "  Tail with: tail -f $log_file"
}

case "${1:-help}" in
  launch)
    run_one \
      "exp50_rest_car_long" \
      "toto2/scripts/configs/pretrain_eeg_exp50_rest_car_long.yaml" \
      "0,1,2,3"
    sleep 5
    run_one \
      "exp48_mr_mpl_long2" \
      "toto2/scripts/configs/pretrain_eeg_exp48_mr_mpl_long2.yaml" \
      "4,5,6,7"
    echo
    echo "Both runs launched.  ETA ~23.4 h.  Run \`$0 status\` to check progress."
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp50_rest_car_long.pid "$LOG_ROOT"/exp48_mr_mpl_long2.pid; do
      [[ -f "$pid_file" ]] || continue
      local_name=$(basename "$pid_file" .pid)
      local_pid=$(cat "$pid_file")
      log_file="$LOG_ROOT/${local_name}.log"
      if kill -0 "$local_pid" 2>/dev/null; then
        echo "  [$local_name] PID $local_pid running"
        if [[ -f "$log_file" ]]; then
          echo "    Last 3 log lines:"
          tail -n 3 "$log_file" | tr '\r' '\n' | tail -3 | sed 's/^/      /'
        fi
      else
        echo "  [$local_name] PID $local_pid NOT running (job ended)"
        if [[ -f "$log_file" ]]; then
          echo "    Last 3 log lines:"
          tail -n 3 "$log_file" | tr '\r' '\n' | tail -3 | sed 's/^/      /'
        fi
      fi
    done
    echo
    echo "=== GPU usage ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    ;;

  kill_all)
    echo "Killing exp50_rest_car_long and exp48_mr_mpl_long2..."
    for pid_file in "$LOG_ROOT"/exp50_rest_car_long.pid "$LOG_ROOT"/exp48_mr_mpl_long2.pid; do
      [[ -f "$pid_file" ]] || continue
      local_pid=$(cat "$pid_file")
      if kill -0 "$local_pid" 2>/dev/null; then
        pkill -KILL -P "$local_pid" 2>/dev/null
        kill -KILL "$local_pid" 2>/dev/null
        echo "  killed pid $local_pid"
      fi
      rm -f "$pid_file"
    done
    echo "Done."
    ;;

  help|*)
    cat <<HELP
Usage: $0 {launch|status|kill_all}

  launch    Launch both runs in parallel:
              * exp50_rest_car_long on GPUs 0-3 (CAR + MR-MPL, 150k steps)
              * exp48_mr_mpl_long2  on GPUs 4-7 (pure MR-MPL, 150k steps)
            ~23.4h wall-clock for each.

  status    Show running PIDs + tail logs + GPU utilisation.

  kill_all  Stop both runs cleanly (SIGKILL).
HELP
    ;;
esac
