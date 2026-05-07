#!/bin/bash
# --------------------------------------------------------------------
# Overnight launcher for exp47 (MR-MPL + triple aux) and exp48 (MR-MPL
# pure long) to fully utilise all 8 H100s on the Mumbai p5.48xlarge.
#
# Two parallel experiments:
#   * exp47_mr_mpl_triple   on GPUs 0-3  (30000 steps, ~10h, has smoke)
#   * exp48_mr_mpl_long     on GPUs 4-7  (60000 steps, ~9.4h)
#
# Both fit into a 10-hour overnight window.  exp47 matches exp36's
# 30000-step production schedule EXACTLY for a head-to-head A/B with
# the previous best baseline (exp36 val_loss=0.1384).  exp48 takes
# pure MR-MPL deeper than any prior run (11x exp46's 5500 steps) to
# find the actual ceiling without auxiliary supervision.
#
# Workflow (recommended overnight order):
#   1. ./launch_overnight_exp47_exp48.sh exp48_full
#      Launch exp48 (low-risk; same recipe as the just-finished exp46).
#      Starts immediately and uses GPUs 4-7.
#
#   2. ./launch_overnight_exp47_exp48.sh exp47_smoke
#      Launch the 1000-step smoke for exp47 on GPUs 0-3 to validate
#      the MR-MPL + AAMP+JEPA+PARS combination (~15 min wall).
#
#   3. After smoke passes (cos_mean > 0.5, no NaN, all aux losses
#      finite), promote to:
#      ./launch_overnight_exp47_exp48.sh exp47_full
#      Launches the full 15000-step run on GPUs 0-3.
#
#   4. ./launch_overnight_exp47_exp48.sh status
#      Show running PIDs + tail logs for both runs.
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
  exp47_smoke)
    run_one \
      "exp47_mr_mpl_triple_smoke" \
      "toto2/scripts/configs/pretrain_eeg_exp47_mr_mpl_triple_smoke.yaml" \
      "0,1,2,3"
    echo
    echo "Smoke launched on GPUs 0-3.  After ~15 min, check that:"
    echo "  - train_mr_mpl decreasing"
    echo "  - train_mr_mpl_cos_mean_mean GROWING past 0.5"
    echo "  - train_jepa_loss / train_aamp_loss / train_pars_loss all decreasing"
    echo "  - NO NaN / Inf in any logged value"
    echo "  - amp_ratio_mean in [0.3, 2.0]"
    echo "Then run: $0 exp47_full"
    ;;

  exp47_full)
    run_one \
      "exp47_mr_mpl_triple_full" \
      "toto2/scripts/configs/pretrain_eeg_exp47_mr_mpl_triple.yaml" \
      "0,1,2,3"
    ;;

  exp48_full)
    run_one \
      "exp48_mr_mpl_long_full" \
      "toto2/scripts/configs/pretrain_eeg_exp48_mr_mpl_long.yaml" \
      "4,5,6,7"
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp4{7,8}_*.pid; do
      [[ -f "$pid_file" ]] || continue
      local_name=$(basename "$pid_file" .pid)
      local_pid=$(cat "$pid_file")
      log_file="$LOG_ROOT/${local_name}.log"
      if kill -0 "$local_pid" 2>/dev/null; then
        echo "  [$local_name] PID $local_pid running"
        if [[ -f "$log_file" ]]; then
          echo "    Last 3 log lines:"
          tail -n 3 "$log_file" | sed 's/^/      /'
        fi
      else
        echo "  [$local_name] PID $local_pid NOT running (job ended)"
        if [[ -f "$log_file" ]]; then
          echo "    Last 3 log lines:"
          tail -n 3 "$log_file" | sed 's/^/      /'
        fi
      fi
    done
    echo
    echo "=== GPU usage ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    ;;

  help|*)
    cat <<HELP
Usage: $0 {exp47_smoke|exp47_full|exp48_full|status}

  exp47_smoke  Launch 1000-step smoke for MR-MPL + triple aux on GPUs
               0-3 (~15 min).  Run this BEFORE exp47_full.

  exp47_full   Launch the full 30000-step exp47 (MR-MPL + JEPA + AAMP +
               PARS) on GPUs 0-3 (~10h wall clock).  Matches exp36's
               step count for a clean head-to-head vs val_loss=0.1384.

  exp48_full   Launch the full 60000-step exp48 (MR-MPL pure, longer)
               on GPUs 4-7 (~9.4h wall clock).  Low risk -- can launch
               immediately without smoke.

  status       Show running PIDs + tail recent log lines + GPU usage.

Recommended overnight order:
  1. $0 exp48_full       # immediately, low risk, GPUs 4-7
  2. $0 exp47_smoke      # validate aux+MR-MPL, GPUs 0-3, ~15 min
  3. $0 exp47_full       # after smoke passes, GPUs 0-3, ~5h
HELP
    ;;
esac
