#!/bin/bash
# --------------------------------------------------------------------
# exp43 AMSE on the normal baseline -- launch script for the Mumbai
# p5.48xlarge (8xH100 SXM5 80GB).
#
# Tests user-stated hypothesis:
#   "Is the trunk collapse caused by pinball's amplitude/phase
#    entanglement?  AMSE (Subich et al. ICML 2025, arXiv:2501.19374)
#    decomposes spectral error into amplitude + phase."
#
# This script runs a short SMOKE (~10 min, 500 steps) FIRST and only
# launches the full 5500-step run if the smoke passes the
# pre-registered checks (loss decreases, no NaNs, amplitude+phase
# terms both moving).
#
# WORKFLOW
# --------
#   ./launch_exp43_amse.sh smoke   # 500 steps, ~10 min, GPUs 0-3
#   ./launch_exp43_amse.sh full    # 5500 steps, ~75 min, GPUs 0-3
#   ./launch_exp43_amse.sh triple  # 30000 steps + JEPA+AAMP+PARS,
#                                  # ~10 hr, GPUs 4-7 (only after the
#                                  # baseline smoke passes)
#   ./launch_exp43_amse.sh status  # tail logs and show pids
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
  local extra_tag=$4

  local log_file="$LOG_ROOT/${name}.log"
  local pid_file="$LOG_ROOT/${name}.pid"

  echo "==================================================================="
  echo "Launching $name"
  echo "  config:        $config"
  echo "  GPUs:          $visible_devices"
  echo "  log:           $log_file"
  echo "  WandB tag:     $extra_tag"
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
  smoke)
    run_one \
      "exp43_amse_baseline_smoke" \
      "toto2/scripts/configs/pretrain_eeg_exp43_amse_baseline_smoke.yaml" \
      "0,1,2,3" \
      "smoke"
    echo
    echo "Smoke launched.  After ~10 min, check the logs for:"
    echo "  - train_amse decreasing"
    echo "  - train_amse_amp_term + train_amse_coh_term both > 0"
    echo "  - train_amse_amp_ratio in [0.3, 2.0]"
    echo "  - no NaN / Inf"
    echo "Then run: $0 full"
    ;;

  full)
    run_one \
      "exp43_amse_baseline_full" \
      "toto2/scripts/configs/pretrain_eeg_exp43_amse_baseline.yaml" \
      "0,1,2,3" \
      "long_run"
    ;;

  triple)
    run_one \
      "exp44_amse_triple" \
      "toto2/scripts/configs/pretrain_eeg_exp44_amse_triple.yaml" \
      "4,5,6,7" \
      "triple_long_run"
    echo
    echo "exp44 launched on GPUs 4-7.  Runs in parallel with exp43_amse_baseline_full"
    echo "if that's still running on GPUs 0-3."
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/*.pid; do
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
    ;;

  help|*)
    cat <<HELP
Usage: $0 {smoke|full|triple|status}

  smoke   Launch the 500-step pre-flight smoke on GPUs 0-3 (~10 min).
          Run this FIRST.  Verify the WandB run shows:
            * train_amse strictly decreasing
            * train_amse_amp_term + train_amse_coh_term both > 0
            * train_amse_amp_ratio in [0.3, 2.0]
            * no NaN / Inf in any logged value

  full    Launch the full 5500-step exp43 baseline on GPUs 0-3 (~75 min).
          Same architecture as exp26 Probe F (the no-aux baseline), only
          the loss is changed from pinball to AMSE.

  triple  Launch the 30000-step exp44 (AMSE + JEPA+AAMP+PARS) on GPUs 4-7
          (~10 hours).  Only run AFTER the baseline smoke has passed.

  status  Show running / completed launches and tail their logs.
HELP
    ;;
esac
