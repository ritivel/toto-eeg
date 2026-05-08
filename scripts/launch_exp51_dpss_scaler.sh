#!/bin/bash
# --------------------------------------------------------------------
# exp51 -- DPSS-tapered (Slepian multitaper) causal scaler
# (universal-EEG #3).  Launch script for the Mumbai p5.48xlarge
# (8xH100 SXM5 80GB).
#
# Universal-EEG synthesis #3: replace the Welford cumulative variance
# with a per-patch Thomson multitaper variance using K=3 leading DPSS
# tapers of length patch_size and time-bandwidth NW=2.5.  DPSS sequences
# are uniquely maximally concentrated in time and frequency under the
# Heisenberg-Donoho-Stark uncertainty bound (Slepian 1978); multitaper
# PSD is minimum-variance among all linear estimators (Thomson 1982).
#
# Headline win: dramatic reduction in scaler bias on bursty signals.
# A single 50sigma edge spike inflates Welford sigma ~7x; DPSS u_0
# is bell-shaped and down-weights the edges, giving a much less
# inflated estimate.
#
# Workflow
# --------
#   ./launch_exp51_dpss_scaler.sh smoke   # 500 steps, ~6 min, GPUs 0-3
#   ./launch_exp51_dpss_scaler.sh full    # 30000 steps, ~4.4 h, GPUs 0-3
#   ./launch_exp51_dpss_scaler.sh status  # tail logs and show pids
#
# Smoke gate (read in WandB before launching ``full``)
#   * train_mr_mpl strictly decreasing
#   * train_mr_mpl_cos_mean past 0.3 within 500 steps
#   * train_mr_mpl_amp_ratio in [0.3, 2.0]
#   * grad_norm finite, in [1e-4, 1e1]
#   * NO NaN / Inf anywhere
#   * val_mr_mpl at step 500 within ~10% of exp46's smoke (~0.6) --
#     a much larger delta would suggest the bias correction is mis-
#     applied (input distribution shifted, model visibly behind).
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
  smoke)
    run_one \
      "exp51_dpss_scaler_smoke" \
      "toto2/scripts/configs/pretrain_eeg_exp51_dpss_scaler_smoke.yaml" \
      "0,1,2,3"
    echo
    echo "Smoke launched on GPUs 0-3."
    echo "After ~6 min, check that:"
    echo "  - train_mr_mpl strictly decreasing"
    echo "  - train_mr_mpl_cos_mean past 0.3 within 500 steps"
    echo "  - val_mr_mpl @ step 500 within ~10% of exp46's smoke (~0.6)"
    echo "  - grad_norm in [1e-4, 1e1]"
    echo "  - NO NaN / Inf anywhere"
    echo "Then run: $0 full"
    ;;

  full)
    run_one \
      "exp51_dpss_scaler_full" \
      "toto2/scripts/configs/pretrain_eeg_exp51_dpss_scaler.yaml" \
      "0,1,2,3"
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp51_dpss_scaler_*.pid; do
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
Usage: $0 {smoke|full|status}

  smoke   Launch the 500-step pre-flight smoke on GPUs 0-3 (~6 min).
          Verifies that the DPSS taper buffers materialise on GPU,
          per-patch variance estimator is finite on real EEG, and
          the bias correction is being applied (input distribution
          stays close to the Welford baseline).  Run this FIRST.

  full    Launch the full 30000-step exp51 on GPUs 0-3 (~4.4 h).
          Same architecture + loss as exp48; only the layer-0 scaler
          changes (Welford -> Thomson multitaper / Slepian DPSS).
          Decision rule: A/B vs exp48 at matched compute (see
          Notion exp51).

  status  Show running / completed launches and tail their logs.
HELP
    ;;
esac
