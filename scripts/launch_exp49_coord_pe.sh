#!/bin/bash
# --------------------------------------------------------------------
# exp49 — Continuous coordinate patch embedding (universal-EEG #1).
# Launch script for the Mumbai p5.48xlarge (8xH100 SXM5 80GB).
#
# Universal-EEG synthesis #1 (highest-ROI, all-stream-endorsed): replace
# the opaque ``series_id`` with a continuous, geometry-aware patch
# embedding combining 4D random Fourier features (over patch-time +
# 3D electrode position) and 81-mode real spherical harmonics for
# l = 0..8 (zero-initialised SH head so the SH branch grows organically
# during training).  See:
#
#   * Notion exp49 page — full hypothesis / methodology / decision rule.
#   * Universal-EEG synthesis (Notion exp48 → exp49 transition).
#   * scripts/configs/pretrain_eeg_exp49_coord_pe.yaml — full config.
#   * toto2/toto2/model.py:CoordPE — the new module.
#
# Workflow
# --------
#   ./launch_exp49_coord_pe.sh smoke    # 500 steps, ~10 min, GPUs 0-3
#   ./launch_exp49_coord_pe.sh full     # 30000 steps, ~4.7 h, GPUs 0-3
#   ./launch_exp49_coord_pe.sh status   # tail logs and show pids
#
# Smoke gate (read in WandB before launching ``full``)
#   * train_mr_mpl strictly decreasing
#   * train_mr_mpl_cos_mean_mean past 0.3 within 500 steps
#   * train_mr_mpl_amp_ratio_mean in [0.3, 2.0]
#   * grad_norm finite, in [1e-4, 1e1]
#   * NO NaN / Inf anywhere
#   * First batch's keys include 'electrode_coords' (logged by
#     LightningModule on the first forward pass)
#
# If ANY of those fails, do NOT promote to ``full``.  Most likely
# remediation: drop ``coord_pe_sigma_B`` from 1.0 to 0.5 in the YAML
# (high-frequency aliasing of γ → asinh-saturated patch_proj).
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
      "exp49_coord_pe_smoke" \
      "toto2/scripts/configs/pretrain_eeg_exp49_coord_pe_smoke.yaml" \
      "0,1,2,3"
    echo
    echo "Smoke launched on GPUs 0-3. After ~10 min, check that:"
    echo "  - train_mr_mpl strictly decreasing"
    echo "  - train_mr_mpl_cos_mean_mean past 0.3 within 500 steps"
    echo "  - train_mr_mpl_amp_ratio_mean in [0.3, 2.0]"
    echo "  - grad_norm in [1e-4, 1e1]"
    echo "  - NO NaN / Inf anywhere"
    echo "Then run: $0 full"
    ;;

  full)
    run_one \
      "exp49_coord_pe_full" \
      "toto2/scripts/configs/pretrain_eeg_exp49_coord_pe.yaml" \
      "0,1,2,3"
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp49_coord_pe_*.pid; do
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

  smoke   Launch the 500-step pre-flight smoke on GPUs 0-3 (~10 min).
          Run this FIRST.  Verifies coord-PE plumbing is healthy and
          σ_B = 1.0 doesn't alias.

  full    Launch the full 30000-step exp49 on GPUs 0-3 (~4.7 h).
          Same architecture + loss as exp48; only the patch embedding
          is changed (series_ids → continuous coord-PE).  Decision
          rule: A/B vs exp48 at matched compute (see Notion exp49).

  status  Show running / completed launches and tail their logs.
HELP
    ;;
esac
