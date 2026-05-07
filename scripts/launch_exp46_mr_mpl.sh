#!/bin/bash
# --------------------------------------------------------------------
# exp46 MR-MPL on the normal baseline -- launch script for the Mumbai
# p5.48xlarge (8xH100 SXM5 80GB).
#
# Tests the hypothesis from the exp43 (AMSE) post-mortem:
#   "AMSE got the spectrum right (amp_ratio=0.84) but the phase wrong
#    (coh_mean=0.19) because its phase coherence term is whole-sequence.
#    A multi-resolution STFT loss with magnitude-weighted phase coherence
#    (the canonical waveform-synthesis loss from HiFi-GAN/MelGAN/BigVGAN)
#    should fix the phase failure mode by giving per-(time, freq)-bin
#    phase signal at multiple time-frequency scales."
#
# Workflow
# --------
#   ./launch_exp46_mr_mpl.sh smoke   # 500 steps, ~10 min, GPUs 0-3
#   ./launch_exp46_mr_mpl.sh full    # 5500 steps, ~75 min, GPUs 0-3
#   ./launch_exp46_mr_mpl.sh status  # tail logs and show pids
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
      "exp46_mr_mpl_baseline_smoke" \
      "toto2/scripts/configs/pretrain_eeg_exp46_mr_mpl_baseline_smoke.yaml" \
      "0,1,2,3" \
      "smoke"
    echo
    echo "Smoke launched.  After ~10 min, check the logs for:"
    echo "  - train_mr_mpl decreasing"
    echo "  - train_mr_mpl_cos_mean_mean GROWING past 0.3 (THE headline metric)"
    echo "  - amp_ratio_mean in [0.3, 2.0]"
    echo "  - all sub-losses (lm, sc, pc) decreasing"
    echo "  - no NaN / Inf"
    echo "Then run: $0 full"
    ;;

  full)
    run_one \
      "exp46_mr_mpl_baseline_full" \
      "toto2/scripts/configs/pretrain_eeg_exp46_mr_mpl_baseline.yaml" \
      "0,1,2,3" \
      "long_run"
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp46_mr_mpl_*.pid; do
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
Usage: $0 {smoke|full|status}

  smoke   Launch the 500-step pre-flight smoke on GPUs 0-3 (~10 min).
          Run this FIRST.  Verify the WandB run shows:
            * train_mr_mpl strictly decreasing
            * train_mr_mpl_cos_mean_mean GROWING past 0.3 (key metric;
              if this stalls below 0.2 like AMSE, abort)
            * amp_ratio_mean in [0.3, 2.0]
            * all sub-losses (lm, sc, pc) decreasing
            * no NaN / Inf in any logged value

  full    Launch the full 5500-step exp46 baseline on GPUs 0-3 (~75 min).
          Same architecture as exp26 Probe F (no auxiliaries), only the
          loss is changed from pinball to MR-MPL.

  status  Show running / completed launches and tail their logs.
HELP
    ;;
esac
