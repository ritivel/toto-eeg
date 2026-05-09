#!/bin/bash
# --------------------------------------------------------------------
# Overnight 12h scaling launcher for two parallel 4xH100 runs:
#   * exp51_chain_long  (CAR + DPSS + MR-MPL @ 80k) on GPUs 0-3
#   * exp50_long_seed55 (CAR + MR-MPL @ 80k, seed=55) on GPUs 4-7
#
# Pure scaling experiments, no new ideas.  exp51_chain stacks
# universal-EEG #3 (DPSS multitaper scaler) on top of yesterday's
# winner #2 (CAR).  exp50_long_seed55 is a reproducibility check
# on the 150k SOTA at a different seed.
#
# Both runs use the same training schedule (warmup=1500 / stable=74000
# / decay=4500 / max=80000) so a step-by-step A/B is fair.
#
# Workflow
# --------
#   ./launch_overnight_exp51_chain_exp50_seed55.sh launch
#   ./launch_overnight_exp51_chain_exp50_seed55.sh status
#   ./launch_overnight_exp51_chain_exp50_seed55.sh kill_all
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
}

case "${1:-help}" in
  launch)
    run_one \
      "exp51_chain_long" \
      "toto2/scripts/configs/pretrain_eeg_exp51_chain_long.yaml" \
      "0,1,2,3"
    sleep 2
    run_one \
      "exp50_long_seed55" \
      "toto2/scripts/configs/pretrain_eeg_exp50_long_seed55.yaml" \
      "4,5,6,7"
    echo
    echo "Both runs launched.  Check progress with: $0 status"
    ;;

  status)
    echo "=== Active runs ==="
    for pid_file in "$LOG_ROOT"/exp{51_chain_long,50_long_seed55}.pid; do
      [[ -f "$pid_file" ]] || continue
      n=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      log="$LOG_ROOT/${n}.log"
      if kill -0 "$pid" 2>/dev/null; then
        echo "  [$n] PID $pid running"
      else
        echo "  [$n] PID $pid NOT running"
      fi
      [[ -f "$log" ]] && echo "    last log line:" && tail -n 1 "$log" | tr '\r' '\n' | tail -1 | sed 's/^/      /'
    done
    echo
    echo "=== GPU usage ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    ;;

  kill_all)
    for pid_file in "$LOG_ROOT"/exp{51_chain_long,50_long_seed55}.pid; do
      [[ -f "$pid_file" ]] || continue
      n=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      if kill -0 "$pid" 2>/dev/null; then
        echo "killing [$n] pid $pid"
        pkill -KILL -P "$pid" 2>/dev/null
        kill -KILL "$pid" 2>/dev/null
      fi
      rm -f "$pid_file"
    done
    sleep 3
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    ;;

  help|*)
    cat <<HELP
Usage: $0 {launch|status|kill_all}

  launch   Start both 80k-step runs in parallel:
             * exp51_chain_long  on GPUs 0-3 (CAR + DPSS + MR-MPL)
             * exp50_long_seed55 on GPUs 4-7 (CAR + MR-MPL, seed=55)
           Each run is ~12h.  Logs in $LOG_ROOT/.

  status   Show running PIDs + tail logs + GPU usage.

  kill_all Kill both runs (SIGKILL).
HELP
    ;;
esac
