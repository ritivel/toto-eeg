#!/bin/bash
# --------------------------------------------------------------------
# Single-command status check for the overnight exp47 + exp48 runs.
# Run from your laptop with:  ssh eeg-mumbai bash -s < scripts/check_overnight_status.sh
# Or directly on the box.
# --------------------------------------------------------------------

LOG_ROOT="${LOG_ROOT:-/opt/dlami/nvme/eeg/runs/launch_logs}"
LL_ROOT="${LL_ROOT:-/opt/dlami/nvme/eeg/runs/lightning_logs}"
PY="${PY:-/home/ubuntu/eegModel/.venv/bin/python}"

echo "================================================================"
echo "  Toto-EEG overnight runs status @ $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================"
echo

echo "=== processes ==="
for pid_file in "$LOG_ROOT"/exp4{7,8}_*_full.pid; do
  [[ -f "$pid_file" ]] || continue
  name=$(basename "$pid_file" .pid)
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
    echo "  [$name] PID $pid running (elapsed: $etime)"
  else
    echo "  [$name] PID $pid NOT running (job ended)"
  fi
done
echo

echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo

for run in exp47_mr_mpl_triple exp48_mr_mpl_long; do
  echo "=== $run metrics ==="
  metrics=$(find "$LL_ROOT/toto2_eeg_${run}" -name 'metrics.csv' 2>/dev/null | sort | tail -1)
  if [[ -z "$metrics" ]]; then
    echo "  (no metrics file yet)"
    echo
    continue
  fi
  "$PY" -c "
import pandas as pd
df = pd.read_csv('$metrics')
print(f'  csv: {len(df)} rows, max train step: {int(df[\"step\"].max())}')
val_cols = [c for c in ['step','val_loss','val_pinball','val_mr_mpl','val_mr_mpl_cos_mean_mean','val_mr_mpl_amp_ratio_mean','val_trunk_eff_rank_ratio'] if c in df.columns]
val = df[val_cols].dropna(how='all', subset=val_cols[1:]).reset_index(drop=True)
if len(val) == 0:
    print('  (no val checkpoints yet)')
else:
    with pd.option_context('display.float_format', lambda x: f'{x:.4g}', 'display.width', 220):
        print(val.to_string(index=False).replace('\n', '\n  '))
last_loss = df['train_mr_mpl_step'].dropna()
if len(last_loss) > 0:
    print(f'  latest train_mr_mpl_step: {last_loss.iloc[-1]:.4g}')
"
  echo
done

echo "=== log tails (last 2 lines each) ==="
for log in "$LOG_ROOT"/exp4{7,8}_*_full.log; do
  [[ -f "$log" ]] || continue
  echo "--- $(basename "$log") ---"
  tail -2 "$log" | tr '\r' '\n' | tail -2
done
