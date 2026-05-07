#!/bin/bash
# Re-launch slow datasets with ridge_probe only (closed-form, ~3 min each)
# Plus full_finetune which is still useful but expensive

set -u

OUT=/opt/dlami/nvme/eeg/runs/eval_exp36_full
mkdir -p "$OUT"
CKPT="/opt/dlami/nvme/eeg/runs/checkpoints/toto2_eeg_exp36_triple_aux/epoch=6-step=30000-val_loss=0.1384-train_jepa_loss_step=7.435e-03.ckpt"
PY=/home/ubuntu/eegModel/.venv/bin/python
cd /home/ubuntu/toto-eeg/toto2 || exit 1

# Run ridge_probe only on the slow datasets we killed
launch_ridge() {
  local gpu=$1
  local ds=$2
  echo "Launching $ds RIDGE on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u -m toto2.eval.run_benchmark \
    --checkpoint "$CKPT" \
    --datasets $ds \
    --strategies ridge_probe \
    --heads linear_head \
    --n-seeds 3 \
    --device cuda \
    --output "$OUT/${ds}_ridge.csv" \
    > "$OUT/${ds}_ridge.log" 2>&1 &
  local pid=$!
  echo $pid > "$OUT/${ds}_ridge.pid"
  disown
  echo "  ${ds}_ridge PID=$pid"
}

# Free GPUs (3, 4, 5 were the slow ones we killed)
launch_ridge 3 chbmit
launch_ridge 4 faced
launch_ridge 5 isruc_sleep

sleep 3
echo
echo "=== Ridge-only relaunch ==="
for f in "$OUT"/*_ridge.pid; do
  ds=$(basename "$f" .pid)
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    echo "$ds: PID $pid running"
  else
    echo "$ds: PID $pid NOT running"
  fi
done
