#!/bin/bash
# Launch comprehensive eval of exp36 TRIPLE checkpoint across 8 datasets
# x 4 strategies (ridge_probe, frozen, lora, full_finetune) on 8 GPUs.
# One dataset per GPU; each runs all 4 strategies sequentially within
# the same process.

set -u

OUT=/opt/dlami/nvme/eeg/runs/eval_exp36_full
mkdir -p "$OUT"
CKPT="/opt/dlami/nvme/eeg/runs/checkpoints/toto2_eeg_exp36_triple_aux/epoch=6-step=30000-val_loss=0.1384-train_jepa_loss_step=7.435e-03.ckpt"
PY=/home/ubuntu/eegModel/.venv/bin/python
cd /home/ubuntu/toto-eeg/toto2 || exit 1

# Strategies to run for each dataset
STRATS="frozen ridge_probe lora full_finetune"

# Map: GPU → dataset
launch() {
  local gpu=$1
  local ds=$2
  echo "Launching $ds on GPU $gpu (strategies: $STRATS)"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u -m toto2.eval.run_benchmark \
    --checkpoint "$CKPT" \
    --datasets $ds \
    --strategies $STRATS \
    --heads linear_head \
    --n-seeds 3 \
    --device cuda \
    --output "$OUT/$ds.csv" \
    > "$OUT/$ds.log" 2>&1 &
  local pid=$!
  echo $pid > "$OUT/$ds.pid"
  disown
  echo "  $ds PID=$pid"
}

# 8 datasets across 8 GPUs (one per GPU)
launch 0 arithmetic_zyma2019
launch 1 bcic2a
launch 2 bcic2020_3
launch 3 chbmit
launch 4 faced
launch 5 isruc_sleep
launch 6 mdd_mumtaz2016
launch 7 physionet

sleep 3
echo
echo "=== Launched ==="
for f in "$OUT"/*.pid; do
  ds=$(basename "$f" .pid)
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    echo "$ds: PID $pid running"
  else
    echo "$ds: PID $pid NOT running"
  fi
done
