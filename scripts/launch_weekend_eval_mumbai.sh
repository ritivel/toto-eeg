#!/bin/bash
# Eval launcher for weekend-trained production models on Mumbai p5.48xlarge.
# Each GPU runs the SAME dataset for both models sequentially (model A -> model B),
# so we get all (4 model-pair, 8 dataset) cells in 2 sequential rounds @ 1 GPU each.
#
# Round 1: exp50_rest_car_long (CAR + MR-MPL @150k, the SOTA)
# Round 2: exp48_mr_mpl_long2 (CAR-less MR-MPL @150k baseline)

set -u

REPO=/home/ubuntu/toto-eeg
PY=/home/ubuntu/eegModel/.venv/bin/python
RUN_ROOT=/opt/dlami/nvme/eeg/runs

CKPT_EXP50_LONG="${RUN_ROOT}/checkpoints/toto2_eeg_exp50_rest_car_long/epoch=40-step=150000-val_loss=0.2294-train_mr_mpl_step=2.722e-01.ckpt"
CKPT_EXP48_LONG2="${RUN_ROOT}/checkpoints/toto2_eeg_exp48_mr_mpl_long2/epoch=40-step=150000-val_loss=0.2513-train_mr_mpl_step=2.917e-01.ckpt"

DATASETS=(arithmetic_zyma2019 bcic2a bcic2020_3 chbmit faced isruc_sleep mdd_mumtaz2016 physionet)
STRATS="frozen ridge_probe lora"
HEAD="linear_head"
N_SEEDS=3

OUT_EXP50="${RUN_ROOT}/eval_exp50_long_fast"
OUT_EXP48="${RUN_ROOT}/eval_exp48_long2_fast"
mkdir -p "$OUT_EXP50" "$OUT_EXP48" /home/ubuntu/transfer_logs

cd "$REPO/toto2" || { echo "FATAL: $REPO/toto2 not found"; exit 1; }

[[ -f "$CKPT_EXP50_LONG" ]] || { echo "FATAL: exp50_long ckpt not found: $CKPT_EXP50_LONG"; exit 1; }
[[ -f "$CKPT_EXP48_LONG2" ]] || { echo "FATAL: exp48_long2 ckpt not found: $CKPT_EXP48_LONG2"; exit 1; }

run_pair() {
  local gpu=$1
  local ds=$2

  local pid_file="/home/ubuntu/transfer_logs/eval_gpu${gpu}.pid"
  local wrap_log="/home/ubuntu/transfer_logs/eval_gpu${gpu}_wrapper.log"

  local log50="${OUT_EXP50}/${ds}.log"
  local csv50="${OUT_EXP50}/${ds}.csv"
  local log48="${OUT_EXP48}/${ds}.log"
  local csv48="${OUT_EXP48}/${ds}.csv"

  echo "[GPU $gpu / $ds] launching wrapped pair (exp50_long -> exp48_long2)"

  nohup bash -c "
    set -u
    cd $REPO/toto2
    export PYTHONPATH=$REPO/toto2
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo '=== [GPU $gpu / $ds] ROUND 1: exp50_rest_car_long start ==='
    date -u +%FT%TZ
    CUDA_VISIBLE_DEVICES=$gpu \
      $PY -u -m toto2.eval.run_benchmark_fast \
        --checkpoint '$CKPT_EXP50_LONG' \
        --datasets $ds \
        --strategies $STRATS \
        --heads $HEAD \
        --n-seeds $N_SEEDS \
        --device cuda \
        --output '$csv50' \
        > '$log50' 2>&1
    echo '=== [GPU $gpu / $ds] ROUND 1 exit=$? ==='
    date -u +%FT%TZ
    echo '=== [GPU $gpu / $ds] ROUND 2: exp48_mr_mpl_long2 start ==='
    CUDA_VISIBLE_DEVICES=$gpu \
      $PY -u -m toto2.eval.run_benchmark_fast \
        --checkpoint '$CKPT_EXP48_LONG2' \
        --datasets $ds \
        --strategies $STRATS \
        --heads $HEAD \
        --n-seeds $N_SEEDS \
        --device cuda \
        --output '$csv48' \
        > '$log48' 2>&1
    echo '=== [GPU $gpu / $ds] ROUND 2 exit=$? ==='
    date -u +%FT%TZ
    echo '=== [GPU $gpu / $ds] ALL DONE ==='
  " > "$wrap_log" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  disown
  echo "  PID=$pid wrapper-log=$wrap_log"
}

case "${1:-launch}" in
  launch)
    echo "==================================================================="
    echo "Launching weekend eval pipeline on 8 H100s (2 rounds, sequential)"
    echo "  Round 1: exp50_rest_car_long  (CAR + MR-MPL @150k, SOTA)"
    echo "  Round 2: exp48_mr_mpl_long2   (CAR-less MR-MPL @150k baseline)"
    echo "==================================================================="
    for i in "${!DATASETS[@]}"; do
      run_pair "$i" "${DATASETS[$i]}"
    done
    ;;
  status)
    echo "=== weekend-eval status (Mumbai) ==="
    for gpu in 0 1 2 3 4 5 6 7; do
      ds="${DATASETS[$gpu]}"
      pid_file="/home/ubuntu/transfer_logs/eval_gpu${gpu}.pid"
      [[ -f "$pid_file" ]] || { echo "  [GPU $gpu / $ds] (no pid file)"; continue; }
      pid=$(cat "$pid_file")
      r1_csv="${OUT_EXP50}/${ds}.csv"
      r2_csv="${OUT_EXP48}/${ds}.csv"
      r1_rows=0; r2_rows=0
      [[ -f "$r1_csv" ]] && r1_rows=$(tail -n +2 "$r1_csv" 2>/dev/null | wc -l | tr -d ' ')
      [[ -f "$r2_csv" ]] && r2_rows=$(tail -n +2 "$r2_csv" 2>/dev/null | wc -l | tr -d ' ')
      if kill -0 "$pid" 2>/dev/null; then
        etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        echo "  [GPU $gpu / $ds] RUNNING pid=$pid elapsed=$etime  exp50:$r1_rows/9  exp48:$r2_rows/9"
      else
        echo "  [GPU $gpu / $ds] DONE pid=$pid  exp50:$r1_rows/9  exp48:$r2_rows/9"
      fi
    done
    ;;
  *)
    echo "Usage: $0 {launch|status}"
    ;;
esac
