#!/usr/bin/env python3
"""
Optimized open-eeg-bench evaluation on a Toto2 checkpoint.

Performance fixes vs. the stock run_benchmark.py:
  1. preload=True on every Dataset       -> load once, batch from RAM
  2. num_workers=8 on Training            -> parallel data loading
  3. batch_size=256 (was 64)              -> 4x GPU utilization
  4. max_epochs=30 (was 50) + early stop  -> avoid the long flat tail
  5. wandb disabled                       -> skip per-experiment init cost
  6. **incremental CSV write**            -> see progress as it lands;
                                              kill-and-restart is cheap
  7. **shared exca cache**                -> automatic skip of completed
                                              (dataset, strategy, head, seed) runs

The 1+2 pair are by far the biggest win: heavy datasets (chbmit, faced,
isruc_sleep) bottleneck on disk I/O when run with num_workers=0; we saw
3-7x slowdowns from CPU/IO contention with multiple parallel jobs.
preload moves the whole dataset into RAM (we have 2 TB on the box),
eliminating per-batch disk reads entirely.

Usage (drop-in replacement for run_benchmark.py):

    python -m toto2.eval.run_benchmark_fast \\
        --checkpoint <path>.ckpt \\
        --datasets chbmit \\
        --strategies frozen ridge_probe lora \\
        --heads linear_head \\
        --n-seeds 3 \\
        --device cuda \\
        --output /opt/dlami/nvme/eeg/runs/eval_X/chbmit.csv

Restart-safe: re-running the same command will skip experiments whose
EXCA cache already has results. Delete the cache folder to force re-eval.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


ALL_DATASETS = [
    "arithmetic_zyma2019",
    "bcic2a",
    "bcic2020_3",
    "physionet",
    "chbmit",
    "faced",
    "isruc_sleep",
    "mdd_mumtaz2016",
    "seed_v",
    "seed_vig",
    "tuab",
    "tuev",
]


def main():
    parser = argparse.ArgumentParser(description="Run open-eeg-bench on Toto2 (optimized)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="*", default=None, help=f"Choices: {ALL_DATASETS}")
    parser.add_argument("--strategies", nargs="*", default=["frozen"])
    parser.add_argument("--heads", nargs="*", default=["linear_head"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--output", type=str, required=True, help="Output CSV path (incremental writes)")
    # Optimization knobs (with sensible defaults)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per training job")
    parser.add_argument("--batch-size", type=int, default=256, help="Per-batch samples (was 64 in stock)")
    parser.add_argument("--max-epochs", type=int, default=30, help="Cap SGD epochs (was 50)")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience (was 10)")
    parser.add_argument("--no-preload", action="store_true", help="Disable in-RAM preload (debug only)")
    parser.add_argument("--cache-dir", type=str, default="/opt/dlami/nvme/eeg/eval_cache",
                        help="Shared exca cache root (across all checkpoints/datasets)")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    # ---- Quiet down some noisy loggers ----
    for logger_name in ("braindecode", "mne", "torch.utils.data"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    import open_eeg_bench as oeb
    from open_eeg_bench.experiment import collect_completed_results
    from open_eeg_bench.default_configs.experiments import make_all_experiments
    from open_eeg_bench.backbone import PretrainedBackbone

    peft_modules = None
    if any(s in args.strategies for s in ["lora", "ia3", "adalora", "dora", "oft"]):
        peft_modules = ["in_proj", "out_proj", "fc1", "fc2"]

    model_kwargs = dict(
        d_model=args.d_model,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        pool=args.pool,
        context_length=4096,
    )

    # ---- Convert checkpoint to OEB format (strip 'model.' prefix) ----
    ckpt_path = Path(args.checkpoint)
    converted_path = ckpt_path.parent / f"{ckpt_path.stem}_oeb.pt"
    if not converted_path.exists():
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw.get("model_state_dict", raw))
        converted = {}
        for k, v in sd.items():
            clean = k.removeprefix("model.")
            if not clean.startswith("_toto."):
                clean = "_toto." + clean
            converted[clean] = v
        torch.save(converted, converted_path)
        print(f"[fast-eval] Converted checkpoint saved to {converted_path}", flush=True)
    else:
        print(f"[fast-eval] Reusing converted checkpoint {converted_path}", flush=True)

    backbone = PretrainedBackbone.model_validate(dict(
        model_cls="toto2.eval.oeb_adapter.Toto2EEGBenchModel",
        checkpoint_path=str(converted_path),
        model_kwargs=model_kwargs,
        peft_target_modules=peft_modules or [],
        head_module_name="final_layer",
        peft_ff_modules=[],
    ))

    # ---- Build experiment configs ----
    print(f"[fast-eval] Building experiments: datasets={args.datasets} strategies={args.strategies} "
          f"heads={args.heads} n_seeds={args.n_seeds}", flush=True)
    experiments = make_all_experiments(
        datasets=args.datasets,
        heads=args.heads,
        finetuning_strategies=args.strategies,
        n_seeds=args.n_seeds,
    )

    # ---- Override each experiment with optimized settings ----
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    overrides_common = {
        "backbone": backbone,
        "infra": {"folder": str(cache_dir)},
    }

    optimized_experiments = []
    for exp in experiments:
        local_overrides = dict(overrides_common)

        # Dataset: preload to RAM (huge IO win for big datasets)
        if not args.no_preload:
            local_overrides["dataset"] = {"preload": True}

        # Training: more workers, bigger batch, fewer epochs, no wandb
        if exp.training.kind == "sgd":
            local_overrides["training"] = {
                "device": args.device,
                "num_workers": args.num_workers,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "early_stopping": {"enabled": True, "patience": args.patience,
                                   "monitor": "valid_loss", "lower_is_better": True},
                "wandb": {"enabled": False, "project": "open-eeg-bench"},
            }
        else:
            # ridge_probe is closed-form; only the device matters
            local_overrides["training"] = {"device": args.device}

        new_exp = exp.infra.clone_obj(local_overrides)
        optimized_experiments.append(new_exp)

    print(f"[fast-eval] {len(optimized_experiments)} experiments queued. "
          f"Output: {args.output}  Cache: {cache_dir}", flush=True)

    # ---- Run experiments ONE AT A TIME with incremental CSV writes ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally load existing CSV so we can resume / append
    existing_rows = []
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            existing_rows = existing_df.to_dict("records")
            print(f"[fast-eval] Resuming with {len(existing_rows)} existing rows in {out_path}", flush=True)
        except Exception as e:
            print(f"[fast-eval] Could not read existing CSV ({e}); starting fresh", flush=True)

    rows = list(existing_rows)
    for i, exp in enumerate(optimized_experiments, 1):
        uid = exp.infra.uid()
        # Skip if already in CSV (defensive, in addition to exca cache)
        already = any(
            r.get("dataset") == exp.dataset.hf_id
            and r.get("finetuning") == exp.finetuning.kind
            and r.get("head") == exp.head.kind
            and r.get("seed") == exp.seed
            for r in rows
        )
        if already:
            print(f"[fast-eval] [{i}/{len(optimized_experiments)}] SKIP (already in CSV): "
                  f"{exp.dataset.hf_id}/{exp.finetuning.kind}/{exp.head.kind}/seed={exp.seed}", flush=True)
            continue

        print(f"\n[fast-eval] [{i}/{len(optimized_experiments)}] {exp.dataset.hf_id} "
              f"finetuning={exp.finetuning.kind} head={exp.head.kind} seed={exp.seed}", flush=True)

        t0 = time.time()
        try:
            result = exp.run()  # blocks until this single experiment finishes
        except KeyboardInterrupt:
            print(f"[fast-eval] Interrupted by user during experiment {i}; saving partial CSV", flush=True)
            break
        except Exception as e:
            print(f"[fast-eval] FAILED ({type(e).__name__}: {e}); recording row and continuing", flush=True)
            row = {
                "backbone": "Toto2EEGBenchModel",
                "dataset": exp.dataset.hf_id,
                "training": exp.training.kind,
                "finetuning": exp.finetuning.kind,
                "head": exp.head.kind,
                "seed": exp.seed,
                "status": "failed",
                "exception": str(e)[:500],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            continue
        elapsed = time.time() - t0

        # Pull the result row (use the same shape as collect_completed_results)
        row = {
            "backbone": "Toto2EEGBenchModel",
            "dataset": exp.dataset.hf_id,
            "training": exp.training.kind,
            "finetuning": exp.finetuning.kind,
            "head": exp.head.kind,
            "seed": exp.seed,
            "status": "completed",
            "wall_clock_s": round(elapsed, 1),
        }
        if isinstance(result, dict):
            row.update(result)

        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)

        metric_key = "test_balanced_accuracy" if "test_balanced_accuracy" in row else "test_r2"
        metric = row.get(metric_key, "?")
        if isinstance(metric, float):
            metric = f"{metric:.4f}"
        print(f"[fast-eval] [{i}/{len(optimized_experiments)}] DONE in {elapsed:.1f}s  "
              f"{metric_key}={metric}  -> wrote row to {out_path}", flush=True)

    print(f"\n[fast-eval] All done. Final CSV: {out_path} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
