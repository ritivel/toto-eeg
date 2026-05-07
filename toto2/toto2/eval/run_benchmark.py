#!/usr/bin/env python3
"""
Run open-eeg-bench evaluation on a Toto2 checkpoint.

Usage (on the remote GPU server):

    # Quick smoke test on one dataset:
    python -m toto2.eval.run_benchmark \
        --checkpoint /opt/dlami/nvme/eeg/runs/checkpoints/.../last.ckpt \
        --datasets tuab \
        --device cuda

    # Full benchmark (all 12 datasets):
    python -m toto2.eval.run_benchmark \
        --checkpoint /opt/dlami/nvme/eeg/runs/checkpoints/.../last.ckpt \
        --device cuda

    # With LoRA fine-tuning:
    python -m toto2.eval.run_benchmark \
        --checkpoint /opt/dlami/nvme/eeg/runs/checkpoints/.../last.ckpt \
        --strategies frozen lora \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run open-eeg-bench on Toto2")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--datasets", nargs="*", default=None, help=f"Datasets to eval (default: all 12). Choices: {ALL_DATASETS}")
    parser.add_argument("--strategies", nargs="*", default=["frozen"], help="Fine-tuning strategies: frozen, ridge_probe, lora, full_finetune")
    parser.add_argument("--heads", nargs="*", default=["linear_head"], help="Classification heads: linear_head, mlp_head")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu, cuda, cuda:0, etc.")
    parser.add_argument("--d-model", type=int, default=384, help="Backbone d_model (must match checkpoint)")
    parser.add_argument("--patch-size", type=int, default=64, help="Backbone patch_size (must match checkpoint)")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy")
    parser.add_argument("--output", type=str, default=None, help="Save results CSV to this path")

    args = parser.parse_args()
    args._config_overrides = None

    import open_eeg_bench as oeb

    peft_modules = None
    if any(s in args.strategies for s in ["lora", "ia3", "adalora", "dora", "oft"]):
        # Target the attention + FFN linear layers in every transformer block.
        # PEFT does suffix matching, so these names match across all
        # ``_toto.transformer.layers.{i}.attn.in_proj``,
        # ``_toto.transformer.layers.{i}.attn.out_proj``,
        # ``_toto.transformer.layers.{i}.ffn.fc1``,
        # ``_toto.transformer.layers.{i}.ffn.fc2``.
        # uu.Linear is a subclass of nn.Linear so PEFT accepts it.
        peft_modules = ["in_proj", "out_proj", "fc1", "fc2"]

    model_kwargs = dict(
        d_model=args.d_model,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        pool=args.pool,
        context_length=4096,
        config_overrides=getattr(args, "_config_overrides", None),
    )

    ckpt_path = Path(args.checkpoint)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw.get("model_state_dict", raw))
    converted = {}
    for k, v in sd.items():
        clean = k.removeprefix("model.")
        if not clean.startswith("_toto."):
            clean = "_toto." + clean
        converted[clean] = v
    converted_path = ckpt_path.parent / f"{ckpt_path.stem}_oeb.pt"
    torch.save(converted, converted_path)
    print(f"Converted checkpoint saved to {converted_path}")

    print(f"Running open-eeg-bench on checkpoint: {args.checkpoint}")
    print(f"  Architecture: d_model={args.d_model}, patch={args.patch_size}, layers={args.num_layers}")
    print(f"  Datasets: {args.datasets or 'all 12'}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Device: {args.device}")
    print()

    results = oeb.benchmark(
        model_cls="toto2.eval.oeb_adapter.Toto2EEGBenchModel",
        checkpoint_path=str(converted_path),
        model_kwargs=model_kwargs,
        datasets=args.datasets,
        finetuning_strategies=args.strategies,
        heads=args.heads,
        n_seeds=args.n_seeds,
        device=args.device,
        head_module_name="final_layer",
        peft_target_modules=peft_modules,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results.to_string())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
