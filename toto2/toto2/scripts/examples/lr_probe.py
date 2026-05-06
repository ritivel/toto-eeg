# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Toto 2.0 EEG learning-rate probe.

Quick LR-sweep on a single GPU: builds a fresh model for each candidate
LR, takes ``--steps`` AdamW (or NorMuon) updates on the same fixed batch,
and reports

  - per-step loss trajectory
  - knot-spread evolution (head-collapse early-warning)
  - effective LR for representative weight tensors

Use to triage stalled runs before relaunching DDP. Typical invocation::

    python -m toto2.scripts.examples.lr_probe \
        --config toto2/toto2/scripts/configs/pretrain_eeg_from_scratch.yaml \
        --dataset-builder toto2.scripts.examples.eeg_builder:build_datasets \
        --lrs 5e-4 1e-3 2e-3 5e-3 1e-2 \
        --steps 50 --batch-size 4

A healthy LR will show monotonic loss decline and growing knot spread
within 20 steps. A too-small LR mirrors the production failure (loss
flat, spread frozen). A too-large LR diverges (loss explodes, NaNs).
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import dd_unit_scaling as uu
from einops import rearrange

from toto2.training import Toto2ForTraining, collate_timeseries


def _import_builder(spec: str):
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _common(cfg, base_lr: float):
    train_cfg = cfg.get("training", {})
    return dict(
        context_length=int(cfg.get("data", {}).get("context_length", 4096)),
        base_lr=float(base_lr),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        stable_steps=int(train_cfg.get("stable_steps", 50_000)),
        decay_steps=int(train_cfg.get("decay_steps", 5_000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        optimizer_name=train_cfg.get("optimizer", "adamw"),
        huber_kappa=float(train_cfg.get("huber_kappa", 0.0)),
    )


def _build_module(cfg, base_lr: float) -> Toto2ForTraining:
    common = _common(cfg, base_lr)
    model_cfg = dict(cfg.get("model", {}))
    return Toto2ForTraining(config=model_cfg, **common)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-builder", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, required=True,
                    help="Candidate base_lr values, e.g. 5e-4 1e-3 2e-3 5e-3 1e-2.")
    ap.add_argument("--optimizer", default=None,
                    help="Override training.optimizer (adamw|normuon|dion2). "
                         "Default: read from YAML.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    if args.optimizer is not None:
        cfg.setdefault("training", {})["optimizer"] = args.optimizer

    builder = _import_builder(args.dataset_builder)

    # Build ONE batch and reuse for every LR candidate.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds, _ = builder(cfg)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=0, shuffle=True,
        collate_fn=lambda s: collate_timeseries(s, pad_series_id=-1),
    )
    fixed_batch = next(iter(loader))
    device = torch.device(args.device)
    fixed_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fixed_batch.items()}

    target = fixed_batch["target"]
    target_mask = fixed_batch["target_mask"]
    series_ids = fixed_batch["series_ids"]

    print(f"\nFixed batch: target={tuple(target.shape)}  "
          f"obs_frac={target_mask.float().mean().item():.3f}")
    print(f"\nLR sweep: {args.lrs}  (optimizer={cfg.get('training', {}).get('optimizer', 'adamw')}, "
          f"steps={args.steps}, grad_clip={args.gradient_clip})\n")

    summary: List[Dict] = []
    for base_lr in args.lrs:
        torch.manual_seed(args.seed)  # identical init across LRs

        module = _build_module(cfg, base_lr).to(device)
        module.train()
        # u-µP init must precede optimizer construction.
        uu.cache_fan_values(module.model.named_parameters())
        uu.init_world_size_cache(world_size=1)
        uu.set_grad_accumulation_steps(1)

        optim_cfg = module.configure_optimizers()
        opt = optim_cfg["optimizer"]
        # Bypass WSD warmup so we measure the *flat* base_lr response;
        # warmup is what we're trying to escape, not characterize.
        # (Manually set scheduler's last_epoch high so get_lr returns base_lr.)
        sch = optim_cfg["lr_scheduler"]["scheduler"]
        sch.last_epoch = sch.warmup_steps  # land in stable phase

        P = module.model.config.patch_size
        ctx = module.context_length
        input_target = target[..., :ctx]
        input_mask = target_mask[..., :ctx]
        gt_next = target[..., P:ctx + P]
        valid = (series_ids != -1).unsqueeze(-1)
        gt_mask = target_mask[..., P:ctx + P] & valid
        gt_per_patch = rearrange(gt_next, "b v (s p) -> b v s p", p=P)
        gt_mask_per_patch = rearrange(gt_mask, "b v (s p) -> b v s p", p=P)

        eps = torch.finfo(target.dtype).eps
        losses, spreads = [], []
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            outs = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
            loc = rearrange(outs.loc, "b v (s p) -> b v s p", p=P)[..., 0]
            scl = rearrange(outs.scale, "b v (s p) -> b v s p", p=P)[..., 0]
            sg = (gt_per_patch - loc.unsqueeze(-1)) / (scl.unsqueeze(-1) + eps)
            sg = torch.where(gt_mask_per_patch, sg, torch.zeros_like(sg))
            ag = torch.asinh(sg)
            loss = module.loss_fn(outs.quantiles, ag,
                                  weights=gt_mask_per_patch.to(outs.quantiles.dtype))
            with torch.no_grad():
                ks = outs.quantiles.std(dim=0).mean().item()
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(module.model.parameters(),
                                               max_norm=args.gradient_clip)
            opt.step()
            losses.append(loss.item())
            spreads.append(ks)

        l0, lN = losses[0], losses[-1]
        s0, sN = spreads[0], spreads[-1]
        any_nan = any(math.isnan(x) for x in losses)
        diverged = any_nan or lN > 2 * l0
        verdict = (
            "DIVERGED" if diverged
            else "stuck" if abs(lN - l0) < 1e-3 and sN < 5e-3
            else "moving" if (l0 - lN) > 5e-3 or (sN - s0) > 5e-3
            else "marginal"
        )
        summary.append(dict(base_lr=base_lr, l0=l0, lN=lN, s0=s0, sN=sN, verdict=verdict))

        # Print compact trajectory: every 5 steps
        traj = ", ".join(f"{losses[i]:.4f}" for i in range(0, args.steps, max(1, args.steps // 10)))
        spread_traj = ", ".join(f"{spreads[i]:.2e}" for i in range(0, args.steps, max(1, args.steps // 10)))
        print(f"base_lr={base_lr:.1e}  loss[0]={l0:.4f}  loss[{args.steps-1}]={lN:.4f}  "
              f"spread[0]={s0:.2e}  spread[{args.steps-1}]={sN:.2e}  -> {verdict}")
        print(f"  loss   traj: {traj}")
        print(f"  spread traj: {spread_traj}")

        del module, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(f"{'base_lr':<10} {'l0':>8} {'lN':>8} {'Δl':>8} {'spread0':>10} {'spreadN':>10} {'verdict':>10}")
    for r in summary:
        print(f"{r['base_lr']:<10.1e} {r['l0']:>8.4f} {r['lN']:>8.4f} "
              f"{r['lN']-r['l0']:>+8.4f} {r['s0']:>10.2e} {r['sN']:>10.2e} {r['verdict']:>10}")

    # Recommendation: largest LR that is still "moving" without diverging.
    movers = [r for r in summary if r["verdict"] == "moving"]
    if movers:
        best = max(movers, key=lambda r: r["base_lr"])
        print(f"\nRecommended base_lr: {best['base_lr']:.1e}  (largest stable LR with movement)")
    else:
        print("\nNo LR in the sweep produced clear movement. Try widening the range:")
        print("  - if all stuck:    --lrs 5e-3 1e-2 2e-2 5e-2")
        print("  - if all diverged: --lrs 5e-5 1e-4 2e-4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
