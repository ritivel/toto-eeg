# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Single-GPU multi-batch smoke trainer for Toto 2.0 EEG.

Distinct from `lr_probe.py`, which reuses one fixed batch — this script
draws a fresh batch per step from the real DataLoader so it captures the
multi-recording, multi-subject variance of EEG training. Use to confirm
the lr_probe finding generalises before relaunching DDP.

Reports loss + knot-spread per `--print-every` steps. Designed to run in
~5 minutes on one H100 (no DDP, no compile, no checkpointing).
"""

from __future__ import annotations

import argparse
import functools
import importlib
import math
import os
from typing import Optional

import numpy as np
import torch
import yaml
from einops import rearrange
from torch.utils.data import DataLoader

import dd_unit_scaling as uu

from toto2.training import Toto2ForTraining, collate_timeseries


def _collate(samples):
    """Module-level collate (pickle-safe for DataLoader workers)."""
    return collate_timeseries(samples, pad_series_id=-1)


def _import_builder(spec: str):
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _common(cfg, base_lr_override: Optional[float], optimizer_override: Optional[str]):
    train_cfg = cfg.get("training", {})
    return dict(
        context_length=int(cfg.get("data", {}).get("context_length", 4096)),
        base_lr=float(base_lr_override if base_lr_override is not None else train_cfg.get("base_lr", 5e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        stable_steps=int(train_cfg.get("stable_steps", 50_000)),
        decay_steps=int(train_cfg.get("decay_steps", 5_000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        optimizer_name=optimizer_override or train_cfg.get("optimizer", "adamw"),
        huber_kappa=float(train_cfg.get("huber_kappa", 0.0)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-builder", required=True)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override training.base_lr.")
    ap.add_argument("--optimizer", default=None,
                    help="Override training.optimizer.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    builder = _import_builder(args.dataset_builder)
    train_ds, _ = builder(cfg)
    # `num_workers=0` keeps the loader in-process so the lambda-free collate
    # doesn't need to traverse a fork/forkserver pickle hop. Smoke-trainer
    # intentionally avoids worker setup overhead anyway.
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=0, shuffle=True,
        collate_fn=_collate,
    )
    device = torch.device(args.device)

    common = _common(cfg, args.lr, args.optimizer)
    print(f"\nSmoke training: optimizer={common['optimizer_name']!r} "
          f"base_lr={common['base_lr']:.4g} huber_kappa={common['huber_kappa']} "
          f"grad_clip={args.gradient_clip} batch={args.batch_size}\n")

    module = Toto2ForTraining(config=dict(cfg["model"]), **common)

    module.to(device)
    module.train()
    uu.cache_fan_values(module.model.named_parameters())
    uu.init_world_size_cache(world_size=1)
    uu.set_grad_accumulation_steps(1)

    optim_cfg = module.configure_optimizers()
    opt = optim_cfg["optimizer"]
    sch = optim_cfg["lr_scheduler"]["scheduler"]
    # Land at flat base_lr (skip warmup) — this is a steady-state probe.
    sch.last_epoch = sch.warmup_steps

    P = module.model.config.patch_size
    ctx = module.context_length
    eps = torch.finfo(torch.float32).eps

    print(f"{'step':>6} {'loss':>10} {'spread':>10} {'lr_seen_by_param0':>20}")
    losses, spreads = [], []
    step = 0
    for batch in loader:
        if step >= args.steps:
            break
        target = batch["target"].to(device)
        target_mask = batch["target_mask"].to(device)
        series_ids = batch["series_ids"].to(device)
        valid = (series_ids != -1).unsqueeze(-1)

        input_target = target[..., :ctx]
        input_mask = target_mask[..., :ctx]
        gt_next = target[..., P:ctx + P]
        gt_mask = target_mask[..., P:ctx + P] & valid
        gt_per_patch = rearrange(gt_next, "b v (s p) -> b v s p", p=P)
        gt_mask_per_patch = rearrange(gt_mask, "b v (s p) -> b v s p", p=P)

        opt.zero_grad(set_to_none=True)
        outs = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
        loc = rearrange(outs.loc, "b v (s p) -> b v s p", p=P)[..., 0]
        sc = rearrange(outs.scale, "b v (s p) -> b v s p", p=P)[..., 0]
        sg = (gt_per_patch - loc.unsqueeze(-1)) / (sc.unsqueeze(-1) + eps)
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
        if step % args.print_every == 0 or step == args.steps - 1:
            lr_seen = opt.param_groups[0]["lr"]
            print(f"{step:>6d} {loss.item():>10.4f} {ks:>10.3e} {lr_seen:>20.3e}")
        step += 1

    if not losses:
        print("No batches received; check the dataset builder.")
        return 1

    print()
    print(f"Loss:   {losses[0]:.4f} -> {losses[-1]:.4f}  (Δ = {losses[-1]-losses[0]:+.4f})")
    print(f"Spread: {spreads[0]:.3e} -> {spreads[-1]:.3e}")
    print(f"min loss seen: {min(losses):.4f}")
    print(f"max spread seen: {max(spreads):.3e}")

    diverged = any(math.isnan(x) or x > 5 for x in losses)
    if diverged:
        print("VERDICT: DIVERGED")
        return 2
    if losses[-1] - losses[0] < -0.01 and spreads[-1] > spreads[0]:
        print("VERDICT: HEALTHY (loss dropping AND knot spread growing)")
        return 0
    if losses[-1] - losses[0] > 0:
        print("VERDICT: REGRESSED")
        return 3
    print("VERDICT: marginal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
