# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Toto 2.0 EEG training sanity check.

Diagnoses why a training run has stalled by inspecting:

1. **Effective LRs** for every parameter (after u-µP fan-in/depth scaling).
2. **Prediction collapse** — does the head emit identical quantiles?
   The 9-knot pinball loss has a known constant-median floor of ~0.27 on
   asinh(N(0,1)) targets; a stuck loss near that floor means the head has
   collapsed to median ≈ 0 with zero quantile spread.
3. **Per-layer activation / gradient norms** — verifies signal flow.
4. **Optimizer step magnitude** — confirms parameters actually move.
5. **Target distribution** — verifies the asinh-z-scored EEG residual
   really is the standard-normal-ish quantity the head was trained for.

Usage::

    python -m toto2.scripts.examples.sanity_check \\
        --config toto2/toto2/scripts/configs/pretrain_eeg_from_scratch.yaml \\
        --dataset-builder toto2.scripts.examples.eeg_builder:build_datasets \\
        --checkpoint /opt/dlami/nvme/eeg/runs/checkpoints/toto2_eeg_pretrain/last.ckpt

The script is read-only with respect to the original training run — it loads
the checkpoint into a fresh process, runs one forward / backward / optim
step in isolation, prints a structured report, and exits.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import dd_unit_scaling as uu

from toto2.training import Toto2ForTraining, collate_timeseries
from toto2.training.lazy_npz import LazyNpzTimeSeriesDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_builder(spec: str):
    if ":" not in spec:
        raise ValueError(f"--dataset-builder must be 'module:callable', got {spec!r}.")
    mod_path, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod_path), attr)


def _section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _fmt(x: Any, fmt: str = ".4g") -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().mean().item() if x.numel() > 1 else x.detach().float().item()
    return f"{x:{fmt}}"


def _stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    return (
        f"shape={tuple(t.shape)} "
        f"mean={t.mean().item():+.3e} "
        f"std={t.std().item():.3e} "
        f"min={t.min().item():+.3e} "
        f"max={t.max().item():+.3e} "
        f"finite_frac={torch.isfinite(t).float().mean().item():.3f}"
    )


def _quantile_loss_per_knot(
    quantiles: torch.Tensor,           # (Q, B, V, S, P)
    targets: torch.Tensor,             # (B, V, S, P)
    weights: torch.Tensor,             # (B, V, S, P), float
) -> torch.Tensor:
    """Return per-knot mean pinball loss, shape (Q,)."""
    Q = quantiles.shape[0]
    levels = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device=quantiles.device, dtype=quantiles.dtype,
    )
    err = targets.unsqueeze(0) - quantiles      # (Q, B, V, S, P)
    w = weights.to(quantiles.dtype).expand_as(err[0])
    eps = torch.finfo(err.dtype).eps
    out = torch.empty(Q, device=quantiles.device)
    for q in range(Q):
        l = torch.maximum(levels[q] * err[q], (levels[q] - 1.0) * err[q])
        out[q] = (l * w).sum() / w.sum().clamp_min(eps)
    return out


# ---------------------------------------------------------------------------
# Diagnostic: effective LRs (u-µP)
# ---------------------------------------------------------------------------


def report_effective_lrs(model: torch.nn.Module, base_lr: float) -> None:
    """Compute base_lr × depth_scale × (1/√fan_in if weight) for every param."""
    _section("EFFECTIVE LEARNING RATES (base_lr=%g, after u-µP scaling)" % base_lr)

    # cache_fan_values must have been called by the LightningModule.setup hook;
    # we replay it here defensively in case the user hasn't done so yet.
    try:
        uu.cache_fan_values(model.named_parameters())
    except Exception as e:
        print(f"[warn] cache_fan_values failed: {e}")

    from unit_scaling.optim import lr_scale_for_depth, _get_fan_in as _ufa

    rows: list[Tuple[str, str, int, Optional[int], float]] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    untagged: list[str] = []

    for name, p in model.named_parameters():
        if not hasattr(p, "mup_type"):
            untagged.append(name)
            continue
        mup_type = p.mup_type
        depth = getattr(p, "mup_scaling_depth", None)
        depth_scale = lr_scale_for_depth(p)  # 1/sqrt(depth) or 1
        if mup_type == "weight":
            fan_in = _ufa(p) if p.ndim >= 2 else 1
            scale = depth_scale * fan_in ** -0.5
        else:
            fan_in = None
            scale = depth_scale
        eff_lr = base_lr * scale
        rows.append((name, mup_type, p.numel(), fan_in, eff_lr))
        by_type[mup_type].append(eff_lr)

    # Group summary
    print(f"{'mup_type':<10} {'count':>6} {'min_lr':>12} {'median_lr':>12} {'max_lr':>12}")
    for k, v in sorted(by_type.items()):
        a = np.array(v)
        print(f"{k:<10} {len(v):>6d} {a.min():>12.3e} {float(np.median(a)):>12.3e} {a.max():>12.3e}")
    if untagged:
        print(f"\n[!] {len(untagged)} params have NO mup_type (will use default LR={base_lr:g}); "
              f"first 5: {untagged[:5]}")

    # Highlight bottom 8 / top 8
    print("\nBottom 8 effective LRs:")
    for name, t, n, f, lr in sorted(rows, key=lambda r: r[4])[:8]:
        print(f"  {lr:.3e}  fan_in={f}  type={t:<6}  numel={n:<7}  {name}")
    print("\nTop 8 effective LRs:")
    for name, t, n, f, lr in sorted(rows, key=lambda r: -r[4])[:8]:
        print(f"  {lr:.3e}  fan_in={f}  type={t:<6}  numel={n:<7}  {name}")


# ---------------------------------------------------------------------------
# Diagnostic: per-layer activation hooks
# ---------------------------------------------------------------------------


class _ActHook:
    """Captures the L2 RMS of forward outputs for selected modules."""

    def __init__(self):
        self.records: list[Tuple[str, float, tuple]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: torch.nn.Module, names_substr=("attn", "ffn", "norm", "out_proj")):
        for name, mod in model.named_modules():
            if any(s in name.lower() for s in names_substr):
                self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return
            v = t.detach().float()
            rms = (v.pow(2).mean()).sqrt().item()
            self.records.append((name, rms, tuple(v.shape)))
        return _hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_one_batch(cfg: Dict[str, Any], builder, device: torch.device, n: int = 4):
    train_ds, _ = builder(cfg)
    loader = DataLoader(
        train_ds,
        batch_size=n,
        num_workers=0,
        shuffle=True,
        collate_fn=lambda s: collate_timeseries(s, pad_series_id=-1),
    )
    batch = next(iter(loader))
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Same YAML used for training.")
    ap.add_argument("--dataset-builder", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Optional Lightning .ckpt; if missing, a fresh model is initialised.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4,
                    help="Number of optimizer steps to take during the move-the-needle test.")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    builder = _import_builder(args.dataset_builder)

    # ------------------------------------------------------------------
    # Build the same Lightning module the training run used.
    # ------------------------------------------------------------------
    _section("STEP 1 - Build / load model")
    train_cfg = cfg.get("training", {})
    common_kwargs = dict(
        context_length=int(cfg.get("data", {}).get("context_length", 4096)),
        base_lr=float(train_cfg.get("base_lr", 5e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        stable_steps=int(train_cfg.get("stable_steps", 50_000)),
        decay_steps=int(train_cfg.get("decay_steps", 5_000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        optimizer_name=train_cfg.get("optimizer", "adamw"),
        huber_kappa=float(train_cfg.get("huber_kappa", 0.0)),
    )

    model_cfg = dict(cfg.get("model", {}))
    if not model_cfg:
        raise ValueError("config.model is required.")
    module = Toto2ForTraining(config=model_cfg, **common_kwargs)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading Lightning checkpoint: {args.checkpoint}")
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # Strip every Lightning / torch.compile prefix combination encountered:
        #   "model._orig_mod.X"  (compile-wrapped Lightning)
        #   "model.X"            (uncompiled Lightning)
        #   "_orig_mod.X"        (compiled bare module)
        cleaned = {}
        for k, v in sd.items():
            for pref in ("model._orig_mod.", "model.", "_orig_mod."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    break
            cleaned[k] = v
        missing, unexpected = module.model.load_state_dict(cleaned, strict=False)
        n_loaded = len(cleaned) - len(unexpected)
        print(f"loaded {n_loaded} tensors from checkpoint "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[warn] first 5 missing: {missing[:5]}")
        if unexpected:
            print(f"[warn] first 5 unexpected: {unexpected[:5]}")
    else:
        print("Using freshly-initialised weights (no checkpoint provided).")

    device = torch.device(args.device)
    module.to(device)
    module.train()

    n_params = sum(p.numel() for p in module.model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params  device={device}  d_model={module.model.config.d_model}  "
          f"num_layers={module.model.config.num_layers}  patch_size={module.model.config.patch_size}  "
          f"residual_mult={module.model.config.residual_mult}  "
          f"residual_attn_ratio={module.model.config.residual_attn_ratio:.4f}")

    # u-µP requires this BEFORE optimizer construction
    uu.cache_fan_values(module.model.named_parameters())
    uu.init_world_size_cache(world_size=1)
    uu.set_grad_accumulation_steps(1)

    # ------------------------------------------------------------------
    report_effective_lrs(module.model, base_lr=common_kwargs["base_lr"])

    # ------------------------------------------------------------------
    _section("STEP 2 - Build a real EEG batch")
    batch = build_one_batch(cfg, builder, device, n=args.batch_size)
    print("target:        ", _stats(batch["target"]))
    print("target_mask:   ", _stats(batch["target_mask"].float()))
    print("series_ids:    ", _stats(batch["series_ids"].float()))

    # ------------------------------------------------------------------
    _section("STEP 3 - Forward pass with per-layer activation hooks")
    hook = _ActHook()
    hook.attach(module.model)
    with torch.no_grad():
        # Deterministic forward (no grad needed yet)
        from einops import rearrange
        target = batch["target"]
        target_mask = batch["target_mask"]
        series_ids = batch["series_ids"]
        P = module.model.config.patch_size
        ctx = module.context_length
        input_target = target[..., :ctx]
        input_mask = target_mask[..., :ctx]
        outs = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
        quantiles, loc, scale = outs.quantiles, outs.loc, outs.scale
    hook.detach()

    print(f"\nQuantiles output: {_stats(quantiles)}")
    print(f"loc:              {_stats(loc)}")
    print(f"scale:            {_stats(scale)}")

    # Quantile spread = std across the 9 knots, averaged. Should be > 0 if head learned spread.
    knot_spread = quantiles.std(dim=0).mean().item()
    print(f"Knot spread (std across 9 quantiles, mean): {knot_spread:.3e}  "
          f"(<1e-2 ⇒ head has collapsed to ~constant median)")

    # Per-layer activation summary
    print("\nPer-module forward-output RMS (first 24 entries):")
    for nm, rms, sh in hook.records[:24]:
        print(f"  rms={rms:.3e}  shape={sh}  {nm}")
    print(f"... ({len(hook.records)} total hook records)")

    # ------------------------------------------------------------------
    _section("STEP 4 - Compute training loss + per-knot pinball decomposition")
    # Reproduce the LightningModule's _step logic but in eval-friendly form.
    from einops import rearrange
    P = module.model.config.patch_size
    ctx = module.context_length
    target = batch["target"]
    target_mask = batch["target_mask"]
    series_ids = batch["series_ids"]
    valid_var = (series_ids != -1).unsqueeze(-1)

    input_target = target[..., :ctx]
    input_mask = target_mask[..., :ctx]
    gt_next = target[..., P:ctx + P]
    gt_mask = (target_mask[..., P:ctx + P] & valid_var)

    outputs = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
    quantiles, loc, scale = outputs.quantiles, outputs.loc, outputs.scale
    loc_per_patch = rearrange(loc, "b v (s p) -> b v s p", p=P)[..., 0]
    scale_per_patch = rearrange(scale, "b v (s p) -> b v s p", p=P)[..., 0]
    gt_per_patch = rearrange(gt_next, "b v (s p) -> b v s p", p=P)
    gt_mask_per_patch = rearrange(gt_mask, "b v (s p) -> b v s p", p=P)
    eps = torch.finfo(gt_per_patch.dtype).eps
    scaled_gt = (gt_per_patch - loc_per_patch.unsqueeze(-1)) / (scale_per_patch.unsqueeze(-1) + eps)
    scaled_gt = torch.where(gt_mask_per_patch, scaled_gt, torch.zeros_like(scaled_gt))
    asinh_gt = torch.asinh(scaled_gt)

    print(f"asinh(scaled gt) target dist: {_stats(asinh_gt[gt_mask_per_patch])}")

    knot_losses = _quantile_loss_per_knot(
        quantiles, asinh_gt, gt_mask_per_patch.to(quantiles.dtype),
    )
    knots = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nPer-knot pinball loss:")
    for k, v in zip(knots, knot_losses.tolist()):
        print(f"  τ={k:.1f}  loss={v:.4f}")
    print(f"Mean (training objective): {knot_losses.mean().item():.4f}")
    spread_of_per_knot = knot_losses.std().item()
    print(f"Spread of per-knot losses (std across τ): {spread_of_per_knot:.4f}  "
          f"(very small ⇒ head outputs identical quantiles ⇒ collapse confirmed)")
    print("\nReference theoretical floors:")
    print("  Constant 0 predictor on asinh(N(0,1)):     ~0.329")
    print("  Optimal Gaussian quantile predictor:        ~0.270")
    print("  Constant 0 predictor on raw N(0,1):         ~0.399 (1/sqrt(2pi))")

    # ------------------------------------------------------------------
    _section("STEP 5 - Backward pass: per-layer grad-RMS + global grad-norm")
    module.zero_grad(set_to_none=True)
    loss = module.loss_fn(quantiles, asinh_gt, weights=gt_mask_per_patch.to(quantiles.dtype))
    print(f"loss = {loss.item():.4f}  (matches lightning training_step)")
    loss.backward()
    # First report raw, unclipped grad statistics.
    grad_rms = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    n_zero, n_total = 0, 0
    for name, p in module.model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        rms = g.pow(2).mean().sqrt().item()
        grad_rms.append((name, rms, p.numel()))
        kind = "weight" if (hasattr(p, "mup_type") and p.mup_type == "weight") else (
            getattr(p, "mup_type", "untagged"))
        by_kind[kind].append(rms)
        n_total += 1
        if rms < 1e-8:
            n_zero += 1
    total = math.sqrt(sum(p.grad.pow(2).sum().item() for p in module.model.parameters() if p.grad is not None))
    print(f"\nUNCLIPPED global grad L2 norm: {total:.4e}")
    # Now apply the grad clip used in training (gradient_clip_val=1.0) and re-report.
    torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
    clipped = math.sqrt(sum(p.grad.pow(2).sum().item() for p in module.model.parameters() if p.grad is not None))
    print(f"CLIPPED   global grad L2 norm: {clipped:.4e}  (training run reports ~1e-3 here)")
    print(f"{n_zero}/{n_total} parameters have grad-RMS < 1e-8 (effectively zero before clipping)")
    print(f"\nGrad-RMS by mup_type:")
    print(f"{'kind':<12} {'count':>6} {'min':>10} {'median':>10} {'max':>10}")
    for k, v in sorted(by_kind.items()):
        a = np.array(v)
        print(f"{k:<12} {len(v):>6d} {a.min():>10.3e} {float(np.median(a)):>10.3e} {a.max():>10.3e}")

    print("\nBottom 6 grad-RMS:")
    for name, rms, n in sorted(grad_rms, key=lambda r: r[1])[:6]:
        print(f"  rms={rms:.3e}  numel={n:<7}  {name}")
    print("\nTop 6 grad-RMS:")
    for name, rms, n in sorted(grad_rms, key=lambda r: -r[1])[:6]:
        print(f"  rms={rms:.3e}  numel={n:<7}  {name}")

    # ------------------------------------------------------------------
    _section("STEP 6 - Optimizer step: do parameters actually move? (with grad clip 1.0)")
    # Snapshot weights, take args.steps optimizer steps, measure delta.
    optim_cfg = module.configure_optimizers()
    opt = optim_cfg["optimizer"]
    snap = {n: p.detach().clone() for n, p in module.model.named_parameters() if p.requires_grad}
    losses = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        outs2 = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
        loc2, sc2 = outs2.loc, outs2.scale
        loc_pp = rearrange(loc2, "b v (s p) -> b v s p", p=P)[..., 0]
        sc_pp = rearrange(sc2, "b v (s p) -> b v s p", p=P)[..., 0]
        sg = (gt_per_patch - loc_pp.unsqueeze(-1)) / (sc_pp.unsqueeze(-1) + eps)
        sg = torch.where(gt_mask_per_patch, sg, torch.zeros_like(sg))
        ag = torch.asinh(sg)
        l2 = module.loss_fn(outs2.quantiles, ag, weights=gt_mask_per_patch.to(outs2.quantiles.dtype))
        l2.backward()
        # Match real training: gradient_clip_val=1.0 in the configs.
        torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(l2.item())
        print(f"  step {step}: loss={l2.item():.5f}")

    deltas: list[Tuple[str, float, float]] = []
    for n, p in module.model.named_parameters():
        if not p.requires_grad:
            continue
        rel = (p.detach() - snap[n]).abs().max().item() / (snap[n].abs().max().item() + 1e-12)
        absd = (p.detach() - snap[n]).abs().mean().item()
        deltas.append((n, absd, rel))

    print(f"\nLoss trajectory over {args.steps} steps: {[round(x,5) for x in losses]}")
    print(f"Loss change: {losses[-1] - losses[0]:+.5f}")

    deltas.sort(key=lambda r: -r[1])
    print(f"\nTop 5 absolute mean parameter change after {args.steps} steps:")
    for n, ad, rd in deltas[:5]:
        print(f"  abs_mean_delta={ad:.3e}  rel_max_delta={rd:.3e}  {n}")
    print(f"\nBottom 5 absolute mean parameter change:")
    for n, ad, rd in deltas[-5:]:
        print(f"  abs_mean_delta={ad:.3e}  rel_max_delta={rd:.3e}  {n}")

    # ------------------------------------------------------------------
    _section("VERDICT")
    base_lr = common_kwargs["base_lr"]
    median_eff_weight_lr = float(np.median(
        [base_lr * uu.optim.lr_scale_for_depth(p)
         * (1.0 / math.sqrt(uu.optim._get_fan_in_base(p)) if p.ndim >= 2 else 1.0)
         for n, p in module.model.named_parameters()
         if hasattr(p, "mup_type") and p.mup_type == "weight"]
    )) if any(getattr(p, "mup_type", None) == "weight" for p in module.model.parameters()) else float("nan")
    print(f"base_lr (config):                      {base_lr:.2e}")
    print(f"median effective LR for `weight`:      {median_eff_weight_lr:.2e}")
    print(f"global grad L2 norm at init/ckpt:      {total:.2e}")
    print(f"per-knot loss spread:                  {spread_of_per_knot:.4f}  "
          f"(< 0.02 ⇒ head collapse)")
    print(f"loss change over {args.steps} steps:                {losses[-1]-losses[0]:+.5f}")
    print()
    print("Suggested next actions if any of these red flags fire:")
    print("  - If grad-RMS for trunk weights is < 1e-5 ⇒ activations too small, "
          "raise base_lr or check residual_mult / residual_attn_ratio.")
    print("  - If knot spread < 1e-2 ⇒ head collapsed; raise base_lr (output knots get FULL "
          "base_lr, no fan-in scaling).")
    print("  - If loss doesn't drop after 4 steps with the current LR ⇒ relaunch with 4-10x "
          "higher base_lr (recipe: pretrain_eeg_22m_aggressive.yaml).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
