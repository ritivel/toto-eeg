# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Toto 2.0 EEG deep-dive diagnostic.

Loads a Lightning checkpoint, runs forward + backward on a real EEG batch,
and reports / saves:

1. Per-layer activation RMS (forward) and grad RMS (backward).
2. Per-knot and per-position pinball loss decomposition.
3. Predicted quantile spread distribution across batch / channels.
4. Raw-predictions sanity: a saved PNG plot of (target, predicted median,
   predicted IQR) for 2 channels of 1 sample.
5. EEG-specific scaler diagnostics (loc/scale stability across patches).

Output goes to stdout + an output PNG specified via --plot-out.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from einops import rearrange
from torch.utils.data import DataLoader

import dd_unit_scaling as uu

from toto2.training import Toto2ForTraining, collate_timeseries


def _import_builder(spec: str):
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _collate(samples):
    return collate_timeseries(samples, pad_series_id=-1)


def _strip_lightning_prefix(sd):
    out = {}
    for k, v in sd.items():
        for p in ("model._orig_mod.", "model.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-builder", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--plot-out", default="/tmp/toto2_eeg_deep_dive.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    builder = _import_builder(args.dataset_builder)

    # ------------------------------------------------------------------
    # Build module & load checkpoint
    # ------------------------------------------------------------------
    train_cfg = cfg.get("training", {})
    common = dict(
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

    module = Toto2ForTraining(config=dict(cfg["model"]), **common)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _strip_lightning_prefix(sd)
    missing, unexpected = module.model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {len(sd) - len(unexpected)} tensors  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing[:3]:
        print(f"[ckpt] sample missing keys: {missing[:3]}")
    if unexpected[:3]:
        print(f"[ckpt] sample unexpected keys: {unexpected[:3]}")

    device = torch.device(args.device)
    module.to(device).train()
    uu.cache_fan_values(module.model.named_parameters())
    uu.init_world_size_cache(world_size=1)
    uu.set_grad_accumulation_steps(1)

    cfgm = module.model.config
    n_params = sum(p.numel() for p in module.model.parameters())
    print()
    print("=" * 78)
    print(f"MODEL: {n_params/1e6:.2f}M params  d_model={cfgm.d_model}  "
          f"layers={cfgm.num_layers}  patch={cfgm.patch_size}  "
          f"residual_mult={cfgm.residual_mult}  "
          f"residual_attn_ratio={cfgm.residual_attn_ratio:.4f}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Get a real batch
    # ------------------------------------------------------------------
    train_ds, _ = builder(cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0,
                        shuffle=True, collate_fn=_collate)
    batch = next(iter(loader))
    target = batch["target"].to(device)
    target_mask = batch["target_mask"].to(device)
    series_ids = batch["series_ids"].to(device)
    valid = (series_ids != -1).unsqueeze(-1)

    P = cfgm.patch_size
    ctx = module.context_length
    input_target = target[..., :ctx]
    input_mask = target_mask[..., :ctx]
    gt_next = target[..., P:ctx + P]
    gt_mask = target_mask[..., P:ctx + P] & valid

    # ------------------------------------------------------------------
    # Scaler diagnostic (EEG-specific): how loc / scale evolve across
    # patches. Unstable causal scaler in first ~few patches is a known
    # failure mode for asinh+z-score on EEG amplitudes.
    # ------------------------------------------------------------------
    print("\n[A] PATCHED-CAUSAL-STD SCALER DIAGNOSTIC")
    with torch.no_grad():
        scaled, loc, scale = module.model.scaler(input_target, input_mask)
    loc_pp = rearrange(loc, "b v (s p) -> b v s p", p=P)[..., 0]    # (B, V, S)
    scale_pp = rearrange(scale, "b v (s p) -> b v s p", p=P)[..., 0]
    raw_amp = input_target.abs().mean(dim=-1)                       # (B, V)
    print(f"    raw EEG abs amplitude (avg per ch):  "
          f"mean={raw_amp.mean().item():.3e}  std={raw_amp.std().item():.3e}")
    print(f"    loc (first patch):    mean={loc_pp[..., 0].mean():.3e}  "
          f"std={loc_pp[..., 0].std():.3e}")
    print(f"    loc (last  patch):    mean={loc_pp[..., -1].mean():.3e}  "
          f"std={loc_pp[..., -1].std():.3e}")
    print(f"    scale (first patch):  mean={scale_pp[..., 0].mean():.3e}  "
          f"std={scale_pp[..., 0].std():.3e}  min={scale_pp[..., 0].min():.3e}")
    print(f"    scale (last  patch):  mean={scale_pp[..., -1].mean():.3e}  "
          f"std={scale_pp[..., -1].std():.3e}  min={scale_pp[..., -1].min():.3e}")
    print(f"    fraction of patches with scale < 1e-3: "
          f"{(scale_pp < 1e-3).float().mean().item():.3f}")
    print(f"    fraction of patches with scale at minimum_scale (1e-6): "
          f"{(scale_pp == 1e-6).float().mean().item():.3f}")

    # ------------------------------------------------------------------
    # Per-layer activation RMS (forward only)
    # ------------------------------------------------------------------
    print("\n[B] PER-LAYER FORWARD ACTIVATION RMS")
    activations = OrderedDict()

    def _make_hook(name):
        def _hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                activations[name] = t.detach().float().pow(2).mean().sqrt().item()
        return _hook

    handles = []
    for n, m in module.model.named_modules():
        # Pick representative modules per block.
        if any(s in n for s in ["norm1", "attn.in_proj", "attn.out_proj",
                                "attn.qk_proj", "ffn.fc1", "ffn.fc2",
                                "patch_proj", "output_head"]) and \
           not n.endswith("query_proj") and not n.endswith("key_proj"):
            handles.append(m.register_forward_hook(_make_hook(n)))

    # Run forward (under grad) so we can backprop after
    outputs = module.forward(target=input_target, target_mask=input_mask, series_ids=series_ids)
    for h in handles:
        h.remove()

    # Collapse per-layer summary
    by_block = defaultdict(dict)
    for k, v in activations.items():
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            li = int(parts[2])
            sub = ".".join(parts[3:])
            by_block[li][sub] = v
    print(f"    {'layer':>5} {'norm1':>10} {'attn.in':>10} {'attn.out':>10} {'ffn.fc1':>10} {'ffn.fc2':>10}")
    for li in sorted(by_block):
        b = by_block[li]
        print(f"    {li:>5d} "
              f"{b.get('norm1', float('nan')):>10.3e} "
              f"{b.get('attn.in_proj', float('nan')):>10.3e} "
              f"{b.get('attn.out_proj', float('nan')):>10.3e} "
              f"{b.get('ffn.fc1', float('nan')):>10.3e} "
              f"{b.get('ffn.fc2', float('nan')):>10.3e}")
    print(f"    --- non-block ---")
    for k, v in activations.items():
        if not k.startswith("transformer.layers."):
            print(f"    {k}: {v:.3e}")

    # ------------------------------------------------------------------
    # Loss + per-knot + per-position decomposition
    # ------------------------------------------------------------------
    print("\n[C] LOSS DECOMPOSITION")
    quantiles, loc, scale = outputs.quantiles, outputs.loc, outputs.scale
    loc_per_patch = rearrange(loc, "b v (s p) -> b v s p", p=P)[..., 0]
    scale_per_patch = rearrange(scale, "b v (s p) -> b v s p", p=P)[..., 0]
    gt_per_patch = rearrange(gt_next, "b v (s p) -> b v s p", p=P)
    gt_mask_per_patch = rearrange(gt_mask, "b v (s p) -> b v s p", p=P)
    eps = torch.finfo(gt_per_patch.dtype).eps
    sg = (gt_per_patch - loc_per_patch.unsqueeze(-1)) / (scale_per_patch.unsqueeze(-1) + eps)
    sg = torch.where(gt_mask_per_patch, sg, torch.zeros_like(sg))
    asinh_gt = torch.asinh(sg)
    full_loss = module.loss_fn(quantiles, asinh_gt, weights=gt_mask_per_patch.to(quantiles.dtype))
    print(f"    full mean pinball loss = {full_loss.item():.4f}")

    # Per knot
    knots = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    per_knot = []
    weights = gt_mask_per_patch.to(quantiles.dtype)
    for q in range(len(knots)):
        err = asinh_gt - quantiles[q]
        l = torch.maximum(knots[q] * err, (knots[q] - 1.0) * err)
        per_knot.append(((l * weights).sum() / weights.sum().clamp_min(eps)).item())
    print(f"    per-knot pinball:")
    for k, v in zip(knots, per_knot):
        print(f"      tau={k:.1f}  loss={v:.4f}")

    # Per-position (which patch indices are hardest?)
    n_patches = quantiles.shape[-2]
    per_pos = torch.zeros(n_patches)
    for s in range(n_patches):
        ws = weights[..., s, :].sum().clamp_min(eps)
        ls = 0.0
        for q in range(len(knots)):
            err = asinh_gt[..., s, :] - quantiles[q, ..., s, :]
            l = torch.maximum(knots[q] * err, (knots[q] - 1.0) * err)
            ls = ls + (l * weights[..., s, :]).sum() / ws
        per_pos[s] = ls.item() / len(knots)
    print(f"    per-patch-position loss (n_patches={n_patches}):")
    print(f"      patch  0 (warmup):       {per_pos[0].item():.4f}")
    print(f"      patch  1 (still warmup): {per_pos[1].item():.4f}")
    print(f"      patch  3 (warmed up):    {per_pos[3].item():.4f}")
    print(f"      patch {n_patches//4} (1st quarter):{per_pos[n_patches//4].item():.4f}")
    print(f"      patch {n_patches//2} (mid):       {per_pos[n_patches//2].item():.4f}")
    print(f"      patch {n_patches-1} (last):      {per_pos[-1].item():.4f}")
    print(f"      mean (excl first 3 patches): {per_pos[3:].mean().item():.4f}")

    # Predicted quantile spread distribution
    spr = quantiles.std(dim=0)  # (B, V, S, P)
    print(f"    knot spread: mean={spr.mean().item():.4f}  std={spr.std().item():.4f}  "
          f"min={spr.min().item():.4f}  max={spr.max().item():.4f}")
    print(f"    quantile prediction stats:")
    print(f"      median (q=0.5): mean={quantiles[4].mean():.3e}  std={quantiles[4].std():.3f}")
    print(f"      q=0.1:          mean={quantiles[0].mean():.3e}  std={quantiles[0].std():.3f}")
    print(f"      q=0.9:          mean={quantiles[8].mean():.3e}  std={quantiles[8].std():.3f}")

    # ------------------------------------------------------------------
    # Per-layer gradient norms
    # ------------------------------------------------------------------
    print("\n[D] PER-LAYER GRADIENT RMS")
    module.zero_grad(set_to_none=True)
    full_loss.backward()
    grad_by_block = defaultdict(dict)
    other = {}
    for n, p in module.model.named_parameters():
        if p.grad is None:
            continue
        rms = p.grad.detach().float().pow(2).mean().sqrt().item()
        if n.startswith("transformer.layers."):
            parts = n.split(".")
            li = int(parts[2])
            sub = ".".join(parts[3:])
            grad_by_block[li][sub] = rms
        else:
            other[n] = rms
    print(f"    {'layer':>5} {'norm1.w':>10} {'attn.in.w':>10} {'attn.out.w':>10} {'ffn.fc1.w':>10} {'ffn.fc2.w':>10}")
    for li in sorted(grad_by_block):
        b = grad_by_block[li]
        print(f"    {li:>5d} "
              f"{b.get('norm1.weight', float('nan')):>10.3e} "
              f"{b.get('attn.in_proj.weight', float('nan')):>10.3e} "
              f"{b.get('attn.out_proj.weight', float('nan')):>10.3e} "
              f"{b.get('ffn.fc1.weight', float('nan')):>10.3e} "
              f"{b.get('ffn.fc2.weight', float('nan')):>10.3e}")
    print(f"    --- non-block params ---")
    for k, v in sorted(other.items()):
        print(f"    {k}: {v:.3e}")

    # ------------------------------------------------------------------
    # Raw predictions plot
    # ------------------------------------------------------------------
    print("\n[E] RAW PREDICTION PLOT")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Convert quantiles back to original space (sinh + scale + loc)
        quantiles_real = quantiles.detach().sinh() * scale_per_patch.unsqueeze(0).unsqueeze(-1) \
                         + loc_per_patch.unsqueeze(0).unsqueeze(-1)
        quantiles_real = rearrange(quantiles_real, "q b v s p -> q b v (s p)")
        gt_real = gt_next  # (B, V, T)

        # Pick batch 0, two channels with most observed positions
        b = 0
        n_var = (series_ids[b] != -1).sum().item()
        ch_idx = list(range(min(2, n_var)))

        fig, axes = plt.subplots(len(ch_idx), 2, figsize=(14, 3 * len(ch_idx)))
        if len(ch_idx) == 1:
            axes = axes.reshape(1, 2)
        for row, c in enumerate(ch_idx):
            T = gt_real.shape[-1]
            t = np.arange(T)
            gt_np = gt_real[b, c].cpu().numpy()
            med_np = quantiles_real[4, b, c].cpu().numpy()
            q10 = quantiles_real[0, b, c].cpu().numpy()
            q90 = quantiles_real[8, b, c].cpu().numpy()
            mask_np = gt_mask[b, c].cpu().numpy()

            # Full series
            axes[row, 0].plot(t[mask_np], gt_np[mask_np], color="black", lw=0.6, label="target")
            axes[row, 0].plot(t, med_np, color="C0", lw=0.7, label="predicted median")
            axes[row, 0].fill_between(t, q10, q90, alpha=0.25, color="C0", label="predicted 10-90%")
            axes[row, 0].set_title(f"ch {c} — full series ({mask_np.sum()}/{T} observed)")
            axes[row, 0].legend(loc="upper right", fontsize=8)
            axes[row, 0].set_xlabel("time step")

            # Zoom on a 256-sample window
            zoom_start = T // 2
            zoom = slice(zoom_start, zoom_start + 256)
            zt = np.arange(256) + zoom_start
            zm = mask_np[zoom]
            axes[row, 1].plot(zt[zm], gt_np[zoom][zm], color="black", lw=1.0, label="target")
            axes[row, 1].plot(zt, med_np[zoom], color="C0", lw=1.0, label="median")
            axes[row, 1].fill_between(zt, q10[zoom], q90[zoom], alpha=0.25, color="C0")
            axes[row, 1].set_title(f"ch {c} — zoomed mid-window (256 steps)")
            axes[row, 1].set_xlabel("time step")

        fig.suptitle(f"Toto-2 EEG predictions (ckpt={Path(args.checkpoint).name})", y=1.0)
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=110, bbox_inches="tight")
        print(f"    saved plot to {args.plot_out}")
    except Exception as e:
        print(f"    plot failed: {e!r}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
