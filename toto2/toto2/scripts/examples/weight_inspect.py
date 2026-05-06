# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
"""Compare FFN/attn weight magnitudes: trained vs fresh init."""

from __future__ import annotations

import argparse, importlib
from collections import defaultdict
from pathlib import Path

import torch, yaml
import dd_unit_scaling as uu

from toto2.training import Toto2ForTraining


def _strip(sd):
    out = {}
    for k, v in sd.items():
        for p in ("model._orig_mod.", "model.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]; break
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
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

    fresh = Toto2ForTraining(config=dict(cfg["model"]), **common)
    trained = Toto2ForTraining(config=dict(cfg["model"]), **common)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _strip(sd)
    trained.model.load_state_dict(sd, strict=False)

    targets = ["attn.in_proj.weight", "attn.out_proj.weight",
               "ffn.fc1.weight", "ffn.fc2.weight"]

    print(f"\n{'param':<40}  {'fresh.std':>12}  {'trained.std':>12}  {'ratio':>8}")
    print("-" * 80)
    for name, p_t in trained.model.named_parameters():
        if any(t in name for t in targets):
            p_f = dict(fresh.model.named_parameters())[name]
            s_t = p_t.detach().float().std().item()
            s_f = p_f.detach().float().std().item()
            ratio = s_t / max(s_f, 1e-12)
            flag = "  <-- DEAD" if ratio < 0.05 else ("  shrunk" if ratio < 0.5 else "")
            print(f"{name:<40}  {s_f:>12.4e}  {s_t:>12.4e}  {ratio:>8.3f}{flag}")

    # Also report mlp_tau buffers per layer
    print("\nResidual tau (attn / mlp) per layer:")
    for n, b in trained.model.named_buffers():
        if 'tau' in n and 'transformer' in n:
            print(f"  {n}: {b.item():.4f}")


if __name__ == "__main__":
    main()
