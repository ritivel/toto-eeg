# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Warmup-Stable-Decay (WSD) learning-rate scheduler.

Mirrors the WSD recipe used to pre-train Toto 1.0 and reported in the Toto
paper (Cohen et al., 2025; following Hu et al., MiniCPM, 2024). The schedule
has three phases:

1. **Warmup** — linear ramp from ``min_lr`` to ``base_lr`` over ``warmup_steps``.
2. **Stable** — constant ``base_lr`` for ``stable_steps``.
3. **Decay** — ``1 - sqrt(progress)`` annealing from ``base_lr`` down to
   ``min_lr`` over ``decay_steps``.

Beyond ``warmup + stable + decay`` the LR stays at ``min_lr``. The same shape
is used for both pre-training and continued pre-training; only phase lengths
typically change.
"""

from __future__ import annotations

import math

import torch


class WarmupStableDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup ➜ stable ➜ ``1 - sqrt`` decay learning-rate schedule.

    Each phase length is specified independently in optimizer steps so the
    schedule composes cleanly with gradient accumulation.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 1e-5,
        base_lr: float = 1e-3,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or stable_steps < 0 or decay_steps < 0:
            raise ValueError("All phase lengths must be non-negative.")
        if min_lr < 0 or base_lr < 0:
            raise ValueError("Learning rates must be non-negative.")
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.min_lr = float(min_lr)
        self.base_lr = float(base_lr)
        self.total_steps = self.warmup_steps + self.stable_steps + self.decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            factor = step / max(self.warmup_steps, 1)
            lr = self.min_lr + factor * (self.base_lr - self.min_lr)
        elif step < self.warmup_steps + self.stable_steps:
            lr = self.base_lr
        elif step < self.total_steps:
            progress = (step - self.warmup_steps - self.stable_steps) / max(self.decay_steps, 1)
            factor = 1.0 - math.sqrt(progress)
            lr = self.min_lr + factor * (self.base_lr - self.min_lr)
        else:
            lr = self.min_lr

        return [lr for _ in self.optimizer.param_groups]
