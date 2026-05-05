# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Smoke tests for the Toto 2.0 training pipeline.

These run on CPU with a tiny model and tiny synthetic data. They are not
meant as accuracy tests — they verify that the end-to-end plumbing
(dataset → datamodule → Lightning module → optimizer step) is wired up
correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("lightning")

from toto2.configuration import Toto2ModelConfig  # noqa: E402
from toto2.training import (  # noqa: E402
    ArrayTimeSeriesDataset,
    QuantileLoss,
    SlidingWindowConfig,
    TimeSeriesDataModule,
    Toto2ForTraining,
    quantile_loss,
)


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------


def test_quantile_loss_shapes():
    Q, B, V, S, P = 9, 2, 3, 4, 8
    levels = torch.linspace(0.1, 0.9, Q)
    preds = torch.randn(Q, B, V, S, P)
    targets = torch.randn(B, V, S, P)
    loss = quantile_loss(preds, targets, levels)
    assert loss.shape == (B, V, S, P)
    assert torch.isfinite(loss).all()


def test_quantile_loss_median_is_mae_at_q_0_5():
    """For q=0.5 the pinball loss equals 0.5 * |y - y_hat| (analytic check)."""
    levels = torch.tensor([0.5])
    targets = torch.tensor([1.0, -2.0, 3.0])
    preds = torch.tensor([[0.0, 0.0, 0.0]])
    loss = quantile_loss(preds, targets, levels, reduction="none")
    expected = 0.5 * targets.abs().unsqueeze(0)
    assert torch.allclose(loss, expected)


def test_quantile_loss_huber_smoothing_does_not_blow_up_at_zero():
    levels = torch.tensor([0.5])
    targets = torch.zeros(5)
    preds = torch.zeros(1, 5)
    loss = quantile_loss(preds, targets, levels, huber_kappa=0.05)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_quantile_loss_module_with_weights():
    head = QuantileLoss(quantile_levels=(0.1, 0.5, 0.9), huber_kappa=0.0)
    preds = torch.randn(3, 4, 8)
    targets = torch.randn(4, 8)
    weights = torch.ones(4, 8)
    weights[..., -1] = 0.0  # exclude the last step
    loss = head(preds, targets, weights=weights)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


# ----------------------------------------------------------------------
# Datasets / datamodule
# ----------------------------------------------------------------------


def _make_synthetic_recordings(n=4, channels=3, length=2048, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((channels, length)).astype(np.float32) for _ in range(n)
    ]


def test_array_dataset_window_shapes():
    recs = _make_synthetic_recordings(n=3, channels=4, length=1024)
    cfg = SlidingWindowConfig(context_length=512, patch_size=64, stride=256, random=False)
    ds = ArrayTimeSeriesDataset(recs, cfg)
    sample = ds[0]
    assert sample["target"].shape == (4, 512 + 64)
    assert sample["target_mask"].shape == (4, 512 + 64)
    assert sample["series_ids"].shape == (4,)


def test_collate_pads_variates():
    recs = [
        np.random.randn(3, 1024).astype(np.float32),
        np.random.randn(5, 1024).astype(np.float32),
    ]
    cfg = SlidingWindowConfig(context_length=512, patch_size=64, stride=512, random=False)
    ds = ArrayTimeSeriesDataset(recs, cfg)
    dm = TimeSeriesDataModule(train_dataset=ds, val_dataset=None, train_batch_size=2)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["target"].shape[0] == 2
    assert batch["target"].shape[1] == 5  # padded to max
    # Padded variates must use sentinel id and zero mask:
    assert (batch["series_ids"] == -1).any()
    pad_idx = batch["series_ids"].eq(-1)
    assert (~batch["target_mask"][pad_idx]).all()


# ----------------------------------------------------------------------
# Tiny end-to-end training step
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() and not hasattr(torch, "compile"),
    reason="Skipping E2E test on environments without basic torch compile/CUDA hooks.",
)
def test_lightning_module_training_step():
    pytest.importorskip("dd_unit_scaling")

    # Tiny config: ~tens of thousands of params, runs on CPU.
    patch_size = 32
    context_length = 128
    config = Toto2ModelConfig(
        patch_size=patch_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        layer_group_size=2,
        num_variate_layers_per_group=1,
        variate_layer_first=False,
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(
            context_length=context_length, patch_size=patch_size
        ),
    )

    module = Toto2ForTraining(
        config=config,
        context_length=context_length,
        base_lr=1e-3,
        warmup_steps=0,
        stable_steps=10,
        decay_steps=0,
        weight_decay=0.0,
    )
    module.setup("fit")  # populates u-μP fan caches
    optimizer_dict = module.configure_optimizers()
    optimizer = optimizer_dict["optimizer"]

    B, V = 2, 3
    target = torch.randn(B, V, context_length + patch_size)
    target_mask = torch.ones(B, V, context_length + patch_size, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)

    batch = {
        "target": target,
        "target_mask": target_mask,
        "series_ids": series_ids,
        "num_variates": torch.tensor([V] * B, dtype=torch.long),
    }

    # Single training step: must produce a finite scalar loss.
    optimizer.zero_grad()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_residual_attn_ratio_helper_matches_formula():
    P = 64
    for ctx in (256, 4096, 16_384):
        ratio = Toto2ModelConfig.compute_residual_attn_ratio(context_length=ctx, patch_size=P)
        s = ctx / P
        expected = math.sqrt(s / math.log(s))
        assert ratio == pytest.approx(expected)
