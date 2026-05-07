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
    AMSELoss,
    ArrayTimeSeriesDataset,
    QuantileLoss,
    SlidingWindowConfig,
    TimeSeriesDataModule,
    Toto2ForTraining,
    amse_loss_1d,
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
# AMSE loss (Subich et al. ICML 2025, exp43)
# ----------------------------------------------------------------------


def test_amse_zero_at_perfect_prediction_concat():
    """AMSE(x, x) must be ~zero — global optimum is unique (paper §2.3).

    Allows a small fp32 slack because the orthonormal rfft round-trip
    introduces ~1e-6 noise on randn inputs of unit variance.
    """
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 16)  # (B, V, S, P)
    loss = amse_loss_1d(pred, pred.clone(), concat_patches=True)
    assert torch.isfinite(loss)
    assert loss.abs() < 1e-4, f"AMSE(x,x)={loss.item():.3e} should be ~0"


def test_amse_zero_at_perfect_prediction_per_patch():
    """Same as above but with per-patch FFT."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 16)
    loss = amse_loss_1d(pred, pred.clone(), concat_patches=False)
    assert loss.abs() < 1e-4


def test_amse_close_to_mse_near_optimum():
    """AMSE matches MSE near the optimum — same Taylor expansion (paper §2.3)."""
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 16)
    # Tiny perturbation: pred ≈ target + 0.01*ε.  AMSE / MSE ratio should
    # be very close to 1 (the paper proves they have the same Taylor expansion
    # to second order around the optimum).
    pred = target + 0.01 * torch.randn_like(target)
    amse = amse_loss_1d(pred, target, concat_patches=True)
    # Reference MSE in the spectral sense (Σ_k |α_x − α_y|², orthonormal FFT).
    fft_pred = torch.fft.rfft(pred.flatten(start_dim=2), dim=-1, norm="ortho")
    fft_gt = torch.fft.rfft(target.flatten(start_dim=2), dim=-1, norm="ortho")
    spectral_mse = (fft_pred - fft_gt).abs().pow(2).sum(dim=-1).mean()
    # 5% slack — they are exactly equal in the Taylor limit but with finite
    # perturbation there is a tiny difference from the max-vs-geom-mean
    # cross-term split.
    rel_err = ((amse - spectral_mse).abs() / spectral_mse.clamp_min(1e-9)).item()
    assert rel_err < 0.05, f"AMSE/MSE near optimum should match: {amse.item():.3e} vs {spectral_mse.item():.3e}"


def test_amse_amplitude_only_error_uses_only_amp_term():
    """Amplitude-only mismatch (predict half the signal, same phase) should
    leave coh ≈ 1 and put all the loss in the amplitude term."""
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 16)
    pred = target * 0.5  # exact same phase, half amplitude
    _, diags = amse_loss_1d(
        pred, target, concat_patches=True, return_diagnostics=True
    )
    # Coherence should be exactly 1 (predictions are scalar multiple of target).
    assert diags["amse_coh_mean"].item() == pytest.approx(1.0, abs=1e-5)
    # Coherence term must therefore be ~0; amplitude term carries the loss.
    assert diags["amse_coh_term"].item() == pytest.approx(0.0, abs=1e-5)
    assert diags["amse_amp_term"].item() > 0
    # Amplitude ratio should be ~0.5.
    assert diags["amse_amp_ratio"].item() == pytest.approx(0.5, abs=1e-3)


def test_amse_phase_only_error_uses_only_coh_term():
    """Phase-only mismatch (sign flip) should leave amplitudes equal and put
    all the loss in the coherence term."""
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 16)
    pred = -target  # same magnitude, phase flipped by π
    _, diags = amse_loss_1d(
        pred, target, concat_patches=True, return_diagnostics=True
    )
    # Amplitudes are equal: amplitude term = (|x| − |y|)² = 0.
    assert diags["amse_amp_term"].item() == pytest.approx(0.0, abs=1e-5)
    # Coherence is -1 (perfect anti-correlation), so coh_term = 2*max·(1−(−1)) = 4*max.
    assert diags["amse_coh_mean"].item() == pytest.approx(-1.0, abs=1e-5)
    assert diags["amse_coh_term"].item() > 0
    assert diags["amse_amp_ratio"].item() == pytest.approx(1.0, abs=1e-3)


def test_amse_module_with_mask():
    """AMSELoss module wrapper respects the observation mask."""
    head = AMSELoss(concat_patches=True, spectral_weight_exponent=0.0)
    pred = torch.randn(2, 3, 4, 16)
    target = torch.randn(2, 3, 4, 16)
    mask = torch.ones(2, 3, 4, 16, dtype=torch.bool)
    mask[..., -1, :] = False  # mask the last patch entirely
    loss = head(pred, target, mask=mask, patch_size=16)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_amse_loss_decreases_with_gradient_step():
    """One Adam step on a single linear layer must strictly decrease AMSE.

    Pre-flight sanity check: confirms the loss is differentiable, has
    well-shaped gradients, and produces a usable optimization signal.
    """
    torch.manual_seed(0)
    B, V, S, P = 2, 3, 4, 32
    target = torch.randn(B, V, S, P)
    # A trivial "model": one linear layer over the trailing P axis.
    layer = torch.nn.Linear(P, P, bias=True)
    opt = torch.optim.Adam(layer.parameters(), lr=5e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        # Feed the target itself into the layer; the model has to learn
        # the identity to drive AMSE to zero.
        pred = layer(target)
        loss = amse_loss_1d(pred, target, concat_patches=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.5, f"AMSE should drop ≥ 50% in 20 steps; got {losses}"
    # No NaN / Inf along the way.
    assert all(math.isfinite(x) for x in losses)


def test_amse_spectral_weight_changes_loss_value():
    """A non-zero spectral weight exponent must produce a different loss
    than the uniform default."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 32)
    target = torch.randn(2, 3, 4, 32)
    loss_uniform = amse_loss_1d(pred, target, spectral_weight_exponent=0.0)
    loss_eeg = amse_loss_1d(pred, target, spectral_weight_exponent=2 / 3)
    assert loss_uniform != loss_eeg


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
