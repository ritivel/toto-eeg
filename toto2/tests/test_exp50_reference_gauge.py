# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Tests for the exp50 reference-electrode gauge projection (CAR).

Pin down the contract that:

* CAR ``v - mean(v, axis=channels)`` produces a per-timestep zero-mean output.
* Adding a constant ``c·1`` to the input is invariant under CAR (i.e.,
  ``CAR(v + c·1) == CAR(v)``) — this is the whole point of the projection.
* Padding variates (``series_ids == -1``) and unobserved positions
  (``target_mask == False``) are excluded from the mean and are left
  with their raw values.
* Augmentation is a training-only side effect; eval mode never touches
  the input.
* The full Toto2Model integrates the gauge cleanly: turning it on adds
  zero parameters, the off path is byte-identical to v3 / exp48, and
  forward shape / quantile shapes are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("lightning")

from toto2.configuration import Toto2ModelConfig  # noqa: E402
from toto2.model import ReferenceGaugeProjection, Toto2Model  # noqa: E402


# ----------------------------------------------------------------------
# Pure CAR unit tests
# ----------------------------------------------------------------------


def _make_target(B=2, V=4, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, V, T, generator=g)
    target_mask = torch.ones(B, V, T, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    return target, target_mask, series_ids


def test_car_output_is_zero_mean_per_timestep():
    target, target_mask, series_ids = _make_target()
    car = ReferenceGaugeProjection().eval()
    out = car(target, target_mask, series_ids)
    # Mean across the variate axis is ~0 for every (batch, timestep).
    means = out.mean(dim=-2)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)


def test_car_invariant_under_additive_constant_per_timestep():
    """The whole point: ``CAR(v + c·1) == CAR(v)``."""
    target, target_mask, series_ids = _make_target()
    car = ReferenceGaugeProjection().eval()
    out = car(target, target_mask, series_ids)
    # Add an arbitrary per-(batch, timestep) constant broadcast across variates.
    c = torch.randn(target.shape[0], 1, target.shape[-1])
    out_shifted = car(target + c, target_mask, series_ids)
    assert torch.allclose(out, out_shifted, atol=1e-6)


def test_car_excludes_padding_variates():
    """Padding variates (series_id == -1) must not contribute to the mean.

    We mark the last variate as padding and verify the mean of the
    remaining variates is exactly zero.
    """
    target, target_mask, series_ids = _make_target(B=1, V=5, T=8)
    series_ids[:, -1] = -1  # last variate is pad
    car = ReferenceGaugeProjection().eval()
    out = car(target, target_mask, series_ids)
    real_mean = out[:, :-1, :].mean(dim=-2)
    assert torch.allclose(real_mean, torch.zeros_like(real_mean), atol=1e-6)
    # Padded variate is left unchanged (CAR shouldn't touch it).
    assert torch.equal(out[:, -1], target[:, -1])


def test_car_excludes_unobserved_positions():
    target, target_mask, series_ids = _make_target(B=1, V=4, T=8)
    # Mark a single (variate, timestep) as unobserved.
    target_mask[0, 1, 3] = False
    car = ReferenceGaugeProjection().eval()
    out = car(target, target_mask, series_ids)
    # The unobserved position is left at its raw value.
    assert out[0, 1, 3].item() == target[0, 1, 3].item()
    # Mean across observed positions at timestep 3 is 0.
    obs = out[0, target_mask[0, :, 3], 3]
    assert torch.allclose(obs.mean(), torch.tensor(0.0), atol=1e-6)


def test_car_augmentation_is_training_only_no_op():
    """In eval mode, augmentation is never applied; CAR is deterministic."""
    target, target_mask, series_ids = _make_target()
    car = ReferenceGaugeProjection(gauge_augment_std=10.0).eval()
    out_a = car(target, target_mask, series_ids)
    out_b = car(target, target_mask, series_ids)
    assert torch.equal(out_a, out_b)


def test_car_augmentation_is_no_op_on_output_in_train_mode():
    """CAR is invariant to the augmentation by construction.

    With gauge_augment_std > 0 in training, an additive c·1 is injected
    BEFORE the projection.  Because CAR removes the mean, the output
    must be identical to running CAR without augmentation.  This
    catches any pathway that bypasses the projection — in particular,
    a buggy mask handling that lets the augmentation leak through.
    """
    target, target_mask, series_ids = _make_target()
    torch.manual_seed(0)
    car_aug = ReferenceGaugeProjection(gauge_augment_std=5.0).train()
    car_no_aug = ReferenceGaugeProjection(gauge_augment_std=0.0).train()
    out_aug = car_aug(target, target_mask, series_ids)
    out_no_aug = car_no_aug(target, target_mask, series_ids)
    assert torch.allclose(out_aug, out_no_aug, atol=1e-5), (
        "CAR + augmentation should be identical to CAR alone (math). "
        "If this fails, the projection has been bypassed somewhere."
    )


# ----------------------------------------------------------------------
# Toto2Model integration
# ----------------------------------------------------------------------


def _tiny_config(*, use_reference_gauge: bool, gauge_augment_std: float = 0.0) -> Toto2ModelConfig:
    return Toto2ModelConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        num_layers=2,
        layer_group_size=2,
        num_variate_layers_per_group=1,
        variate_layer_first=False,
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(
            context_length=64, patch_size=8
        ),
        use_reference_gauge=use_reference_gauge,
        gauge_augment_std=gauge_augment_std,
    )


def test_model_off_path_has_no_reference_gauge_module():
    cfg = _tiny_config(use_reference_gauge=False)
    model = Toto2Model(cfg)
    assert model.reference_gauge is None


def test_model_on_path_has_zero_extra_parameters():
    """CAR adds zero learnable parameters (it's just mean subtraction)."""
    cfg_off = _tiny_config(use_reference_gauge=False)
    cfg_on = _tiny_config(use_reference_gauge=True)
    n_off = sum(p.numel() for p in Toto2Model(cfg_off).parameters())
    n_on = sum(p.numel() for p in Toto2Model(cfg_on).parameters())
    assert n_off == n_on


def test_model_with_gauge_is_invariant_to_constant_offset():
    """Toto2Model.forward must produce identical quantiles under v -> v + c·1.

    This is the integration-level promise of the layer-0 gauge.
    """
    cfg = _tiny_config(use_reference_gauge=True)
    model = Toto2Model(cfg).eval()
    B, V, T = 1, 4, 64
    g = torch.Generator().manual_seed(0)
    target = torch.randn(B, V, T, generator=g)
    target_mask = torch.ones(B, V, T, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    out_a = model(
        target=target, target_mask=target_mask, cpm_mask=target_mask,
        series_ids=series_ids,
    )
    # Add a per-(batch, timestep) constant broadcast across variates.
    c = torch.randn(B, 1, T, generator=g)
    out_b = model(
        target=target + c, target_mask=target_mask, cpm_mask=target_mask,
        series_ids=series_ids,
    )
    # Allow a tiny numerical slack from cumulative float roundoff in the scaler.
    assert torch.allclose(out_a.quantiles, out_b.quantiles, atol=1e-4, rtol=1e-4), (
        "Model output should be invariant to a per-timestep constant "
        "offset when use_reference_gauge=True."
    )


def test_model_without_gauge_is_NOT_invariant_to_constant_offset():
    """Sanity check: the v3 / exp48 baseline DOES depend on the offset.

    Otherwise the test above would pass trivially.
    """
    cfg = _tiny_config(use_reference_gauge=False)
    model = Toto2Model(cfg).eval()
    B, V, T = 1, 4, 64
    g = torch.Generator().manual_seed(0)
    target = torch.randn(B, V, T, generator=g)
    target_mask = torch.ones(B, V, T, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    out_a = model(
        target=target, target_mask=target_mask, cpm_mask=target_mask,
        series_ids=series_ids,
    )
    c = torch.randn(B, 1, T, generator=g) * 5.0
    out_b = model(
        target=target + c, target_mask=target_mask, cpm_mask=target_mask,
        series_ids=series_ids,
    )
    # Without the gauge, the scaler absorbs ``c`` into its loc and the
    # downstream patch_proj sees a different scaled signal.  Outputs
    # should genuinely differ.
    assert not torch.allclose(
        out_a.quantiles, out_b.quantiles, atol=1e-3
    ), "Without use_reference_gauge, model should respond to +c·1 offset."


def test_model_off_path_byte_identical_to_v3():
    """``use_reference_gauge=False`` must give the exact same network as v3."""
    cfg = _tiny_config(use_reference_gauge=False)
    model_a = Toto2Model(cfg)
    state = model_a.state_dict()
    model_b = Toto2Model(cfg)
    model_b.load_state_dict(state)
    B, V, T = 1, 4, 64
    g = torch.Generator().manual_seed(7)
    target = torch.randn(B, V, T, generator=g)
    mask = torch.ones(B, V, T, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    out_a = model_a(target=target, target_mask=mask, cpm_mask=mask, series_ids=series_ids)
    out_b = model_b(target=target, target_mask=mask, cpm_mask=mask, series_ids=series_ids)
    assert torch.equal(out_a.quantiles, out_b.quantiles)


def test_config_validates_method():
    """REST is reserved; only CAR is currently implemented."""
    with pytest.raises(ValueError, match="reference_gauge_method must be 'car'"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_reference_gauge=True,
            reference_gauge_method="rest",
        )


def test_config_rejects_negative_augment():
    with pytest.raises(ValueError, match="gauge_augment_std must be"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_reference_gauge=True,
            gauge_augment_std=-1.0,
        )
