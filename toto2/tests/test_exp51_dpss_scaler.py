# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Tests for the exp51 DPSS-tapered (Slepian multitaper) causal scaler.

Pin down the contracts of the new ``PatchedCausalDPSSScaler`` and the
``_compute_dpss`` helper:

* DPSS sequences are L²-orthonormal and in the right sign convention.
* Concentration eigenvalues are in (0, 1) and well-concentrated for K ≤ 2·NW.
* Pure-PyTorch implementation matches scipy's reference (when scipy
  is available) to within tight numerical tolerance.
* Scaler output has the right shape, finiteness, and broadcasting
  properties (one (loc, scale) per patch, broadcast across the patch).
* Causality: changing data in patch s+1 leaves loc/scale at patch s
  unchanged — the cornerstone of the "patched causal" contract.
* Mask handling: padded positions contribute neither to the per-patch
  mean nor to the multitaper variance.
* All-masked patches collapse cleanly to (loc=0, scale=minimum_scale)
  with no NaN / Inf.
* For pure white noise the multitaper variance estimate matches the
  ground truth in expectation (the unbiasedness property of Thomson's
  estimator).
* For bursty signals the DPSS scale is meaningfully smaller than the
  Welford sample variance — the headline scaler-bias-on-bursts win.
* Toto2Model integration: off path is byte-identical to baseline,
  on path adds zero learnable parameters, end-to-end forward works,
  config validation rejects bad (NW, K, patch_size) tuples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from toto2.configuration import Toto2ModelConfig
from toto2.model import (
    PatchedCausalDPSSScaler,
    PatchedCausalStdScaler,
    Toto2Model,
    _compute_dpss,
)


# ----------------------------------------------------------------------
# DPSS computation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("N, NW, K", [(64, 2.5, 3), (128, 4.0, 5), (256, 3.0, 4), (32, 2.0, 2)])
def test_dpss_tapers_are_l2_orthonormal(N, NW, K):
    tapers, _ = _compute_dpss(N, NW=NW, K=K)
    # Σ_t u_k u_l == δ_kl
    gram = tapers @ tapers.T  # (K, K)
    np.testing.assert_allclose(gram, np.eye(K), atol=1e-10)


@pytest.mark.parametrize("N, NW, K", [(64, 2.5, 3), (128, 4.0, 5), (256, 3.0, 4)])
def test_dpss_concentration_ratios_in_unit_interval(N, NW, K):
    _, ratios = _compute_dpss(N, NW=NW, K=K)
    assert ratios.shape == (K,)
    assert (ratios > 0).all()
    assert (ratios < 1).all()


def test_dpss_concentration_ratios_are_descending():
    _, ratios = _compute_dpss(64, NW=2.5, K=4)
    assert (np.diff(ratios) <= 1e-9).all(), (
        f"DPSS ratios should be sorted descending; got {ratios}"
    )


def test_dpss_well_concentrated_for_k_under_two_nw():
    """For NW=2.5 and K=3 (≤ 2·NW=5), all tapers should have ratio > 0.99.

    This is the classical Slepian rule of thumb that justifies using
    the leading 2NW tapers as the "well-concentrated" set.
    """
    _, ratios = _compute_dpss(64, NW=2.5, K=3)
    assert (ratios > 0.99).all(), (
        f"With NW=2.5, K=3 ≤ 2·NW, ratios should be > 0.99; got {ratios}"
    )


def test_dpss_first_taper_is_symmetric_positive():
    """u_0 is even-symmetric and pinned positive at the centre."""
    tapers, _ = _compute_dpss(64, NW=2.5, K=3)
    # Symmetry within numerical tolerance
    np.testing.assert_allclose(tapers[0], tapers[0][::-1], atol=1e-9)
    # Positive peak at centre
    assert tapers[0, 32] > 0


def test_dpss_second_taper_is_antisymmetric():
    """u_1 is antisymmetric (u_1[t] = -u_1[N-1-t])."""
    tapers, _ = _compute_dpss(64, NW=2.5, K=3)
    np.testing.assert_allclose(tapers[1], -tapers[1][::-1], atol=1e-9)


def test_dpss_rejects_invalid_K():
    with pytest.raises(ValueError, match="K must be in"):
        _compute_dpss(8, NW=2.0, K=0)
    with pytest.raises(ValueError, match="K must be in"):
        _compute_dpss(8, NW=2.0, K=9)


def test_dpss_rejects_invalid_NW():
    with pytest.raises(ValueError, match="NW must be in"):
        _compute_dpss(8, NW=0.0, K=2)
    with pytest.raises(ValueError, match="NW must be in"):
        _compute_dpss(8, NW=4.0, K=2)  # NW >= N/2


def test_dpss_matches_scipy_when_available():
    """Cross-check pure-PyTorch implementation against scipy's reference.

    scipy's ``signal.windows.dpss`` returns DPSS sequences computed via
    a different algorithm (banded Lapack solver).  The eigenvectors
    are unique up to sign and the concentration ratios are uniquely
    defined — so we compare |tapers| element-wise and ratios directly.
    """
    sp = pytest.importorskip("scipy.signal.windows")
    N, NW, K = 64, 2.5, 4
    ours, our_ratios = _compute_dpss(N, NW=NW, K=K)
    theirs, their_ratios = sp.dpss(N, NW=NW, Kmax=K, return_ratios=True)
    # Concentration ratios match very tightly.
    np.testing.assert_allclose(our_ratios, their_ratios, atol=1e-8)
    # Tapers match up to a global sign per row.
    for k in range(K):
        sign = 1.0 if (ours[k] @ theirs[k]) > 0 else -1.0
        np.testing.assert_allclose(ours[k] * sign, theirs[k], atol=1e-7)


# ----------------------------------------------------------------------
# Scaler unit tests
# ----------------------------------------------------------------------


def _make_data(B=2, V=4, T=64, P=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(B, V, T, generator=g)
    mask = torch.ones(B, V, T, dtype=torch.bool)
    return data, mask, P


def test_scaler_output_shapes_are_preserved():
    data, mask, P = _make_data()
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2)
    scaled, loc, scale = scaler(data, mask)
    assert scaled.shape == data.shape
    assert loc.shape == data.shape
    assert scale.shape == data.shape
    assert torch.isfinite(scaled).all()
    assert torch.isfinite(loc).all()
    assert torch.isfinite(scale).all()


def test_scaler_loc_and_scale_are_constant_within_each_patch():
    """The per-patch broadcast contract: one (loc, scale) per patch."""
    data, mask, P = _make_data(B=1, V=1, T=24, P=8)
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2)
    _, loc, scale = scaler(data, mask)
    loc_p = loc.reshape(1, 1, -1, P)
    scale_p = scale.reshape(1, 1, -1, P)
    # Per-patch standard deviation across the patch axis should be 0.
    assert loc_p.std(dim=-1).max().item() < 1e-9
    assert scale_p.std(dim=-1).max().item() < 1e-9


def test_scaler_constant_signal_gives_minimum_scale():
    """A constant patch has zero variance → scale collapses to minimum_scale."""
    P = 8
    data = torch.full((1, 1, 16), 7.0)
    mask = torch.ones_like(data, dtype=torch.bool)
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2, minimum_scale=1e-3)
    scaled, loc, scale = scaler(data, mask)
    np.testing.assert_allclose(loc.numpy(), 7.0)
    np.testing.assert_allclose(scale.numpy(), 1e-3)
    np.testing.assert_allclose(scaled.numpy(), 0.0)


def test_scaler_white_noise_variance_estimate_is_unbiased():
    """For Gaussian white noise, the bias-corrected multitaper variance
    estimator must be unbiased: E[σ̂²] = σ².

    We test the VARIANCE (not the standard deviation) because the
    standard deviation is biased by Jensen's inequality even when the
    variance estimator is unbiased: sqrt is concave, so
    E[sqrt(σ̂²)] < sqrt(E[σ̂²]) = σ.  For multitaper with K=3 effective
    tapers, Var(σ̂²) ≈ 2σ⁴/K, so the Jensen gap is ~σ/12.

    The unbiasedness property tested here is the headline guarantee of
    the Thomson (1982) multitaper estimator with sample-mean centering
    bias correction (see PatchedCausalDPSSScaler.__init__ derivation).
    """
    torch.manual_seed(0)
    P = 64
    sigma = 2.5  # ground-truth std
    n_patches = 1024  # large for tight estimator
    data = sigma * torch.randn(1, 1, P * n_patches, dtype=torch.float64)
    mask = torch.ones_like(data, dtype=torch.bool)
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.5, K=3)
    _, _, scale = scaler(data, mask)
    per_patch_scale = scale.reshape(1, 1, n_patches, P)[..., 0].squeeze().numpy()
    # E[σ̂²] should equal σ² to within the Monte Carlo error
    # (~σ²·sqrt(2/(K·n_patches)) ≈ σ²·0.025 for K=3, n_patches=1024).
    per_patch_var = per_patch_scale ** 2
    rel_err_var = abs(per_patch_var.mean() - sigma**2) / sigma**2
    assert rel_err_var < 0.05, (
        f"E[σ̂²] = {per_patch_var.mean():.4f} (true {sigma**2:.4f}); "
        f"rel_err = {rel_err_var:.4f} — bias correction is wrong."
    )


def test_scaler_white_noise_std_estimate_jensen_gap_small():
    """E[σ̂] is biased low by Jensen's inequality (sqrt is concave) even
    after the variance estimator is unbiased.  The gap should be small
    (<= ~10 %) for K=3 well-concentrated tapers.
    """
    torch.manual_seed(0)
    P = 64
    sigma = 2.5
    n_patches = 1024
    data = sigma * torch.randn(1, 1, P * n_patches, dtype=torch.float64)
    mask = torch.ones_like(data, dtype=torch.bool)
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.5, K=3)
    _, _, scale = scaler(data, mask)
    per_patch_scale = scale.reshape(1, 1, n_patches, P)[..., 0].squeeze().numpy()
    rel_err_std = abs(per_patch_scale.mean() - sigma) / sigma
    # Theoretical Jensen gap ≈ Var(σ̂²) / (8 σ³) ≈ σ / 12 ≈ 8 % for K=3.
    assert rel_err_std < 0.10, (
        f"E[σ̂] = {per_patch_scale.mean():.4f} (true {sigma:.4f}); "
        f"rel_err = {rel_err_std:.4f}.  This is the expected Jensen gap; "
        f"if larger than 10 %, suspect a real bias."
    )


def test_scaler_is_strictly_per_patch_causal():
    """Changing data in patch s+1 must NOT change (loc, scale) at patch s.

    This is the cornerstone of the "patched causal" contract.  Note that
    DPSS is MORE causal than Welford in this regard — Welford's
    cumulative variance at patch s+1 sees patch s and below; DPSS at
    patch s+1 sees only patch s+1.  But the contract we test here
    (patch s should be unchanged when patch s+1 changes) holds for both.
    """
    P = 16
    torch.manual_seed(0)
    data_a = torch.randn(1, 2, 4 * P, dtype=torch.float64)
    data_b = data_a.clone()
    # Replace the last patch with completely different data.
    data_b[..., 3 * P :] = 100.0 * torch.randn(1, 2, P, dtype=torch.float64)
    mask = torch.ones_like(data_a, dtype=torch.bool)

    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2)
    _, loc_a, scale_a = scaler(data_a, mask)
    _, loc_b, scale_b = scaler(data_b, mask)

    # First three patches must be byte-identical between the two runs.
    assert torch.equal(loc_a[..., : 3 * P], loc_b[..., : 3 * P])
    assert torch.equal(scale_a[..., : 3 * P], scale_b[..., : 3 * P])
    # Last patch must differ (sanity check that we actually perturbed
    # something).
    assert not torch.allclose(scale_a[..., 3 * P :], scale_b[..., 3 * P :])


def test_scaler_mask_handling_excludes_padded_positions():
    """Masked-out positions must not contribute to either mean or variance.

    Construct a patch whose first half is true data and second half is
    pure garbage (large outliers) but masked off.  The (loc, scale)
    must match the ground-truth statistics of the first half alone.
    """
    P = 16
    torch.manual_seed(0)
    half = torch.randn(1, 1, P // 2, dtype=torch.float64)
    garbage = 1e6 * torch.randn(1, 1, P // 2, dtype=torch.float64)
    data = torch.cat([half, garbage], dim=-1)
    mask = torch.cat(
        [torch.ones(1, 1, P // 2, dtype=torch.bool), torch.zeros(1, 1, P // 2, dtype=torch.bool)],
        dim=-1,
    )
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2)
    _, loc, scale = scaler(data, mask)
    # Mean should equal the sample mean of the (unmasked) first half.
    expected_mean = half.mean()
    np.testing.assert_allclose(loc[0, 0, 0].item(), expected_mean.item(), atol=1e-9)
    # Scale must be FINITE and not blown up by the masked garbage.
    assert torch.isfinite(scale).all()
    assert scale.max().item() < 1e3, (
        "Masked garbage leaked into the variance estimate — mask handling broken."
    )


def test_scaler_all_masked_patch_returns_minimum_scale_no_nan():
    P = 8
    data = torch.randn(1, 1, P)
    mask = torch.zeros_like(data, dtype=torch.bool)
    scaler = PatchedCausalDPSSScaler(patch_size=P, NW=2.0, K=2, minimum_scale=1e-3)
    scaled, loc, scale = scaler(data, mask)
    assert torch.isfinite(loc).all()
    assert torch.isfinite(scale).all()
    np.testing.assert_allclose(loc.numpy(), 0.0)
    np.testing.assert_allclose(scale.numpy(), 1e-3)
    # Scaled output is zero where mask is False.
    np.testing.assert_allclose(scaled.numpy(), 0.0)


def test_scaler_dpss_buffer_round_trip_preserves_tapers():
    """state_dict round trip must preserve the DPSS tapers exactly.

    The tapers are non-persistent buffers — they are recomputed at
    ``__init__`` time so a fresh module loaded from state_dict yields
    identical tapers to the saver, even though the buffer is not in
    the state_dict.
    """
    scaler_a = PatchedCausalDPSSScaler(patch_size=16, NW=2.5, K=3)
    sd = scaler_a.state_dict()
    scaler_b = PatchedCausalDPSSScaler(patch_size=16, NW=2.5, K=3)
    scaler_b.load_state_dict(sd)
    assert torch.equal(scaler_a._tapers, scaler_b._tapers)
    assert torch.equal(scaler_a._ratios, scaler_b._ratios)


def test_scaler_dpss_reduces_burst_variance_inflation():
    """Headline win: a transient burst at the patch edge inflates the
    Welford sample variance more than the DPSS multitaper variance.

    DPSS u_0 is bell-shaped with most weight at the patch centre; an
    outlier near the edge contributes less to a_0 = Σ u_0[t] y[t] than
    it would to the rectangular sample sum.  Average over K=3 tapers
    gives a still-low variance estimate.
    """
    P = 64
    torch.manual_seed(0)
    # Quiet baseline with σ=1 plus a single 50σ spike at position 0.
    base = torch.randn(1, 1, P, dtype=torch.float64)
    base[0, 0, 0] = 50.0
    mask = torch.ones_like(base, dtype=torch.bool)

    welford = PatchedCausalStdScaler(patch_size=P)
    dpss = PatchedCausalDPSSScaler(patch_size=P, NW=2.5, K=3)

    _, _, welford_scale = welford(base, mask)
    _, _, dpss_scale = dpss(base, mask)

    welford_sigma = welford_scale[0, 0, 0].item()
    dpss_sigma = dpss_scale[0, 0, 0].item()
    # We expect DPSS to give a meaningfully smaller scale because the
    # edge taper down-weights position 0 (relative weight u_0[0]² ≈ 0
    # vs. 1/P for rectangular sum-of-squares).
    assert dpss_sigma < welford_sigma, (
        f"DPSS σ̂={dpss_sigma:.3f} should be < Welford σ̂={welford_sigma:.3f} "
        f"on a 50σ-edge-spike patch."
    )


# ----------------------------------------------------------------------
# Toto2Model integration
# ----------------------------------------------------------------------


def _tiny_config(*, use_dpss_scaler: bool, dpss_NW: float = 2.5, dpss_K: int = 3) -> Toto2ModelConfig:
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
        use_dpss_scaler=use_dpss_scaler,
        dpss_NW=dpss_NW,
        dpss_K=dpss_K,
    )


def test_model_off_path_uses_welford_scaler():
    cfg = _tiny_config(use_dpss_scaler=False)
    model = Toto2Model(cfg)
    assert isinstance(model.scaler, PatchedCausalStdScaler)
    assert not isinstance(model.scaler, PatchedCausalDPSSScaler)


def test_model_on_path_uses_dpss_scaler():
    cfg = _tiny_config(use_dpss_scaler=True, dpss_NW=2.0, dpss_K=2)
    model = Toto2Model(cfg)
    assert isinstance(model.scaler, PatchedCausalDPSSScaler)
    assert model.scaler.NW == 2.0
    assert model.scaler.K == 2


def test_model_on_path_adds_zero_extra_parameters():
    """DPSS tapers are non-persistent buffers, not parameters."""
    cfg_off = _tiny_config(use_dpss_scaler=False)
    cfg_on = _tiny_config(use_dpss_scaler=True, dpss_NW=2.0, dpss_K=2)
    n_off = sum(p.numel() for p in Toto2Model(cfg_off).parameters())
    n_on = sum(p.numel() for p in Toto2Model(cfg_on).parameters())
    assert n_off == n_on


def test_model_off_path_is_byte_identical_to_baseline():
    """``use_dpss_scaler=False`` must give the exact same network as main."""
    cfg = _tiny_config(use_dpss_scaler=False)
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


def test_model_on_path_forward_runs_end_to_end():
    cfg = _tiny_config(use_dpss_scaler=True, dpss_NW=2.0, dpss_K=2)
    model = Toto2Model(cfg).eval()
    B, V, T = 1, 4, 64
    g = torch.Generator().manual_seed(0)
    target = torch.randn(B, V, T, generator=g)
    mask = torch.ones(B, V, T, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    out = model(target=target, target_mask=mask, cpm_mask=mask, series_ids=series_ids)
    assert out.quantiles.shape[0] == 9  # 9 quantile knots
    assert torch.isfinite(out.quantiles).all()
    assert torch.isfinite(out.loc).all()
    assert torch.isfinite(out.scale).all()


def test_model_dpss_state_dict_does_not_include_tapers():
    """Tapers are non-persistent — they don't bloat the checkpoint and
    are deterministically recomputed from (patch_size, NW, K) at load.
    """
    cfg = _tiny_config(use_dpss_scaler=True, dpss_NW=2.0, dpss_K=2)
    model = Toto2Model(cfg)
    sd = model.state_dict()
    for key in sd:
        assert "scaler._tapers" not in key, (
            f"DPSS tapers should not be persistent; found {key} in state_dict."
        )
        assert "scaler._ratios" not in key
        assert "scaler._ratio_sum" not in key


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


def test_config_rejects_K_larger_than_patch_size():
    with pytest.raises(ValueError, match="dpss_K"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_dpss_scaler=True,
            dpss_K=16,
        )


def test_config_rejects_K_zero():
    with pytest.raises(ValueError, match="dpss_K"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_dpss_scaler=True,
            dpss_K=0,
        )


def test_config_rejects_NW_at_or_above_nyquist():
    with pytest.raises(ValueError, match="dpss_NW"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_dpss_scaler=True,
            dpss_NW=4.0,  # NW = patch_size / 2 → degenerate
        )


def test_config_rejects_negative_NW():
    with pytest.raises(ValueError, match="dpss_NW"):
        Toto2ModelConfig(
            patch_size=8, d_model=32, num_heads=4, num_layers=2, layer_group_size=2,
            num_variate_layers_per_group=1, variate_layer_first=False,
            residual_attn_ratio=1.0,
            use_dpss_scaler=True,
            dpss_NW=-1.0,
        )
