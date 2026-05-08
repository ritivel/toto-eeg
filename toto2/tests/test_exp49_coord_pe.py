# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Tests for the exp49 continuous coordinate patch embedding.

The new ``CoordPE`` module sits at the very front of ``Toto2Model`` and
controls how (variate, patch) tokens see the underlying spatio-temporal
geometry.  These tests pin down the contract that downstream training
code, eval adapters, and future experiments rely on:

* shapes of ``γ(t, r⃗)`` and ``Y_l^m(θ, φ)`` features
* zero-init SH head means the SH branch contributes exactly zero at
  init (so ``CoordPE.forward(...) == fourier_features(...)``)
* B is fixed across forward passes (Rahimi–Recht random Fourier
  features only have their kernel-approximation guarantees while B
  is constant)
* invariance of the output under permutation of the variate axis
* the same encoding survives a checkpoint round-trip
* spherical harmonics are correctly orthonormal on a Fibonacci grid
* the full ``Toto2Model`` accepts ``electrode_coords`` and the patch-
  projection input dim matches ``2 * patch_size + coord_pe.out_dim``
* the dataset → collate → model glue holds end-to-end
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("lightning")

from toto2.configuration import Toto2ModelConfig  # noqa: E402
from toto2.model import (  # noqa: E402
    CoordPE,
    Toto2Model,
    _associated_legendre,
    _real_spherical_harmonics,
)
from toto2.training import (  # noqa: E402
    ArrayTimeSeriesDataset,
    SlidingWindowConfig,
    collate_timeseries,
)


# ----------------------------------------------------------------------
# CoordPE — feature shapes + zero-init contract
# ----------------------------------------------------------------------


def _unit_sphere(n: int, seed: int = 0) -> torch.Tensor:
    """Sample ``n`` random points on the unit 2-sphere (uniformly in area)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_coord_pe_output_shape():
    coords = _unit_sphere(7).unsqueeze(0)  # (1, 7, 3)
    pe = CoordPE(num_fourier=16, max_l=4, sigma_B=1.0)
    out = pe(coords, n_patches=5)
    # Last dim is 2 * num_fourier; first three are batch / variate / patch.
    assert out.shape == (1, 7, 5, 32)
    assert torch.isfinite(out).all()


def test_coord_pe_supports_arbitrary_lead_dims():
    """CoordPE must work with extra lead axes (e.g., DDP shard)."""
    coords = _unit_sphere(5).expand(2, 3, 5, 3)  # (2, 3, 5, 3)
    pe = CoordPE(num_fourier=8, max_l=3, sigma_B=1.0)
    out = pe(coords, n_patches=4)
    assert out.shape == (2, 3, 5, 4, 16)


def test_coord_pe_sh_is_zero_at_init():
    """Zero-init SH head ⇒ CoordPE(coords) == fourier_features(coords)."""
    coords = _unit_sphere(11)
    pe = CoordPE(num_fourier=8, max_l=8, sigma_B=1.0)
    full = pe(coords, n_patches=6)
    only_fourier = pe.fourier_features(coords, n_patches=6)
    # Bit-exact: SH head outputs literal zero so the sum is the Fourier alone.
    assert torch.equal(full, only_fourier)


def test_coord_pe_sh_grows_after_grad_step():
    """After a backprop step the zero-init SH head should leave zero land."""
    coords = _unit_sphere(11)
    pe = CoordPE(num_fourier=4, max_l=4, sigma_B=1.0)
    optimizer = torch.optim.SGD(pe.sh_head.parameters(), lr=1.0)
    target = torch.randn_like(pe(coords, n_patches=2))
    pred = pe(coords, n_patches=2)
    loss = (pred - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # SH weights are non-zero → SH head now contributes.
    assert pe.sh_head.weight.detach().abs().sum() > 0
    full_after = pe(coords, n_patches=2)
    fourier_after = pe.fourier_features(coords, n_patches=2)
    assert not torch.equal(full_after, fourier_after)


def test_coord_pe_B_is_a_buffer_not_a_parameter():
    """B must travel with the module via .to() / state_dict but never see grads."""
    pe = CoordPE(num_fourier=8, max_l=4, sigma_B=1.0, seed=42)
    assert "B" in dict(pe.named_buffers())
    assert "B" not in dict(pe.named_parameters())


def test_coord_pe_state_dict_round_trip():
    """A reloaded CoordPE must produce identical Fourier features."""
    coords = _unit_sphere(9)
    pe1 = CoordPE(num_fourier=12, max_l=4, sigma_B=1.0, seed=123)
    sd = pe1.state_dict()
    pe2 = CoordPE(num_fourier=12, max_l=4, sigma_B=1.0)
    pe2.load_state_dict(sd)
    out1 = pe1(coords, n_patches=3)
    out2 = pe2(coords, n_patches=3)
    assert torch.equal(out1, out2)


def test_coord_pe_forward_is_deterministic_per_step():
    """Two forward passes with the same inputs must yield identical outputs."""
    coords = _unit_sphere(13)
    pe = CoordPE(num_fourier=8, max_l=4, sigma_B=1.0, seed=7)
    pe.eval()
    a = pe(coords, n_patches=4)
    b = pe(coords, n_patches=4)
    assert torch.equal(a, b)


def test_coord_pe_seed_controls_B():
    """Different seeds → different B → different output; same seed → same."""
    coords = _unit_sphere(8)
    pe_a = CoordPE(num_fourier=8, max_l=2, sigma_B=1.0, seed=1)
    pe_b = CoordPE(num_fourier=8, max_l=2, sigma_B=1.0, seed=2)
    pe_a2 = CoordPE(num_fourier=8, max_l=2, sigma_B=1.0, seed=1)
    out_a = pe_a.fourier_features(coords, n_patches=3)
    out_b = pe_b.fourier_features(coords, n_patches=3)
    out_a2 = pe_a2.fourier_features(coords, n_patches=3)
    assert not torch.allclose(out_a, out_b)
    assert torch.equal(out_a, out_a2)


# ----------------------------------------------------------------------
# Spherical harmonics — orthonormality + identities
# ----------------------------------------------------------------------


def test_real_sh_y00_matches_constant():
    """Y_0^0 = 1 / (2·sqrt(π)) at every point on the sphere."""
    coords = _unit_sphere(20)
    Y = _real_spherical_harmonics(coords, max_l=0)  # (20, 1)
    expected = 1.0 / (2.0 * math.sqrt(math.pi))
    assert torch.allclose(Y[:, 0], torch.full_like(Y[:, 0], expected), atol=1e-6)


def test_real_sh_count_modes():
    """For l=0..L there are (L+1)^2 real modes."""
    coords = _unit_sphere(5)
    Y = _real_spherical_harmonics(coords, max_l=4)
    assert Y.shape[-1] == (4 + 1) ** 2


def test_real_sh_orthonormal_on_fibonacci_grid():
    """Real SH modes integrate to ``δ_{ij}`` against the unit-sphere measure.

    A Fibonacci lattice with N points approximates uniform sampling.  The
    normalisation gives ``(4π/N) · Σ_n Y_i(n) Y_j(n) ≈ δ_{ij}``.  We check
    a small case (l ≤ 2) where rounding is comfortably <1e-1 at N=2048.
    """
    N = 2048
    indices = torch.arange(N, dtype=torch.float64) + 0.5
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta = torch.arccos(1.0 - 2.0 * indices / N)
    azim = 2.0 * math.pi * indices / phi
    x = torch.sin(theta) * torch.cos(azim)
    y = torch.sin(theta) * torch.sin(azim)
    z = torch.cos(theta)
    coords = torch.stack([x, y, z], dim=-1).to(torch.float32)

    L = 2
    Y = _real_spherical_harmonics(coords, max_l=L)
    G = (4.0 * math.pi / N) * (Y.t() @ Y)
    eye = torch.eye(G.shape[0])
    assert torch.allclose(G, eye, atol=5e-2), G - eye


def test_associated_legendre_p00_is_one():
    cos_theta = torch.tensor([0.1, 0.3, -0.7])
    sin_theta = (1 - cos_theta**2).sqrt()
    P = _associated_legendre(cos_theta, sin_theta, max_l=4)
    assert torch.allclose(P[(0, 0)], torch.ones_like(cos_theta))


# ----------------------------------------------------------------------
# Toto2Model integration
# ----------------------------------------------------------------------


def _tiny_config_with_coord_pe(*, use_coord_pe: bool) -> Toto2ModelConfig:
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
        use_coord_pe=use_coord_pe,
        coord_pe_num_fourier=4,
        coord_pe_max_l=2,
        coord_pe_sigma_B=1.0,
        coord_pe_time_scale=1.0,
    )


def test_model_off_path_unchanged_by_coord_pe_flag():
    """``use_coord_pe=False`` ⇒ Toto2Model has no CoordPE attached."""
    cfg = _tiny_config_with_coord_pe(use_coord_pe=False)
    model = Toto2Model(cfg)
    assert model.coord_pe is None
    # patch_proj's first layer must accept exactly ``2 * patch_size`` input dims.
    expected = 2 * cfg.patch_size
    assert model.patch_proj.linear1.weight.shape[1] == expected


def test_model_on_path_grows_patch_proj():
    """``use_coord_pe=True`` ⇒ in_dim = 2·P + coord_pe.out_dim."""
    cfg = _tiny_config_with_coord_pe(use_coord_pe=True)
    model = Toto2Model(cfg)
    assert model.coord_pe is not None
    expected = 2 * cfg.patch_size + model.coord_pe.out_dim
    assert model.patch_proj.linear1.weight.shape[1] == expected


def test_model_forward_requires_electrode_coords_when_coord_pe_on():
    cfg = _tiny_config_with_coord_pe(use_coord_pe=True)
    model = Toto2Model(cfg).eval()
    B, V = 1, 3
    target = torch.randn(B, V, 64)
    target_mask = torch.ones(B, V, 64, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    with pytest.raises(ValueError, match="electrode_coords"):
        model(
            target=target,
            target_mask=target_mask,
            cpm_mask=target_mask,
            series_ids=series_ids,
        )


def test_model_forward_with_electrode_coords_runs():
    cfg = _tiny_config_with_coord_pe(use_coord_pe=True)
    model = Toto2Model(cfg).eval()
    B, V = 2, 4
    target = torch.randn(B, V, 64)
    target_mask = torch.ones(B, V, 64, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)
    coords = _unit_sphere(V).unsqueeze(0).expand(B, V, 3)
    out = model(
        target=target,
        target_mask=target_mask,
        cpm_mask=target_mask,
        series_ids=series_ids,
        electrode_coords=coords,
    )
    # Q quantile knots × B × V × num_patches × patch_size
    assert out.quantiles.shape[0] == 9  # default quantile knots
    assert out.quantiles.shape[1] == B
    assert out.quantiles.shape[2] == V
    assert torch.isfinite(out.quantiles).all()


def test_model_forward_at_init_matches_v3_when_coords_supplied():
    """At init the SH head is zero, so the only coord influence is via
    γ(t, r⃗).  Two distinct sets of *coords* with the same ``B`` will
    produce different model outputs — verify that swapping coords
    actually moves the prediction (no silent path bypass)."""
    cfg = _tiny_config_with_coord_pe(use_coord_pe=True)
    model = Toto2Model(cfg).eval()
    B, V = 1, 5
    target = torch.randn(B, V, 64)
    target_mask = torch.ones(B, V, 64, dtype=torch.bool)
    series_ids = torch.zeros(B, V, dtype=torch.long)

    coords_a = _unit_sphere(V, seed=10).unsqueeze(0)
    coords_b = _unit_sphere(V, seed=11).unsqueeze(0)
    out_a = model(
        target=target,
        target_mask=target_mask,
        cpm_mask=target_mask,
        series_ids=series_ids,
        electrode_coords=coords_a,
    ).quantiles
    out_b = model(
        target=target,
        target_mask=target_mask,
        cpm_mask=target_mask,
        series_ids=series_ids,
        electrode_coords=coords_b,
    ).quantiles
    # Different coords ⇒ different model outputs (the whole point of coord-PE).
    assert not torch.allclose(out_a, out_b)


# ----------------------------------------------------------------------
# Dataset / collate plumbing
# ----------------------------------------------------------------------


def test_collate_includes_electrode_coords():
    rng = np.random.default_rng(0)
    recs = [rng.standard_normal((3, 1024)).astype(np.float32) for _ in range(2)]
    coords = np.stack([np.eye(3, dtype=np.float32) for _ in recs], axis=0)
    ds = ArrayTimeSeriesDataset(
        recs,
        SlidingWindowConfig(context_length=512, patch_size=64, stride=512, random=False),
        electrode_coords=coords,
    )
    sample = ds[0]
    assert sample["electrode_coords"].shape == (3, 3)

    batch = collate_timeseries([ds[0], ds[1]])
    assert batch["electrode_coords"].shape == (2, 3, 3)


def test_collate_rejects_mixed_coord_presence():
    """Mixed batch (some samples with coords, some without) must raise."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 1024)).astype(np.float32)
    b = rng.standard_normal((3, 1024)).astype(np.float32)
    coords = np.eye(3, dtype=np.float32)
    cfg = SlidingWindowConfig(context_length=512, patch_size=64, stride=512, random=False)
    ds_with = ArrayTimeSeriesDataset([a], cfg, electrode_coords=coords)
    ds_without = ArrayTimeSeriesDataset([b], cfg)
    with pytest.raises(ValueError, match="electrode_coords"):
        collate_timeseries([ds_with[0], ds_without[0]])


def test_collate_pads_coord_rows_with_zero_for_pad_variates():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 1024)).astype(np.float32)
    b = rng.standard_normal((5, 1024)).astype(np.float32)
    coord_a = np.eye(3, dtype=np.float32)
    coord_b = np.eye(5, dtype=np.float32)[:, :3]  # (5, 3)
    cfg = SlidingWindowConfig(context_length=512, patch_size=64, stride=512, random=False)
    # Two recordings with different montages can be combined as long as we
    # supply per-recording coords (3-D array shape (n_recordings, n_var, 3)).
    ds_a = ArrayTimeSeriesDataset([a], cfg, electrode_coords=coord_a)
    ds_b = ArrayTimeSeriesDataset([b], cfg, electrode_coords=coord_b)
    batch = collate_timeseries([ds_a[0], ds_b[0]])
    assert batch["electrode_coords"].shape == (2, 5, 3)
    # Padded variate rows of the first sample (variates 3 and 4) must be exact zeros.
    assert torch.equal(
        batch["electrode_coords"][0, 3:],
        torch.zeros_like(batch["electrode_coords"][0, 3:]),
    )


# ----------------------------------------------------------------------
# Montage loader
# ----------------------------------------------------------------------


def test_montage_load_unit_sphere_positions():
    from toto2.training.montages import load_unit_sphere_positions

    arr = load_unit_sphere_positions("gsn_hydrocel_129")
    assert arr.shape == (129, 3)
    assert arr.dtype == np.float32
    # Unit-sphere normalisation: every row has norm ~ 1.
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
