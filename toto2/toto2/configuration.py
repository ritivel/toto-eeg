# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Toto2ModelConfig:
    patch_size: int
    d_model: int
    num_heads: int
    num_layers: int
    layer_group_size: int
    num_variate_layers_per_group: int
    variate_layer_first: bool
    dropout_p: float = 0.0
    norm_eps: float = 5e-5
    attn_bias: bool = False
    mlp_bias: bool = False
    num_output_patches: int = 1
    pre_norm: bool = True
    d_ff: Optional[int] = None
    qk_dim: Optional[int] = None
    v_dim: Optional[int] = None
    num_groups: Optional[int] = None
    heads_per_group: Optional[int] = None
    residual_mult: float = 1.0
    residual_attn_ratio: Optional[float] = None
    qk_norm: bool = True
    norm_include_weight: bool = False
    qk_norm_include_weight: Optional[bool] = None
    per_dim_scale: bool = False
    use_xpos: bool = False
    # ----------------------------------------------------------------
    # exp26 candidate fixes for v3 trunk-collapse (post-exp25 Probe C):
    #
    #   sigma_reparam      — replace every Linear with the σReparam
    #     reparameterization ``W_hat = (γ / σ(W)) · W`` (Zhai et al.,
    #     ICML 2023). γ initialized to σ(W_init) so the first forward
    #     pass is identical to a vanilla Linear; one power-iteration
    #     step in fp32 per training step. Apple's paper shows this
    #     decouples spectral-norm growth from dimensionality and lets
    #     a ViT train without warmup, weight decay, or LayerNorm.
    #
    #   mp_residual        — replace the u-μP τ-rule residual (which
    #     normalizes only gradient magnitudes) with the Karras-style
    #     magnitude-preserving residual ``y = (x + α·δ) / sqrt(1+α²)``,
    #     which normalizes activation magnitudes. Removes the implicit
    #     dependence on residual_mult / residual_attn_ratio, makes the
    #     residual budget explicitly bounded so the FFN can have small
    #     magnitudes intentionally without dragging the trunk RMS to 0.
    #
    #   mp_residual_alpha  — α in the formula above. Default 1.0
    #     (symmetric, matches SDXL). Karras EDM2 uses α≈0.3 for
    #     encoder/decoder blocks, α≈0.5 for the embedding network.
    #
    # Both flags default to False so v3 / Probe C runs are unchanged.
    # See toto2/scripts/configs/pretrain_eeg_from_scratch_v3_probe_*
    # for the concrete probe configurations.
    # ----------------------------------------------------------------
    sigma_reparam: bool = False
    mp_residual: bool = False
    mp_residual_alpha: float = 1.0

    # ----------------------------------------------------------------
    # exp50 — Reference-electrode gauge projection (universal-EEG #2)
    #
    # EEG voltages are physically defined only up to an additive
    # per-timestep constant: any choice of reference electrode shifts
    # every channel by the same scalar.  Helmholtz reciprocity (1853)
    # makes this rigorous: ``v = K · j`` is invariant under
    # ``v -> v + c·1``.  The model SHOULD therefore live on the
    # quotient ``R^C / R·1`` rather than on raw ``R^C``.  The cheapest
    # way to enforce this is to project ``v`` onto the zero-mean
    # subspace at model layer 0:
    #
    #     P  = I - (1/C) 11^T          (Common Average Reference)
    #     v_t -> P · v_t               (per timestep)
    #
    # ``P`` is the orthogonal projector onto ``{v : sum(v) = 0}``; it
    # zeroes out the +c·1 gauge mode and preserves all the other
    # ``C - 1`` directions exactly.  Application is O(C) per timestep
    # and adds zero parameters.  See R2-C / R1-S1 for the
    # affine-invariant-SPD ↔ gauge-quotient ↔ sheaf-cohomology chain.
    #
    #   use_reference_gauge       — turn the projection on / off.
    #     False (default) keeps Toto byte-identical to exp48.
    #
    #   reference_gauge_method    — currently only ``"car"`` (Common
    #     Average Reference).  ``"rest"`` is reserved for a future
    #     experiment that will use the Yao 2001 REST projector
    #     ``T_REST = G · (K^T K + λI)^(-1) · K^T`` derived from the
    #     analytic 3-shell head model and electrode coordinates.
    #     Listed but not implemented — defining the head model + lead
    #     field at training time is materially more invasive and
    #     deserves its own A/B.
    #
    #   gauge_augment_std         — std of an additive per-(batch,
    #     timestep) constant ``c·1`` injected on the input *before*
    #     CAR is applied (training only).  Mathematically a no-op
    #     against pure CAR, but acts as a stress-test of the
    #     projection: any path that bypasses CAR (e.g. a numerical
    #     bug in the mask handling) becomes a sensitivity to the
    #     random offset and is caught by an explosion of
    #     val_observed_frac-weighted CAR residual.  Default 0.0
    #     (off); the smoke YAML exercises it briefly.
    # ----------------------------------------------------------------
    use_reference_gauge: bool = False
    reference_gauge_method: str = "car"
    gauge_augment_std: float = 0.0

    @staticmethod
    def compute_residual_attn_ratio(context_length: int, patch_size: int) -> float:
        """sqrt(S / log(S)) where S = context_length / patch_size.

        Restores attn/MLP variance balance lost by using unscaled F.sdpa
        instead of unit-scaled sdpa.
        """
        s = context_length / patch_size
        return math.sqrt(s / math.log(s))

    def __post_init__(self):
        if self.dropout_p != 0.0:
            raise ValueError("Non-zero dropout_p is a bad choice here: it causes long-term training instability.")
        if self.d_ff is None:
            self.d_ff = (int(4 * self.d_model * 2 / 3) + 7) // 8 * 8
        if self.qk_norm_include_weight is None:
            self.qk_norm_include_weight = self.norm_include_weight
        if self.use_reference_gauge:
            if self.reference_gauge_method not in ("car",):
                raise ValueError(
                    f"reference_gauge_method must be 'car' (Yao 2001 REST is reserved "
                    f"for a future experiment); got {self.reference_gauge_method!r}."
                )
            if self.gauge_augment_std < 0:
                raise ValueError(
                    f"gauge_augment_std must be >= 0; got {self.gauge_augment_std}."
                )
        if self.residual_attn_ratio is None:
            if self.mp_residual:
                # The τ-rule is unused when magnitude-preserving residual
                # is enabled; pin to a harmless 1.0 so downstream lookups
                # don't crash.  All actual residual scaling is done by
                # ``mp_residual_alpha``.
                self.residual_attn_ratio = 1.0
            else:
                raise ValueError(
                    "residual_attn_ratio must be set explicitly. Use "
                    "Toto2ModelConfig.compute_residual_attn_ratio(context_length, patch_size) "
                    "to compute it, or enable mp_residual=True to bypass the τ-rule."
                )
        self.num_groups = self.num_groups or self.num_heads
        self.qk_dim = self.qk_dim or self.d_model // self.num_heads
        self.v_dim = self.v_dim or self.qk_dim
        self.heads_per_group = self.num_heads // self.num_groups

        assert self.num_layers % self.layer_group_size == 0, (
            f"num_layers must be divisible by layer_group_size"
            f"got num_layers={self.num_layers} and layer_group_size={self.layer_group_size}"
        )
        assert self.num_heads > 0 and self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert (self.num_heads % self.num_groups == 0) and (self.num_heads >= self.num_groups), (
            f"num_heads ({self.num_heads}) must be divisible by num_groups ({self.num_groups}) and greater than or equal to num_groups ({self.num_groups})"
        )

    # @property
    # def heads_per_group(self) -> int:
    #     return self.num_heads // self.num_groups


@dataclass
class Toto2GluonTSModelConfig:
    prediction_length: int
    context_length: int
    target_dim: int
    past_feat_dynamic_real_dim: int = 0
    feat_dynamic_real_dim: int = 0
    decode_block_size: Optional[int] = None
    has_missing_values: bool = True
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
