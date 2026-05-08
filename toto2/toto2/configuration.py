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
    # exp51 — DPSS-tapered causal scaler (universal-EEG #3)
    #
    # The default ``PatchedCausalStdScaler`` uses a Welford-style
    # cumulative sample mean / variance over the entire causal history
    # at each patch boundary.  Sample variance with rectangular
    # weighting is biased upward by transient bursts (alpha bursts,
    # sleep spindles, eye-blink artefacts, electrode pops) — a single
    # outlier sample contributes its full squared deviation to the
    # scale, which then divides the rest of the patch and squashes the
    # whole window's amplitude.
    #
    # exp51 replaces the variance estimator with a per-patch Thomson
    # multitaper using K leading discrete prolate spheroidal sequences
    # (DPSS / Slepian) of length ``patch_size`` and time-bandwidth
    # ``dpss_NW``.  DPSS sequences are the unique signals maximally
    # concentrated in both time [0, P) and frequency [-W, W] under the
    # Heisenberg-Donoho-Stark uncertainty bound (Slepian 1978), and the
    # multitaper PSD they form is minimum-variance among all linear
    # estimators (Thomson 1982; Percival & Walden 1993).  In practice
    # the tapers down-weight the patch edges (where bursts and seam
    # transients live) and average K independent low-bias estimates,
    # giving a substantial reduction in scaler bias on EEG.
    #
    #   use_dpss_scaler   — turn the multitaper variance on / off.
    #     False (default) keeps Toto byte-identical to exp48 / main.
    #
    #   dpss_NW           — time-bandwidth product.  Standard EEG
    #     practice uses NW = 2.5 (Babadi & Brown, IEEE TBME 2014):
    #     low enough to keep 2NW = 5 well-concentrated tapers, high
    #     enough that the half-bandwidth W = NW/P stays comfortably
    #     above the dominant 1/f slope.  For P=64 @ 500Hz, W=2.5/64
    #     corresponds to ~19.5Hz of frequency smoothing — covers the
    #     alpha and beta bands without bleeding into gamma.
    #
    #   dpss_K            — number of leading DPSS tapers used.  K=3
    #     is the convention for NW=2.5 (the K = 2NW - 1 rule of thumb
    #     for "well-concentrated" tapers, eigenvalue > 0.99).  K=4
    #     drops eigenvalue 4 to ~0.95 and is a knob for variance /
    #     bias tradeoff; K=2 minimises variance but loses spectral
    #     coverage.  Validation rejects ``K > P``.
    #
    # The mean estimator stays the simple per-patch sample mean — DPSS
    # has nothing to add for first moments and we retain causality of
    # the per-patch broadcast contract (loc/scale at any timestep
    # within patch s uses only patch s's data).  The Welford-style
    # cumulative mean of the default scaler is replaced with a per-
    # patch sample mean to match: this is mathematically the
    # variance-driven "scaler bias on bursty signals" failure mode
    # exp51 is designed to address.
    # ----------------------------------------------------------------
    use_dpss_scaler: bool = False
    dpss_NW: float = 2.5
    dpss_K: int = 3

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
        if self.use_dpss_scaler:
            if self.dpss_K < 1:
                raise ValueError(f"dpss_K must be >= 1; got {self.dpss_K}.")
            if self.dpss_K > self.patch_size:
                raise ValueError(
                    f"dpss_K ({self.dpss_K}) must not exceed patch_size "
                    f"({self.patch_size}); each DPSS taper has length patch_size "
                    f"and only patch_size orthogonal sequences exist."
                )
            if self.dpss_NW <= 0:
                raise ValueError(f"dpss_NW must be > 0; got {self.dpss_NW}.")
            if self.dpss_NW >= self.patch_size / 2:
                raise ValueError(
                    f"dpss_NW ({self.dpss_NW}) must be < patch_size/2 "
                    f"({self.patch_size / 2}); the half-bandwidth W = NW/P would "
                    f"otherwise reach the Nyquist frequency and DPSS becomes degenerate."
                )
            if self.dpss_K > 2 * self.dpss_NW:
                # Soft warning: only the first 2*NW tapers have eigenvalue >> 0.5.
                # Higher-K tapers have most of their energy outside [-W, W] and
                # add noise instead of signal to the multitaper estimate.  Allowed
                # but explicitly documented as a bias / variance knob.
                pass
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
