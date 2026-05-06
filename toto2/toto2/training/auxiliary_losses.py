# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Auxiliary supervision objectives for Toto 2.0 EEG pre-training.

These losses are used in the **exp27 supervision-augmentation suite** (probes
A-F).  They are designed to give the FFN a real job: the v3 trunk-collapse
post-mortem (exp22-26) showed that the pinball-on-asinh-z-score-quantile target
biases the model toward a smart causal-scaler + identity-residual + output-head
decoder where the FFN is driven to fp32 underflow.  Each auxiliary head adds a
gradient signal that the trunk can only minimise by encoding genuine
spatio-temporal structure, not by collapsing the residual stream.

Six probes (one per exp27..32):

* **A. LeJEPA latent-prediction head** (:class:`JEPAHead`,
  :func:`sigreg_loss`).  EMA target encoder + predictor + SIGReg, following
  Balestriero & LeCun (arXiv:2511.08544).
* **B. AAMP amplitude-aware mask** (:func:`amplitude_aware_cpm_mask`).
  Drop top-K% amplitude patches from the scaler/input, NeurIPT
  (arXiv:2510.16548) recipe.
* **C. PARS pairwise relative shift** (:class:`PARSHead`).  Cross-attention
  head predicting the antisymmetric matrix of patch-pair shifts, Apple
  (arXiv:2511.11940).
* **D. Phase-aware Fourier loss** (:func:`phase_aware_fourier_loss`).
  Single-resolution FFT loss on the median-knot prediction, NeuroRVQ
  (arXiv:2510.13068).
* **E. Multi-resolution STFT** (:class:`MultiResolutionSTFTAuxLoss`).
  TTS-style multi-resolution log-magnitude + spectral-convergence loss,
  Steinmetz auraloss / PhysioWave style.
* **F. Online denoised target** (:func:`online_denoise_target`).
  Replace the pinball target with a band-limited, sigma-clipped, detrended
  version so the trunk has to denoise as it predicts, EEG-X-style
  (arXiv:2511.08861).

Each module is self-contained and depends only on torch + (for E) auraloss.
The lightning_module wires them in conditionally based on yaml config flags
under ``training.auxiliary``.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import dd_unit_scaling as uu


# =====================================================================
# B. AAMP: Amplitude-aware patch masking (NeurIPT, arXiv:2510.16548)
# =====================================================================


@torch.no_grad()
def amplitude_aware_cpm_mask(
    target: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio_range: Tuple[float, float] = (0.20, 0.40),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Drop top-K% high-amplitude patches from the causal-patch mask.

    Random-uniform masking (the implicit default when ``cpm_mask = target_mask``)
    tends to degenerate to local interpolation: the model just averages the
    neighbours of an unobserved point.  NeurIPT shows that masking by amplitude
    instead — preferentially hiding the high-energy spikes/bursts/spindles — gives
    +5.25% on TUEV BAC because the model is forced to extrapolate the pattern
    of brief high-amplitude events from the long-running low-amplitude baseline.

    Parameters
    ----------
    target
        ``(B, V, T)`` raw values (pre-scaler).  ``T`` must be a multiple
        of ``patch_size``.
    base_mask
        ``(B, V, T)`` boolean observation mask (True = observed).
    patch_size
        Number of timesteps per patch (e.g., 64).
    mask_ratio_range
        ``(low, high)`` inclusive range from which a per-batch
        amplitude-mask ratio is drawn uniformly at every step.  Following
        NeurIPT's ``[20, 35, 50]`` we pick a slightly tighter ``[20, 40]``.
    generator
        Optional ``torch.Generator`` for reproducibility.

    Returns
    -------
    torch.Tensor
        ``(B, V, T)`` boolean ``cpm_mask`` to feed the model: True at
        positions the scaler / input embedding may use, False at
        positions that are dropped (forced to zero in the input).  This
        is **not** the loss mask — pinball is still computed everywhere
        ``base_mask & target_mask`` is True.
    """
    B, V, T = target.shape
    if T % patch_size != 0:
        raise ValueError(
            f"target last-dim {T} must be a multiple of patch_size {patch_size}"
        )
    S = T // patch_size

    # Patch amplitude = max-abs over the patch (robust to single-sample spikes).
    amp_per_patch = rearrange(target.abs(), "b v (s p) -> b v s p", p=patch_size).amax(dim=-1)
    # (B, V, S)

    # Sample a fresh ratio per batch.
    if generator is not None:
        u = torch.rand((), generator=generator, device=target.device)
    else:
        u = torch.rand((), device=target.device)
    lo, hi = mask_ratio_range
    ratio = lo + (hi - lo) * u  # scalar

    # Per-(B,V) threshold at the (1 - ratio) quantile.
    q = (1.0 - ratio).clamp(0.0, 1.0)
    # `quantile` on bf16 is rejected on some kernels; cast to fp32 for the
    # quantile calculation, then back.
    threshold = torch.quantile(amp_per_patch.float(), q, dim=-1, keepdim=True).to(amp_per_patch.dtype)
    high_amp = amp_per_patch > threshold  # (B, V, S)

    # Patches to keep in the cpm_mask: NOT high amplitude.
    keep_per_patch = ~high_amp  # (B, V, S)
    keep_full = repeat(keep_per_patch, "b v s -> b v (s p)", p=patch_size)
    return base_mask & keep_full


# =====================================================================
# F. Online denoised target (EEG-X, arXiv:2511.08861)
# =====================================================================


@torch.no_grad()
def online_denoise_target(
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sfreq: float = 500.0,
    low_hz: float = 1.0,
    high_hz: float = 50.0,
    clip_sigma: float = 6.0,
    detrend: bool = True,
) -> torch.Tensor:
    """Per-recording online denoise to produce a cleaner pinball target.

    EEG-X observed that asking the model to reconstruct the *raw* signal
    biases the encoder toward high-variance artifact (eye blinks ~ 100 µV,
    EMG ~ 50 µV, motion ~ 1 mV) — features the trunk would otherwise
    filter out.  The fix is to swap the reconstruction target for an
    artifact-removed version while keeping the input raw, so the model
    has to denoise as it predicts.  Their offline ICA / ASR pipeline
    is too expensive to run inline; this function approximates it with
    three cheap operations that capture most of the benefit:

    1. **Causal high-pass + soft anti-alias** — a one-pole IIR filter
       at ``low_hz`` removes DC drift / sweat artefacts; a one-pole
       low-pass at ``high_hz`` rejects line noise + high-freq EMG.
       Done causally so we don't leak future statistics.
    2. **Linear detrend** per (B, V) window subtracts the linear fit,
       removing the slow component that the high-pass missed.
    3. **Sigma clip** clamps each (B, V, T) value to
       ``[-clip_sigma * std, clip_sigma * std]`` where ``std`` is the
       per-(B, V) std after the previous two steps; this nukes
       impulse artefacts ASR would have removed.

    The function operates on **the target only**.  The model input is
    unchanged.  Because the same ``sfreq`` is applied to every recording
    in the HBN-EEG corpus, we hard-code 500 Hz; for other corpora the
    config block carries ``data.eeg.sfreq`` which the lightning module
    forwards here.

    Parameters
    ----------
    target
        ``(B, V, T)`` raw target values.
    mask
        ``(B, V, T)`` observation mask (True = observed).  Padding
        positions (``False``) are zeroed before filtering and re-zeroed
        afterwards so they don't leak through the IIR.
    sfreq
        Sampling rate (Hz).
    low_hz, high_hz
        Pass-band of the bandpass.  ``1-50 Hz`` is a standard EEG
        cleaning window.
    clip_sigma
        Multiple of (post-bandpass) per-variate std at which to clip.

    Returns
    -------
    torch.Tensor
        Denoised ``(B, V, T)`` target.
    """
    if target.dim() != 3:
        raise ValueError(f"online_denoise_target expects (B, V, T); got shape {tuple(target.shape)}")

    # Promote to fp32 for the IIR (bf16 has too little precision for the
    # geometric series of a one-pole filter at 1 Hz / 500 Hz = 0.002).
    x = target.float()
    m = mask.float()
    x = x * m  # zero out padding

    # --------- causal high-pass at low_hz ---------
    #
    # One-pole HPF: y[t] = α (y[t-1] + x[t] - x[t-1])
    # with α = exp(-2π · low_hz / sfreq).  This is differentiable (no autograd
    # needed since we wrap in @no_grad above) and runs in O(T) sequentially
    # — fine for T=4096 at our batch sizes.
    if low_hz > 0:
        alpha_hp = math.exp(-2.0 * math.pi * float(low_hz) / float(sfreq))
        y_hp = torch.zeros_like(x)
        x_prev = torch.zeros_like(x[..., 0])
        y_prev = torch.zeros_like(x[..., 0])
        for t in range(x.shape[-1]):
            x_t = x[..., t]
            y_t = alpha_hp * (y_prev + x_t - x_prev)
            y_hp[..., t] = y_t
            x_prev = x_t
            y_prev = y_t
        x = y_hp

    # --------- causal low-pass at high_hz ---------
    #
    # One-pole LPF: y[t] = β y[t-1] + (1-β) x[t]   with β = exp(-2π · high_hz / sfreq).
    # At high_hz=50 / sfreq=500 we get β≈0.533 — sharp enough to attenuate
    # 60 Hz line noise by ~10 dB without ringing.
    if high_hz > 0:
        beta_lp = math.exp(-2.0 * math.pi * float(high_hz) / float(sfreq))
        y_lp = torch.zeros_like(x)
        y_prev = torch.zeros_like(x[..., 0])
        for t in range(x.shape[-1]):
            x_t = x[..., t]
            y_t = beta_lp * y_prev + (1.0 - beta_lp) * x_t
            y_lp[..., t] = y_t
            y_prev = y_t
        x = y_lp

    # --------- linear detrend ---------
    if detrend:
        T = x.shape[-1]
        t_axis = torch.arange(T, device=x.device, dtype=x.dtype)
        # Solve y = a + b*t via closed-form least squares per (B, V).
        n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        sum_t = (t_axis * m).sum(dim=-1, keepdim=True)
        sum_y = (x * m).sum(dim=-1, keepdim=True)
        sum_tt = ((t_axis * t_axis) * m).sum(dim=-1, keepdim=True)
        sum_ty = (t_axis * x * m).sum(dim=-1, keepdim=True)
        denom = (n * sum_tt - sum_t * sum_t).clamp_min(1e-12)
        b = (n * sum_ty - sum_t * sum_y) / denom
        a = (sum_y - b * sum_t) / n
        x = x - (a + b * t_axis)

    # --------- sigma clip ---------
    if clip_sigma > 0:
        sigma = ((x * m).pow(2).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)).sqrt()
        sigma = sigma.clamp_min(1e-6)
        x = x.clamp(min=-clip_sigma * sigma, max=clip_sigma * sigma)

    # Re-zero padded positions (in case the IIR / detrend leaked anything).
    x = x * m
    return x.to(target.dtype)


# =====================================================================
# A. SIGReg + JEPA latent-prediction head (LeJEPA, arXiv:2511.08544)
# =====================================================================
#
# Vendored from the official LeJEPA codebase (rbalestr-lab/lejepa, CC BY-NC 4.0)
# so we do not depend on the optional ``stable_pretraining`` install path.
# The original is ~50 LOC; we keep it inline.


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_slices: int = 1024,
    n_quad_points: int = 17,
    sigma: float = 1.0,
    global_step: int = 0,
    max_samples: Optional[int] = 4096,
) -> torch.Tensor:
    """SIGReg with the Epps-Pulley characteristic-function statistic.

    See Algorithm 1 of LeJEPA.  The implementation slices each (N, K)
    embedding tensor along ``num_slices`` random unit directions and
    measures how much the empirical 1-D characteristic function deviates
    from the standard-Gaussian one.  Because the integrand is symmetric
    and bounded, a 17-knot trapezoidal quadrature is enough (Figure 20
    in the paper).  Linear in N and K, DDP-friendly, hyperparameter-free.

    Parameters
    ----------
    embeddings
        ``(N, K)`` tensor — typically the trunk output flattened over
        all batch / variate / time axes.
    num_slices
        Number of random directions.  1024 is the LeJEPA default;
        for our smaller K=384 trunk a smaller value works too.
    n_quad_points
        Number of quadrature knots in [0, sigma].  17 is the paper's
        default and the figure-20 sweet spot.
    sigma
        Bandwidth of the Epps-Pulley kernel.  1.0 matches the paper.
    global_step
        Used to seed slice sampling so different DDP ranks pick the
        same directions (sync sampling).  Using the trainer
        ``global_step`` makes the directions evolve over training so
        the regulariser does not memorise a fixed projection.
    max_samples
        If ``N > max_samples`` we randomly subsample to keep memory
        bounded.  The intermediate ``(n_quad, N, num_slices)`` tensor
        is the dominant cost: 17 * 99K * 1024 * 4 bytes ≈ 7GB on
        Toto-2's full (B=12, V=129, S=64) flattened context.  4096
        samples gives ~70MB and is more than enough for the
        Cramér-Wold sketch (LeJEPA paper §4.3).  Set to ``None`` to
        disable subsampling.

    Returns
    -------
    torch.Tensor
        Scalar SIGReg loss.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"SIGReg expects (N, K); got {tuple(embeddings.shape)}")
    N_full, K = embeddings.shape

    # Subsample for memory.  We do this *with* gradient (it's just an index
    # gather) so SIGReg still trains the encoder.
    if max_samples is not None and N_full > int(max_samples):
        g_sub = torch.Generator(device="cpu")
        g_sub.manual_seed(int(global_step) * 31 + 17)  # different seed than slice sampling
        idx = torch.randperm(N_full, generator=g_sub)[: int(max_samples)].to(embeddings.device)
        embeddings = embeddings.index_select(0, idx)
    N, K = embeddings.shape
    device = embeddings.device

    g = torch.Generator(device="cpu")  # CPU generator works on every backend
    g.manual_seed(int(global_step))
    A = torch.randn(K, num_slices, generator=g).to(device=device, dtype=embeddings.dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    # Project: (N, num_slices)
    proj = embeddings @ A

    # Standardise per slice to match the Epps-Pulley assumption that
    # the target is N(0, 1).  (LeJEPA does this implicitly via the
    # SIGReg regulariser pushing var=1 — but we also subtract the mean
    # to avoid the bias term blowing the loss before std stabilises.)
    proj = proj - proj.mean(dim=0, keepdim=True)
    std = proj.std(dim=0, keepdim=True).clamp_min(1e-6)
    proj = proj / std

    # Trapezoidal quadrature of |φ̂(t) - φ_target(t)|² · w(t) over t∈[0, sigma].
    #
    # For a standard Gaussian target, φ_target(t) = exp(-σ² t² / 2).
    # The empirical char fn is (1/N) Σ_n exp(i t z_n).  Squared modulus:
    #   |φ̂|² = (1/N²) [(Σ cos(t z))² + (Σ sin(t z))²]
    t = torch.linspace(1e-6, sigma, n_quad_points, device=device, dtype=embeddings.dtype)
    # broadcast to (n_quad_points, N, num_slices)
    tz = t[:, None, None] * proj[None, :, :]
    cos_sum = tz.cos().sum(dim=1)  # (n_quad_points, num_slices)
    sin_sum = tz.sin().sum(dim=1)
    abs_phi_hat_sq = (cos_sum * cos_sum + sin_sum * sin_sum) / (N * N)

    phi_target = torch.exp(-0.5 * (sigma * sigma) * (t * t))[:, None]  # (n_quad_points, 1)
    abs_phi_target_sq = phi_target * phi_target

    # Real part of φ̂ for the cross term:
    re_phi_hat = cos_sum / N

    integrand = (abs_phi_hat_sq - 2.0 * phi_target * re_phi_hat + abs_phi_target_sq)

    # Trapezoidal rule weights — endpoints get weight Δt/2, interior gets Δt.
    # For uniform spacing this is just (Δt) · sum(integrand) - 0.5 Δt (endpoints).
    dt = (sigma - 1e-6) / (n_quad_points - 1)
    weights = torch.full((n_quad_points,), dt, device=device, dtype=embeddings.dtype)
    weights[0] = 0.5 * dt
    weights[-1] = 0.5 * dt
    integral = (weights[:, None] * integrand).sum(dim=0)  # (num_slices,)

    return integral.mean()


class JEPAHead(nn.Module):
    """LeJEPA-style latent-prediction head with EMA target encoder.

    Built on top of :class:`toto2.model.Toto2Model`.  The context encoder
    is the **shared** Toto2 trunk (we don't duplicate parameters); the
    target encoder is a deep-copy of the trunk + scaler + patch_proj that
    is updated via exponential moving average (no gradient).  At each
    step we run two forward passes through the trunk:

    1. Context: gradient-tracking, used by the pinball head and produces
       the trunk activations we predict from.
    2. Target: ``no_grad``, EMA-updated trunk weights, produces the
       targets we predict to.

    The predictor is a single-hidden-layer MLP that maps context_act[i]
    → predicted_target_act[i+1] (i.e., next-patch latent prediction,
    matching G3 / I-JEPA's recipe).  L2 loss between predictor output
    and the EMA target's i+1 representation, masked by valid positions.

    SIGReg (regulariser) is computed on the projected context tokens
    (not the raw trunk) — projecting first removes the per-dimension
    scale issues exp24 raised about RMSNorm + bf16.

    Parameters
    ----------
    d_model
        Trunk feature dim.
    proj_dim
        Projector / predictor latent dim (typically 256-512).
    ema_tau
        EMA momentum: target_param = tau * target_param + (1-tau) * online_param.
        ``0.999`` for a cosine-warmed schedule is the default; we use a
        constant for simplicity.
    sigreg_num_slices
        See :func:`sigreg_loss`.

    Notes
    -----
    - We **don't** stop_grad the SIGReg term: the LeJEPA paper specifically
      argues that SIGReg is the only collapse preventer needed and is
      compatible with end-to-end gradient flow.
    - The target encoder lives on ``self.target_module``.  ``setup_target``
      must be called *after* the lightning module has been built but
      *before* the first forward pass, so the deep copy sees the
      fully-initialised online weights.
    """

    def __init__(
        self,
        *,
        d_model: int,
        proj_dim: int = 256,
        ema_tau: float = 0.999,
        sigreg_num_slices: int = 1024,
        sigreg_n_quad_points: int = 17,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.ema_tau = float(ema_tau)
        self.sigreg_num_slices = int(sigreg_num_slices)
        self.sigreg_n_quad_points = int(sigreg_n_quad_points)

        # Projector: 2-hidden-layer MLP using u-μP-aware Linear + RMSNorm.
        # We **do not** use BatchNorm here:
        #
        #   1. Under DDP, BatchNorm only averages within each rank, so the
        #      stats it produces are per-rank — fine for image SSL where
        #      every rank sees the same distribution but suspect for our
        #      EEG corpus which has very different stats across recordings.
        #   2. The trunk + projector + predictor all use base_lr=0.3 (the
        #      u-μP-balanced trunk LR).  ``nn.Linear`` does not carry mup
        #      metadata, so the optimizer would apply this LR raw — far
        #      too high for fan_in=256 where the unit-scaled LR should be
        #      ~0.3/sqrt(256) = 0.019.  ``uu.Linear`` carries the correct
        #      mup_type='weight' tag.
        #   3. RMSNorm + uu.Linear is exactly the projector pattern used
        #      in LeJEPA-MINIMAL (see vendor/lejepa/lejepa/__init__.py
        #      and the paper §5.1 ablation showing RMSNorm matches BN
        #      for ViT-class projectors at our scale).
        self.projector = nn.Sequential(
            uu.Linear(d_model, 4 * d_model, bias=True, constraint=None),
            uu.RMSNorm(4 * d_model, eps=1e-5, include_weight=False),
            nn.GELU(),
            uu.Linear(4 * d_model, 4 * d_model, bias=True, constraint=None),
            uu.RMSNorm(4 * d_model, eps=1e-5, include_weight=False),
            nn.GELU(),
            uu.Linear(4 * d_model, proj_dim, bias=True, constraint=None),
        )

        # Predictor: one-hidden-layer MLP, also u-μP-compliant.
        self.predictor = nn.Sequential(
            uu.Linear(proj_dim, proj_dim, bias=True, constraint=None),
            nn.GELU(),
            uu.Linear(proj_dim, proj_dim, bias=True, constraint=None),
        )

        # EMA target — created lazily by setup_target so it inherits the
        # online weights *after* the lightning module finishes init.
        self._target_trunk: Optional[nn.Module] = None
        self._target_scaler: Optional[nn.Module] = None
        self._target_patch_proj: Optional[nn.Module] = None
        self._target_projector: Optional[nn.Module] = None

    def train(self, mode: bool = True):
        """Override to keep the EMA target tower in eval mode regardless.

        Lightning calls ``self.train(True)`` at the start of every
        training epoch which would normally toggle every submodule's
        ``training`` flag.  For the EMA target tower we always want
        ``eval()`` semantics — BatchNorm should read its EMA-updated
        running stats, not compute fresh batch stats.
        """
        super().train(mode)
        for tower in (
            self._target_trunk,
            self._target_scaler,
            self._target_patch_proj,
            self._target_projector,
        ):
            if tower is not None:
                tower.eval()
        return self

    def setup_target(
        self,
        *,
        scaler: nn.Module,
        patch_proj: nn.Module,
        trunk: nn.Module,
    ) -> None:
        """Deep-copy the online encoder + projector into the target tower.

        Called once at training-start by the lightning module.  After this
        the target tower is independent of the online tower; EMA updates
        sync them.  We deliberately **do not** add the target params to
        the optimiser — they get no gradients and are updated manually
        in ``update_ema``.

        We also force the target tower into ``eval`` mode so its
        BatchNorm reads its (EMA-updated) running stats instead of
        computing fresh batch stats — those would be doubly stochastic
        and noisy.
        """
        self._target_scaler = copy.deepcopy(scaler).eval()
        self._target_patch_proj = copy.deepcopy(patch_proj).eval()
        self._target_trunk = copy.deepcopy(trunk).eval()
        self._target_projector = copy.deepcopy(self.projector).eval()
        for p in self._target_scaler.parameters():
            p.requires_grad_(False)
        for p in self._target_patch_proj.parameters():
            p.requires_grad_(False)
        for p in self._target_trunk.parameters():
            p.requires_grad_(False)
        for p in self._target_projector.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(
        self,
        *,
        scaler: nn.Module,
        patch_proj: nn.Module,
        trunk: nn.Module,
    ) -> None:
        """EMA-update the target tower from the online tower.

        Called after each optimiser step by the lightning module.  Uses
        ``self.ema_tau`` as the target retention factor.
        """
        if self._target_trunk is None:
            return
        tau = self.ema_tau
        for tgt, src in (
            (self._target_scaler, scaler),
            (self._target_patch_proj, patch_proj),
            (self._target_trunk, trunk),
            (self._target_projector, self.projector),
        ):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.mul_(tau).add_(sp.detach(), alpha=1.0 - tau)
            for tb, sb in zip(tgt.buffers(), src.buffers()):
                if tb.dtype.is_floating_point:
                    tb.mul_(tau).add_(sb.detach(), alpha=1.0 - tau)
                else:
                    tb.copy_(sb.detach())

    @torch.no_grad()
    def _target_forward(
        self,
        *,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        cpm_mask: torch.Tensor,
        series_ids: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        """Run the EMA-updated target tower up to the projector.

        Returns a ``(B, V, S, proj_dim)`` tensor of target latents to
        predict.  Mirrors :meth:`Toto2Model.forward` up to (and including)
        ``transformer.out_norm`` then applies the EMA projector.  We
        **don't** invoke the head — JEPA targets live in latent space.
        """
        if self._target_trunk is None:
            raise RuntimeError("JEPAHead.setup_target must be called before forward.")

        scaled_series, _, _ = self._target_scaler(target, target_mask & cpm_mask)
        scaled_series = scaled_series.asinh()
        x = self._target_patch_proj(
            torch.cat(
                [
                    rearrange(scaled_series, "... (s p) -> ... s p", p=patch_size),
                    rearrange(
                        (~(target_mask & cpm_mask)).to(target.dtype),
                        "... (s p) -> ... s p",
                        p=patch_size,
                    ),
                ],
                dim=-1,
            )
        )
        group_ids = repeat(series_ids, "... n_var -> ... n_var seq", seq=x.shape[-2]).clone()
        from einops import reduce  # local import to avoid top-level pollution

        group_ids[
            (reduce(target_mask, "... (s p) -> ... s", "sum", p=patch_size) == 0)
            & (reduce(cpm_mask, "... (s p) -> ... s", "prod", p=patch_size) == 1)
        ] = -1
        x = self._target_trunk(x, group_ids=group_ids)
        # Project through the (also EMA'd) projector — uu.RMSNorm /
        # uu.Linear both work on (..., D) so no flatten is needed.
        x = self._target_projector(x)
        return x

    def forward(
        self,
        *,
        context_trunk_act: torch.Tensor,
        target_input: torch.Tensor,
        target_input_mask: torch.Tensor,
        target_input_cpm_mask: torch.Tensor,
        series_ids: torch.Tensor,
        patch_size: int,
        global_step: int,
        valid_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute (jepa_loss, sigreg_loss, metrics).

        ``context_trunk_act`` is the post-trunk pre-out_norm activation
        of the **online** encoder (captured by a forward-pre-hook).  We
        project it, predict next-patch latents, and L2 against the EMA
        target's projection.

        ``target_input`` etc. are the same input that fed the online
        encoder; they are fed through the **EMA target tower** in this
        method.

        ``valid_loss_mask`` is a ``(B, V, S)`` boolean tensor (True where
        the next-patch position is observed and the variate is real).
        Positions that are False are excluded from the latent-prediction
        loss.  ``None`` ⇒ all positions counted.
        """
        if context_trunk_act.shape[-1] != self.projector[0].in_features:
            raise ValueError(
                f"context_trunk_act last-dim {context_trunk_act.shape[-1]} != "
                f"projector in_features {self.projector[0].in_features}"
            )

        # Project online context: (B, V, S, proj_dim).  uu.Linear and
        # uu.RMSNorm both work on (..., D) so no flatten is needed.
        proj_context = self.projector(context_trunk_act)
        # Predict next-patch latents.
        pred_next = self.predictor(proj_context)

        # Get EMA target latents for the same input.
        target_lat = self._target_forward(
            target=target_input,
            target_mask=target_input_mask,
            cpm_mask=target_input_cpm_mask,
            series_ids=series_ids,
            patch_size=patch_size,
        )  # (B, V, S, proj_dim)

        # We predict next-patch (i -> i+1) so align: pred[..., i] vs target[..., i+1]
        pred_aligned = pred_next[..., :-1, :]
        target_aligned = target_lat[..., 1:, :].detach()  # stop-grad on target side
        if valid_loss_mask is not None:
            mask_aligned = valid_loss_mask[..., :-1].to(pred_aligned.dtype)  # (B, V, S-1)
            sq = (pred_aligned - target_aligned).pow(2).mean(dim=-1)  # (B, V, S-1)
            denom = mask_aligned.sum().clamp_min(1.0)
            jepa = (sq * mask_aligned).sum() / denom
        else:
            jepa = (pred_aligned - target_aligned).pow(2).mean()

        # SIGReg on flattened context (only the online side; the target
        # is detached and need not be regularised).
        sigreg = sigreg_loss(
            proj_context.reshape(-1, self.proj_dim),
            num_slices=self.sigreg_num_slices,
            n_quad_points=self.sigreg_n_quad_points,
            global_step=global_step,
        )

        with torch.no_grad():
            metrics = {
                "jepa_pred_norm": pred_aligned.norm(dim=-1).mean(),
                "jepa_target_norm": target_aligned.norm(dim=-1).mean(),
                "jepa_pred_target_cos": F.cosine_similarity(
                    pred_aligned.reshape(-1, self.proj_dim),
                    target_aligned.reshape(-1, self.proj_dim),
                    dim=-1,
                ).mean(),
            }
        return jepa, sigreg, metrics


# =====================================================================
# C. PARS pairwise relative shift head (Apple, arXiv:2511.11940)
# =====================================================================


class PARSHead(nn.Module):
    """Cross-attention head predicting the antisymmetric N×N matrix of
    pairwise relative temporal shifts.

    Implementation follows arXiv:2511.11940 §2.3 with a single
    cross-attention block for simplicity.  We sample ``N`` patch indices
    uniformly per batch and **mask** their positional info with a
    learnable ``no_position`` token so the encoder cannot trivially
    recover the position from RoPE.  The head then takes pairwise
    concatenations and predicts ``δ_ij = (j - i) / S`` ∈ [-1, 1].

    We deliberately keep the head small (1 attention layer, 256 hidden)
    so the auxiliary's parameter count stays well below the trunk's.

    Parameters
    ----------
    d_model
        Trunk feature dim.
    num_pairs
        Number of patches sampled per batch.  Higher = harder task but
        quadratic memory.
    head_dim
        Internal cross-attention head dim (per head).  ``num_heads`` is
        derived from ``d_model // head_dim``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_pairs: int = 12,
        head_dim: int = 64,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_pairs = int(num_pairs)
        if d_model % head_dim != 0:
            raise ValueError(f"d_model {d_model} must be divisible by head_dim {head_dim}")
        self.num_heads = d_model // head_dim
        self.head_dim = head_dim

        # Cross-attention: tokens (Q) attend to all sampled tokens (K, V).
        # nn.MultiheadAttention's internal Linear layers are not u-μP-aware
        # but the head is small (~1 layer) and we weight it 0.1, so the
        # mismatch is bounded.  An option for v2 of this probe is to
        # implement the cross-attention by hand using uu.Linear.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.norm1 = uu.RMSNorm(d_model, eps=1e-5, include_weight=False)
        self.norm2 = uu.RMSNorm(d_model, eps=1e-5, include_weight=False)
        self.mlp = nn.Sequential(
            uu.Linear(d_model, mlp_hidden, bias=True, constraint=None),
            nn.GELU(),
            uu.Linear(mlp_hidden, d_model, bias=True, constraint=None),
        )

        # Pairwise embedding -> scalar shift.
        self.pair_proj = uu.Linear(2 * d_model, mlp_hidden, bias=True, constraint=None)
        self.shift_head = uu.Linear(mlp_hidden, 1, bias=True, constraint=None)

    def forward(
        self,
        trunk_act: torch.Tensor,
        valid_seq_mask: Optional[torch.Tensor] = None,
        *,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the PARS auxiliary loss.

        Parameters
        ----------
        trunk_act
            ``(B, V, S, D)`` post-trunk pre-out_norm activations.  We
            collapse (B, V) into a single batch axis so we get one
            relative-shift task per (B, V) recording.
        valid_seq_mask
            Optional ``(B, V, S)`` boolean tensor.  Padding positions
            are excluded from the random sampling.
        rng
            Optional ``torch.Generator`` for deterministic sampling.

        Returns
        -------
        torch.Tensor
            Scalar PARS loss.
        dict
            Diagnostic metrics: ``pars_mae`` (mean abs shift error in
            normalised units), ``pars_pred_std`` (spread of predicted
            shifts).
        """
        if trunk_act.dim() != 4:
            raise ValueError(f"PARSHead expects (B, V, S, D); got {tuple(trunk_act.shape)}")
        B, V, S, D = trunk_act.shape
        N = self.num_pairs
        if N > S:
            raise ValueError(f"num_pairs {N} > seq len {S}")
        device = trunk_act.device

        # Flatten (B, V) -> single batch axis.
        flat = trunk_act.reshape(B * V, S, D)
        if valid_seq_mask is not None:
            valid_flat = valid_seq_mask.reshape(B * V, S)
        else:
            valid_flat = torch.ones(B * V, S, dtype=torch.bool, device=device)

        # Sample N indices per row.  We do an independent random sort and take
        # the top-N indices that are valid; this works around the lack of
        # vmap-style randperm in pytorch.  Deterministic if rng provided.
        u = torch.rand(B * V, S, device=device, generator=rng) - (~valid_flat).float() * 1e9
        idx = u.topk(N, dim=-1).indices  # (B*V, N)

        # Gather sampled tokens.
        sampled = torch.gather(
            flat,
            1,
            idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B*V, N, D)
        # Skip-connection: cross-attention with sampled tokens as Q, K, V.
        attn_out, _ = self.attn(sampled, sampled, sampled, need_weights=False)
        h = self.norm1(sampled + attn_out)
        h = self.norm2(h + self.mlp(h))  # (B*V, N, D)

        # Build pairwise embeddings (i, j) for all i, j and predict δ_ij.
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([h_i, h_j], dim=-1)  # (B*V, N, N, 2D)
        pair_h = F.gelu(self.pair_proj(pair))
        pred_shift = self.shift_head(pair_h).squeeze(-1)  # (B*V, N, N)

        # Ground-truth normalised shifts.  Larger S is the unit scale.
        # δ_ij = (idx_j - idx_i) / max(S - 1, 1)
        idx_f = idx.to(pred_shift.dtype)
        gt_shift = (idx_f.unsqueeze(1) - idx_f.unsqueeze(2)) / max(S - 1, 1)

        # MSE on the upper triangle (antisymmetric = redundant lower, zero diagonal).
        triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        loss = F.mse_loss(pred_shift[:, triu_mask], gt_shift[:, triu_mask])

        with torch.no_grad():
            mae = (pred_shift[:, triu_mask] - gt_shift[:, triu_mask]).abs().mean()
            pred_std = pred_shift[:, triu_mask].std()
            metrics = {
                "pars_mae": mae,
                "pars_pred_std": pred_std,
            }
        return loss, metrics


# =====================================================================
# D. Phase-aware Fourier loss (NeuroRVQ, arXiv:2510.13068)
# =====================================================================


def phase_aware_fourier_loss(
    *,
    quantiles: torch.Tensor,
    asinh_gt: torch.Tensor,
    weights: torch.Tensor,
    median_idx: int,
) -> Tuple[torch.Tensor, dict]:
    """Phase- and amplitude-aware Fourier loss on the median-knot prediction.

    Operates **in asinh / scaled space** (the same space the head
    predicts in) so we don't have to un-asinh / un-scale at every step.
    This is a small departure from NeuroRVQ proper, but the asinh is
    monotonic and the scale/loc are constant per patch, so the FFT of
    the asinh-scaled signal has the same phase structure as the FFT of
    the raw signal up to a frequency-domain re-weighting that is
    absorbed by the log-amplitude term.

    Parameters
    ----------
    quantiles
        ``(Q, B, V, S, P)`` predicted quantiles in asinh-scaled space.
    asinh_gt
        ``(B, V, S, P)`` ground-truth in asinh-scaled space.
    weights
        ``(B, V, S, P)`` observation weight (1.0 if observed, 0 else).
        Patches with zero weight are excluded from the loss.
    median_idx
        Index of the 0.5 quantile in ``quantiles``.

    Returns
    -------
    torch.Tensor
        Scalar loss = MSE(log-amp) + MSE(sin φ) + MSE(cos φ).
    dict
        Diagnostic metrics.
    """
    pred = quantiles[median_idx]  # (B, V, S, P)
    P = pred.shape[-1]
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)  # (B, V, S, P//2+1)
    gt_fft = torch.fft.rfft(asinh_gt.float(), dim=-1)

    eps = 1e-8

    pred_amp = pred_fft.abs()
    gt_amp = gt_fft.abs()
    log_amp_loss = (torch.log1p(pred_amp) - torch.log1p(gt_amp)).pow(2)

    pred_unit = pred_fft / (pred_amp + eps)
    gt_unit = gt_fft / (gt_amp + eps)
    sin_loss = (pred_unit.imag - gt_unit.imag).pow(2)
    cos_loss = (pred_unit.real - gt_unit.real).pow(2)

    fft_loss = log_amp_loss + sin_loss + cos_loss  # (B, V, S, P//2+1)

    # Mask: any patch where any sample is observed counts.
    patch_w = weights.float().mean(dim=-1, keepdim=True)  # (B, V, S, 1)
    fft_loss = fft_loss.mean(dim=-1) * patch_w.squeeze(-1)
    denom = patch_w.squeeze(-1).sum().clamp_min(1e-6)
    loss = fft_loss.sum() / denom

    with torch.no_grad():
        metrics = {
            "fft_log_amp": log_amp_loss.mean(),
            "fft_sin": sin_loss.mean(),
            "fft_cos": cos_loss.mean(),
        }
    return loss, metrics


# =====================================================================
# E. Multi-resolution STFT loss (Yamamoto et al., 2019; auraloss)
# =====================================================================


class MultiResolutionSTFTAuxLoss(nn.Module):
    """Vendored multi-resolution STFT loss (auraloss-compatible).

    We re-implement instead of pulling in ``auraloss`` so the project
    keeps a thin dependency tree and works in environments where pip
    can't reach the public PyPI mirror.  The math matches Yamamoto
    2019 / Steinmetz 2020 exactly:

        L_MR = (1/M) Σ_m [ L_SC(y, ŷ; m) + L_SM(y, ŷ; m) ]

    where ``L_SC`` is spectral convergence (Frobenius-normalised) and
    ``L_SM`` is the L1 of the log-magnitude difference.  We average
    over M = len(``fft_sizes``) resolutions.

    Inputs are the median-knot prediction in asinh-scaled space, i.e.
    ``quantiles[median_idx]`` flattened across the patch axis.
    """

    def __init__(
        self,
        *,
        fft_sizes: Tuple[int, ...] = (32, 128, 512),
        hop_sizes: Tuple[int, ...] = (8, 32, 128),
        win_sizes: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        if win_sizes is None:
            win_sizes = fft_sizes
        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError("fft_sizes / hop_sizes / win_sizes must be the same length")
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        self.win_sizes = tuple(win_sizes)

        for n_fft, win in zip(fft_sizes, win_sizes):
            self.register_buffer(
                f"window_{n_fft}",
                torch.hann_window(win, dtype=torch.float32),
                persistent=False,
            )

    def _stft_one(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        """Compute |STFT(x)| with the registered window.

        ``x`` is ``(N, T)`` (any leading axes flattened).  Output is
        ``(N, F, T_frames)`` magnitude.
        """
        window = getattr(self, f"window_{n_fft}").to(device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
            center=False,
        )
        return stft.abs()

    def forward(
        self,
        *,
        quantiles: torch.Tensor,
        asinh_gt: torch.Tensor,
        weights: torch.Tensor,
        median_idx: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the multi-resolution STFT auxiliary loss.

        Parameters mirror :func:`phase_aware_fourier_loss`.  We
        concatenate predicted patches across the seq axis to form a
        ``(B*V, S*P)`` 1-D signal, then run the STFTs.  Variate-padding
        positions (where every patch is unobserved) are dropped from
        the per-recording mean so they don't drag the loss.
        """
        pred = quantiles[median_idx]  # (B, V, S, P)
        B, V, S, P = pred.shape
        T = S * P

        # Reduce dtype to fp32 for STFT (cuFFT in bf16 is unsupported on
        # several backends).
        pred_flat = pred.reshape(B * V, T).float()
        gt_flat = asinh_gt.reshape(B * V, T).float()
        # Variate validity: True if any patch has any observed sample.
        valid = (weights.float().sum(dim=(-1, -2)) > 0).reshape(B * V)  # (B*V,)
        if not valid.any():
            return pred.new_tensor(0.0), {}

        pred_flat = pred_flat[valid]
        gt_flat = gt_flat[valid]

        sc_terms: list[torch.Tensor] = []
        sm_terms: list[torch.Tensor] = []
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            mag_p = self._stft_one(pred_flat, n_fft=n_fft, hop=hop, win=win)
            mag_g = self._stft_one(gt_flat, n_fft=n_fft, hop=hop, win=win)
            sc = torch.linalg.norm(mag_g - mag_p, dim=(-2, -1)) / (
                torch.linalg.norm(mag_g, dim=(-2, -1)).clamp_min(1e-6)
            )
            sm = (torch.log1p(mag_p) - torch.log1p(mag_g)).abs().mean(dim=(-2, -1))
            sc_terms.append(sc.mean())
            sm_terms.append(sm.mean())

        sc_loss = torch.stack(sc_terms).mean()
        sm_loss = torch.stack(sm_terms).mean()
        loss = sc_loss + sm_loss

        with torch.no_grad():
            metrics = {
                "stft_sc": sc_loss,
                "stft_sm": sm_loss,
            }
        return loss, metrics
