# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Multi-Resolution Magnitude-Phase Loss (MR-MPL) for Toto 2.0 EEG pre-training.

Designed to fix the failure mode observed with AMSE (exp43): the model
recovered the *spectrum* of the target (``amp_ratio = 0.84``) but had
near-random *phase* (``coh_mean = 0.19``).  The root cause is that
single-FFT spectral losses (AMSE, FACL, Whittle, ...) compute phase
coherence over the **entire** sequence — a phase error at any one time
point contributes to every frequency bin, so the gradient is averaged
across the full 4096-sample sequence and per-position phase information
dilutes to noise.

MR-MPL borrows the canonical loss from waveform-level audio synthesis
(HiFi-GAN, MelGAN, BigVGAN, Multi-Band MelGAN, HiFi-WaveGAN), where this
exact problem was solved 5+ years ago, and adds an explicit
**magnitude-weighted phase coherence** term to make the model jointly
match amplitudes AND phases at multiple time-frequency scales.

Math (per STFT scale ``m`` ∈ {1, ..., M}).  Let
``X = STFT(pred,  n_fft=N_m, hop=H_m, win=W_m, window=hann)`` and
``Y = STFT(target, n_fft=N_m, hop=H_m, win=W_m, window=hann)``
both with shape ``(B, V, F_m, T_m)``.

* **PSD** computed directly from ``real² + imag²`` so autograd never
  goes through the ``|z| = 0`` complex-abs singularity (lesson from
  the AMSE smoke-run post-mortem):

      psd_x = X.real² + X.imag²,   amp_x = sqrt(psd_x + eps²)

* **Term 1 — log-magnitude L1**::

      L_lm = mean | log(amp_x + eps) - log(amp_y + eps) |

  Log compression puts EEG's 1/f bands on comparable footing so the loss
  isn't dominated by delta and theta.

* **Term 2 — spectral convergence**::

      L_sc = ||amp_x - amp_y||_F / ||amp_y||_F

  Captures bulk magnitude shape relative to the target's energy.
  Yamamoto et al. 2019 / Arik et al. 2018; standard in audio synthesis.

* **Term 3 — magnitude-weighted phase coherence** (the headline
  innovation for time-series).  Phase wraparound at ±π breaks naïve
  ``MSE(angle(X), angle(Y))``; the wrap-safe form is::

      cos(θ_x - θ_y) = (X.real · Y.real + X.imag · Y.imag) / (amp_x · amp_y)
                     = Re(X · conj(Y)) / (amp_x · amp_y)

  Which is differentiable everywhere and bounded in ``[-1, +1]``.  We
  weight by the ground-truth power so the model isn't penalised for
  getting phase wrong where there is no signal::

      L_pc = sum_{f,t} amp_y² · (1 - cos(θ_x - θ_y))  /  sum_{f,t} amp_y²

  Phase coherence weighting by ``|Y|²`` follows HiFi-WaveGAN
  (arXiv:2210.12740), the von Mises STFT phase loss
  (Takamichi et al., IWAENC 2018), and the Phase Continuity Loss
  (arXiv:2202.11918) — all use this trick to keep phase gradients
  meaningful at low-energy bins.

* **Per-scale combination**::

      L_m = α · L_lm_m  +  β · L_sc_m  +  γ · L_pc_m

* **Across-scale combination**::

      L = (1 / M) · Σ_m L_m

Multiple resolutions are essential because EEG spans 5+ frequency decades
(0.5 Hz delta to 100 Hz gamma).  A short window gives good time
localisation for fast oscillations (gamma/beta), a long window gives good
frequency resolution for slow oscillations (delta/theta).  Recommended
defaults at 500 Hz with 4096-sample sequences:

    fft_sizes = (64, 256, 1024)   # 32 ms / 128 ms / 512 ms windows
    hop_sizes = (16, 64,  256)    # 75% overlap
    win_sizes = (64, 256, 1024)

This covers gamma+beta (short), alpha+beta (mid), and delta+theta (long).

Numerical robustness lessons from AMSE are carried through:
* PSD computed from ``real² + imag²`` (no |.| chain).
* All sqrt/log have ``+ eps²`` / ``+ eps`` floors.
* Coherence ratio clamped to ``[-1, 1]``.
* STFT computed in fp32 (cuFFT bf16 backends are unreliable).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


__all__ = [
    "mr_mpl_loss_1d",
    "MRMPLLoss",
]


def _hann_window(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Hann window cached implicitly via the call site (not via lru_cache so
    Lightning/DDP serialisation is well-behaved)."""
    return torch.hann_window(n, device=device, dtype=dtype, periodic=True)


def _stft_one_scale(
    sig: torch.Tensor,
    *,
    n_fft: int,
    hop: int,
    win_len: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute STFT, return (psd, complex_stft).

    ``sig`` is ``(N, T)`` (any leading axes flattened to N).  STFT is
    ``return_complex=True``, ``center=False`` so the output frame count is
    deterministic and there is no edge-padding bias on the per-frame
    phase.

    Output:
      * ``psd``  ∈ R^{N, F, T_frames}, with ``psd >= 0`` always.
      * ``stft`` ∈ C^{N, F, T_frames}.
    """
    if sig.shape[-1] < n_fft:
        # Defensive: torch.stft would raise.  We never expect to hit this
        # at training time, but the smoke tests call with tiny tensors.
        raise ValueError(
            f"signal length {sig.shape[-1]} < n_fft {n_fft}; "
            "MR-MPL requires the signal to be at least one window long."
        )
    window = _hann_window(win_len, device=sig.device, dtype=sig.dtype)
    # Promote to fp32 — bf16 cuFFT is unsupported on several CUDA backends.
    sig_f = sig.float()
    spec = torch.stft(
        sig_f,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=window.float(),
        center=False,
        return_complex=True,
        normalized=False,
        onesided=True,
        pad_mode="constant",
    )
    # PSD direct from real² + imag² (avoid |.| chain — same fix as AMSE).
    psd = spec.real.pow(2) + spec.imag.pow(2)
    return psd, spec


def _flatten_leading(*ts: torch.Tensor) -> Tuple[Tuple[int, ...], list]:
    """Collapse all leading axes except the last (time) into one batch axis.

    Returns the original leading shape (for un-flattening if needed) and
    the list of flattened tensors with shape ``(prod(lead), T)``.
    """
    if not ts:
        return (), []
    lead = ts[0].shape[:-1]
    flat = [t.reshape(-1, t.shape[-1]) for t in ts]
    return tuple(lead), flat


def mr_mpl_loss_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    fft_sizes: Sequence[int] = (64, 256, 1024),
    hop_sizes: Sequence[int] = (16, 64, 256),
    win_sizes: Optional[Sequence[int]] = None,
    alpha_lm: float = 1.0,
    beta_sc: float = 0.0,
    gamma_pc: float = 1.0,
    patch_size: Optional[int] = None,
    eps: float = 1e-4,
    log1p_compress: bool = True,
    nan_guard: bool = True,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Multi-Resolution Magnitude-Phase Loss for 1-D / patched signals.

    Parameters
    ----------
    pred
        Predicted point values.  Shape ``(B, V, S, P)`` (Toto 2.0
        per-patch layout) or any ``(*lead, T)`` shape.
    target
        Ground-truth values, same shape as ``pred``.
    mask
        Optional observation mask, same shape as ``pred``.  ``True``
        denotes observed.  Masked positions are zeroed before STFT so
        they contribute nothing to the spectrum; the per-recording
        validity reduction excludes recordings where every position is
        masked.
    fft_sizes / hop_sizes / win_sizes
        Per-scale STFT parameters.  Defaults are tuned for HBN-EEG
        (500 Hz sampling, 4096-sample sequences).  ``win_sizes`` defaults
        to ``fft_sizes``.
    alpha_lm / beta_sc / gamma_pc
        Sub-loss weights:

        * ``alpha_lm`` — weight on log-magnitude L1 (perceptual amplitude).
        * ``beta_sc``  — weight on spectral convergence (raw amplitude shape).
        * ``gamma_pc`` — weight on magnitude-weighted phase coherence.

        Defaults follow HiFi-GAN / BigVGAN conventions (1.0 / 0.5 / 1.0).
    patch_size
        Optional sanity check: when ``pred`` has the 4-D ``(B, V, S, P)``
        layout, the trailing axis must equal ``patch_size``.  Used only
        for an early-failure assertion; the loss itself doesn't care
        about the patch layout (it concatenates ``S * P`` into a single
        per-recording sequence before STFT).
    eps
        Numerical guard for divisions / sqrts / logs.  Default ``1e-4``;
        bumped from the AMSE convention of ``1e-7`` because the GPU smoke
        showed NaN gradients under DDP when amp_x or amp_y underflowed
        toward eps.  ``1e-4`` is well above any meaningful asinh-EEG
        amplitude (in normalised z-score space the smallest non-trivial
        amp is ~10⁻³ for a single very-low-energy STFT bin) so it does
        not blunt the loss but does give the gradient through
        ``log(amp + eps)`` and ``cos_diff = cross / (amp_x · amp_y)`` a
        bounded-curvature regime to live in.
    log1p_compress
        If ``True`` (default), use ``torch.log1p(amp) = log(1 + amp)``
        instead of ``torch.log(amp + eps)`` for the log-magnitude term.
        ``log1p`` is monotone, equals 0 at ``amp = 0``, and has gradient
        ``1 / (1 + amp)`` which is bounded by 1 — fixes the
        ``1/(amp + eps)`` gradient blow-up at low-amplitude bins that
        triggered NaN under DDP.  Set to ``False`` to recover the
        plain-log form (matches some HiFi-GAN implementations).
    nan_guard
        If ``True`` (default), wrap the final loss in
        ``torch.nan_to_num(loss, nan=0, posinf=1e3, neginf=-1e3)``.
        This is a defensive backstop — if the upstream numerical
        guards miss a single bad bin under DDP, this prevents the bad
        rank's NaN from being all-reduced into every other rank's
        gradient.  The masked value (``0``) is the per-recording
        contribution from a fully-failed STFT scale; downstream
        diagnostics (``mr_mpl_lm/sc/pc``) still preserve the raw
        un-guarded values for debugging.
    return_diagnostics
        If ``True``, also return a dict of detached diagnostic tensors
        (per-scale L_lm / L_sc / L_pc, mean cos(θ_diff), amplitude ratio,
        total frame count).

    Returns
    -------
    torch.Tensor
        Scalar MR-MPL loss.
    dict, optional
        If ``return_diagnostics=True``, the diagnostic dict.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match pred shape {tuple(pred.shape)}"
        )
    if win_sizes is None:
        win_sizes = fft_sizes
    if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
        raise ValueError(
            f"fft_sizes / hop_sizes / win_sizes must have the same length; got "
            f"{len(fft_sizes)} / {len(hop_sizes)} / {len(win_sizes)}"
        )
    M = len(fft_sizes)
    if M == 0:
        raise ValueError("MR-MPL needs at least one STFT scale.")

    # ----- collapse patches to a single 1-D signal per (B, V) recording ---
    # Toto-2's per-patch outputs are (B, V, S, P).  We flatten the trailing
    # (S, P) into (S*P) so the STFT sees the full 4096-sample sequence the
    # model would predict autoregressively.  This matches what AMSE does
    # with concat_patches=True, but here the windows are short (32-512 ms)
    # so the multi-resolution structure picks up the per-patch
    # boundaries instead of averaging across them.
    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        m = mask.float()
        pred_f = pred_f * m
        target_f = target_f * m

    if pred_f.ndim >= 4:
        if patch_size is not None and pred_f.shape[-1] != patch_size:
            raise ValueError(
                f"Trailing axis {pred_f.shape[-1]} != patch_size {patch_size}"
            )
        sig_pred = pred_f.flatten(start_dim=-2)        # (..., S*P)
        sig_gt = target_f.flatten(start_dim=-2)
        valid_per_pos = (
            mask.flatten(start_dim=-2) if mask is not None else None
        )
    else:
        sig_pred = pred_f
        sig_gt = target_f
        valid_per_pos = mask

    lead_shape, (sig_pred_flat, sig_gt_flat) = _flatten_leading(sig_pred, sig_gt)
    if valid_per_pos is not None:
        _, (valid_flat,) = _flatten_leading(valid_per_pos)
    else:
        valid_flat = None

    # Per-recording validity: True if at least one observed sample.
    # (We drop fully-masked recordings from the per-recording mean to
    # avoid contaminating the loss with a zero-signal AMSE = 0 row.)
    if valid_flat is not None:
        recording_valid = valid_flat.any(dim=-1).float()  # (N,)
    else:
        recording_valid = None

    eps_sq = float(eps) * float(eps)

    per_scale_losses: list[torch.Tensor] = []
    diag: Dict[str, list] = {
        "lm": [], "sc": [], "pc": [], "cos_mean": [], "amp_ratio": [], "n_frames": []
    }

    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        psd_x, spec_x = _stft_one_scale(sig_pred_flat, n_fft=n_fft, hop=hop, win_len=win, eps=eps)
        psd_y, spec_y = _stft_one_scale(sig_gt_flat,   n_fft=n_fft, hop=hop, win_len=win, eps=eps)

        amp_x = (psd_x + eps_sq).sqrt()
        amp_y = (psd_y + eps_sq).sqrt()

        # ---- Term 1: log-magnitude L1 ----
        # Per-element L1 between log-amplitudes.  Use log1p(amp) by default
        # because log(amp + eps) has gradient 1/(amp + eps) which blows up
        # to 1/eps when amp underflows toward 0 (caused NaN under DDP all-
        # reduce in the GPU smoke).  log1p(amp) has gradient 1/(1+amp),
        # bounded by 1 everywhere.  Also equals 0 at amp=0 (log of 1=0)
        # which is the right "no signal here" value.
        if log1p_compress:
            lm = (torch.log1p(amp_x) - torch.log1p(amp_y)).abs()      # (N, F, T)
        else:
            lm = (torch.log(amp_x + eps) - torch.log(amp_y + eps)).abs()  # (N, F, T)

        # ---- Term 2: spectral convergence (DISABLED by default) ----
        # Frame-wise relative L2 of the amplitude difference.  In the GPU
        # smoke, SC exploded to ~1e5 on asinh-scaled EEG (wide dynamic
        # range, ±13 z-scores) and dominated the gradient.  Kept for
        # completeness but ``beta_sc=0.0`` by default.
        if beta_sc != 0.0:
            diff = amp_x - amp_y                                          # (N, F, T)
            sc_num = diff.pow(2).sum(dim=(-2, -1)).sqrt()                 # (N,)
            sc_den = amp_y.pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(eps)
            sc_per_rec = sc_num / sc_den                                   # (N,)
        else:
            sc_per_rec = torch.zeros_like(lm.sum(dim=(-2, -1)))            # (N,) of zeros

        # ---- Term 3: magnitude-weighted phase coherence ----
        # cross_re = Re(X * conj(Y)) computed in real arithmetic (no .conj()).
        cross_re = spec_x.real * spec_y.real + spec_x.imag * spec_y.imag  # (N, F, T)
        # Use a *larger* denom-eps (configurable, default 1e-4) so the
        # cos_diff gradient stays bounded even at low-energy bins where
        # amp_x · amp_y is tiny.  This is the critical fix for the DDP NaN.
        denom = (amp_x * amp_y).clamp_min(eps)
        cos_diff = (cross_re / denom).clamp(-1.0, 1.0)                 # (N, F, T)
        weight_pc = psd_y                                              # |Y|² weight
        # Aggregate per recording so the L_pc value is bounded in [0, 2]:
        pc_num_per_rec = (weight_pc * (1.0 - cos_diff)).sum(dim=(-2, -1))
        pc_den_per_rec = weight_pc.sum(dim=(-2, -1)).clamp_min(eps)
        pc_per_rec = pc_num_per_rec / pc_den_per_rec                   # (N,)

        # ---- Per-scale per-recording L_lm: mean over (F, T) ----
        lm_per_rec = lm.mean(dim=(-2, -1))                             # (N,)

        # ---- Per-recording validity mask ----
        if recording_valid is not None:
            denom_n = recording_valid.sum().clamp_min(1.0)
            lm_m = (lm_per_rec * recording_valid).sum() / denom_n
            sc_m = (sc_per_rec * recording_valid).sum() / denom_n
            pc_m = (pc_per_rec * recording_valid).sum() / denom_n
        else:
            lm_m = lm_per_rec.mean()
            sc_m = sc_per_rec.mean()
            pc_m = pc_per_rec.mean()

        L_m = alpha_lm * lm_m + beta_sc * sc_m + gamma_pc * pc_m
        per_scale_losses.append(L_m)

        if return_diagnostics:
            with torch.no_grad():
                # Mean cos(θ_diff) across all bins, gt-PSD weighted (matches the
                # AMSE coh_mean diagnostic for cross-comparison with exp43).
                cos_w_num = (weight_pc * cos_diff).sum()
                cos_w_den = weight_pc.sum().clamp_min(eps)
                cos_mean_w = cos_w_num / cos_w_den
                amp_ratio = amp_x.mean() / amp_y.mean().clamp_min(eps)
                n_frames = torch.tensor(
                    float(spec_x.shape[-1]), device=L_m.device, dtype=L_m.dtype
                )
            diag["lm"].append(lm_m.detach())
            diag["sc"].append(sc_m.detach())
            diag["pc"].append(pc_m.detach())
            diag["cos_mean"].append(cos_mean_w.detach())
            diag["amp_ratio"].append(amp_ratio.detach())
            diag["n_frames"].append(n_frames.detach())

    L = torch.stack(per_scale_losses).mean()

    # Defensive nan_to_num finalizer.  This is a backstop — if any upstream
    # numerical guard misses a single bad bin (e.g., a degenerate STFT
    # output on one rank under DDP), this prevents the bad rank's NaN from
    # being all-reduced into every other rank's gradient and corrupting
    # the entire model in a single step.  In normal operation this is a
    # no-op (the upstream guards keep everything finite).
    if nan_guard:
        L = torch.nan_to_num(L, nan=0.0, posinf=1e3, neginf=-1e3)

    if not return_diagnostics:
        return L

    # Aggregate diagnostics across scales (mean) and per-scale (stack -> ...0/1/2 indices).
    diags: Dict[str, torch.Tensor] = {}
    for k, v in diag.items():
        if not v:
            continue
        stacked = torch.stack(v)
        diags[f"mr_mpl_{k}_mean"] = stacked.mean()
        for i, val in enumerate(v):
            diags[f"mr_mpl_{k}_scale{i}"] = val
    return L, diags


class MRMPLLoss(nn.Module):
    """Module wrapper around :func:`mr_mpl_loss_1d` with cached config.

    No learnable parameters.  Wrapping in ``nn.Module`` lets the Lightning
    module own it as a submodule and surface the configuration through
    ``state_dict()`` for checkpoint introspection.

    Parameters
    ----------
    fft_sizes / hop_sizes / win_sizes
        Per-scale STFT parameters.  Defaults to (64, 256, 1024) /
        (16, 64, 256) / same as fft (Hann window, 75% overlap).  See
        :func:`mr_mpl_loss_1d` for the EEG rationale.
    alpha_lm / beta_sc / gamma_pc
        Sub-loss weights (1.0 / 0.5 / 1.0 by default).
    """

    def __init__(
        self,
        *,
        fft_sizes: Sequence[int] = (64, 256, 1024),
        hop_sizes: Sequence[int] = (16, 64, 256),
        win_sizes: Optional[Sequence[int]] = None,
        alpha_lm: float = 1.0,
        beta_sc: float = 0.0,
        gamma_pc: float = 1.0,
        eps: float = 1e-4,
        log1p_compress: bool = True,
        nan_guard: bool = True,
    ) -> None:
        super().__init__()
        if win_sizes is None:
            win_sizes = fft_sizes
        if not (len(fft_sizes) == len(hop_sizes) == len(win_sizes)):
            raise ValueError("fft / hop / win sizes must be the same length")
        self.fft_sizes = tuple(int(x) for x in fft_sizes)
        self.hop_sizes = tuple(int(x) for x in hop_sizes)
        self.win_sizes = tuple(int(x) for x in win_sizes)
        self.alpha_lm = float(alpha_lm)
        self.beta_sc = float(beta_sc)
        self.gamma_pc = float(gamma_pc)
        self.eps = float(eps)
        self.log1p_compress = bool(log1p_compress)
        self.nan_guard = bool(nan_guard)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        patch_size: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        return mr_mpl_loss_1d(
            pred=pred,
            target=target,
            mask=mask,
            fft_sizes=self.fft_sizes,
            hop_sizes=self.hop_sizes,
            win_sizes=self.win_sizes,
            alpha_lm=self.alpha_lm,
            beta_sc=self.beta_sc,
            gamma_pc=self.gamma_pc,
            patch_size=patch_size,
            eps=self.eps,
            log1p_compress=self.log1p_compress,
            nan_guard=self.nan_guard,
            return_diagnostics=return_diagnostics,
        )

    def extra_repr(self) -> str:
        return (
            f"fft_sizes={self.fft_sizes}, hop_sizes={self.hop_sizes}, "
            f"win_sizes={self.win_sizes}, "
            f"alpha_lm={self.alpha_lm}, beta_sc={self.beta_sc}, gamma_pc={self.gamma_pc}, "
            f"eps={self.eps}, log1p_compress={self.log1p_compress}, "
            f"nan_guard={self.nan_guard}"
        )
