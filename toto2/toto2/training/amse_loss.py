# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Adjusted Mean Squared Error (AMSE) loss for Toto 2.0 EEG pre-training.

Implements the spectral-decomposition loss from Subich et al. *Fixing the
Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical
Harmonic Loss Function* (ICML 2025, arXiv:2501.19374), adapted from
2-D spherical harmonics to 1-D FFTs over time-series patches.

Motivation (the "double penalty" / amplitude-phase entanglement problem).
------------------------------------------------------------------------

Pointwise losses (MSE, L1, pinball) have a well-known failure mode known
as the *double penalty*: a forecast that correctly predicts a feature
(spike, spindle, evoked potential) but at a slightly wrong location is
penalised twice — once for missing the feature at its real location and
once for predicting it at an incorrect location.  The least-cost solution
to a pointwise loss under that penalty is to *smooth out* the feature,
suppressing fine-scale variability that the model knows is going to be
hard to predict precisely.

For EEG, the symptom is a forecast that gets the *bulk amplitude* right
(low pinball) but loses the spectral signature of the underlying neural
activity (low FFN information content, FFN weight RMS underflowing).
exp22-26 / exp36 documented this trunk-collapse pattern in detail.

Subich et al.'s fix.
--------------------

Decompose the MSE into a spectral form (Parseval's identity) and split
the cross-term so amplitude and coherence (phase) are penalised
independently:

    MSE   (x, y) = Σ_k  (√PSD_k(x) − √PSD_k(y))²
                 + 2 √(PSD_k(x) PSD_k(y)) · (1 − Coh_k(x, y))

    AMSE (x, y) = Σ_k  (√PSD_k(x) − √PSD_k(y))²
                 + 2  max(PSD_k(x), PSD_k(y)) · (1 − Coh_k(x, y))

where ``PSD_k(z) = |FFT(z)[k]|²`` is the per-frequency power spectral
density and ``Coh_k = Re(α_z α_y*) / √(PSD_k(z)·PSD_k(y))`` is the
coherence at frequency *k*.

Key properties (see paper §2.3):

* AMSE is **zero iff x = y** — same global optimum as MSE.
* AMSE has the **same Taylor expansion as MSE near the optimum** (i.e.,
  for ``PSD_k(x) ≈ PSD_k(y)``, ``max ≈ √(xy)``, so AMSE ≈ MSE).
* AMSE **gradients always point toward** ``PSD_k(x) → PSD_k(y)``
  *and* ``Coh_k → 1``, even when physical predictability limits coherence.
  Pure MSE prefers to **shrink amplitude when coherence is poor**, which
  is exactly the smoothing pathology we want to avoid.
* AMSE is **parameter-free**: no cutoff frequencies, no schedule, no λ.
  An optional spectral weight ``S(k)`` is exposed below for the EEG-specific
  1/f bias correction discussed in the abandoned exp43 spec, but the
  default (``spectral_weight=0.0``) reproduces vanilla AMSE.

1-D adaptation for time-series patches.
---------------------------------------

The original paper aggregates spectral coefficients across the zonal
modes ``l`` of each total wavenumber ``k`` to obtain a stable PSD
estimate.  In 1-D, each frequency bin is a single complex coefficient
per signal, so ``PSD_k = |α_k|²`` is a 1-sample estimate per signal.
Two natural granularities are exposed:

* ``concat_patches=True`` (default): concatenate the ``S`` predicted
  patches into a single ``(B, V, S·P)`` 1-D signal per (batch, variate)
  recording, take ``rfft`` over the time axis.  This gives the **full**
  frequency range from ``sfreq/(S·P)`` (≈ 0.12 Hz at HBN-EEG) up to
  Nyquist (250 Hz), capturing delta/theta/alpha/beta/gamma bands.
  Recommended for EEG.

* ``concat_patches=False``: per-patch FFT.  Frequency range is limited
  to ``[sfreq/P, Nyquist]`` (≈ 7.8 Hz – 250 Hz at HBN-EEG, missing
  delta/theta), but the loss matches the per-patch granularity of
  pinball exactly.  Useful as a control / for short-context tasks.

Either way, AMSE is averaged across the (B, V) recording axis at the
end so that scaling is comparable to per-position pointwise losses.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange


__all__ = [
    "amse_loss_1d",
    "AMSELoss",
]


def _spectral_weights(
    n_freq: int,
    *,
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a per-frequency weighting vector ``S(k) = (1+k)^exponent``.

    Normalised to mean 1 so the overall loss magnitude does not depend
    on the choice of ``exponent``.  ``exponent = 0`` returns all-ones
    (no weighting), ``exponent = 2/3`` is the "EEG 1/f bias correction"
    suggested in the abandoned exp43 spec — it upweights the high-
    frequency residuals where EEG has low natural power so the model
    cannot win by getting only the low frequencies right.
    """
    if exponent == 0.0:
        return torch.ones(n_freq, device=device, dtype=dtype)
    k = torch.arange(n_freq, device=device, dtype=dtype)
    w = (1.0 + k).pow(exponent)
    return w / w.mean().clamp_min(torch.finfo(dtype).eps)


def amse_loss_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    patch_size: Optional[int] = None,
    concat_patches: bool = True,
    spectral_weight_exponent: float = 0.0,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Adjusted MSE (Subich et al. 2025) for 1-D signals / patched sequences.

    Parameters
    ----------
    pred
        Predicted point values.  Shape ``(B, V, S, P)`` (per-patch
        layout, the same as Toto 2.0's quantile head outputs) or any
        ``(*lead, T)`` shape if ``concat_patches=False``.
    target
        Ground-truth values.  Same shape as ``pred``.
    mask
        Optional observation mask, same shape as ``pred``.  ``True``
        denotes an observed value.  When provided, masked-out positions
        are zeroed *before* the FFT (so they contribute nothing to the
        spectrum) and the per-recording reduction excludes
        recordings/patches with no observed samples.
    patch_size
        Required when ``concat_patches=True`` and ``pred`` has the
        ``(B, V, S, P)`` 4-D layout — used solely to assert the trailing
        axis matches.
    concat_patches
        See module docstring.  ``True`` (default) concatenates patches
        into a single 1-D signal per recording before the FFT, giving
        full frequency-range coverage; ``False`` runs the FFT per patch.
    spectral_weight_exponent
        Optional per-frequency weighting.  ``0.0`` (default) is uniform;
        ``2/3 ≈ 0.6667`` upweights high frequencies to compensate for
        EEG's roughly 1/f power spectrum.
    eps
        Numerical guard for the coherence ratio ``cross/(|X|·|Y|)``.
    return_diagnostics
        If ``True``, also return a dict of summary tensors (mean amplitude
        term, mean coherence term, mean coherence value, predicted/target
        amplitude ratio).  All diagnostics are detached.

    Returns
    -------
    torch.Tensor
        Scalar AMSE loss.
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

    # ----- prepare 1-D signal and a per-recording validity mask --------
    # We promote to fp32 for the FFT — bf16 cuFFT is not supported on
    # several CUDA backends, and the spectral magnitudes here are
    # squared so fp16/bf16 dynamic range would saturate quickly.
    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        m = mask.float()
        # Zero masked-out positions before FFT so they contribute nothing
        # to the spectrum.  This is the conservative choice (versus,
        # e.g., interpolating).  Padding does introduce zero-padding
        # frequencies, but we zero in *both* signals identically so the
        # AMSE residual only sees the predicted-vs-truth difference at
        # observed positions.
        pred_f = pred_f * m
        target_f = target_f * m

    if concat_patches:
        # (B, V, S, P) -> (B, V, S*P).  We support arbitrary leading
        # axes for generality but the typical Toto 2.0 layout is 4-D.
        if pred_f.ndim < 2:
            raise ValueError(
                "concat_patches=True requires at least a (S, P) layout; got "
                f"shape {tuple(pred.shape)}"
            )
        if pred_f.ndim >= 4:
            if patch_size is not None and pred_f.shape[-1] != patch_size:
                raise ValueError(
                    f"Trailing axis {pred_f.shape[-1]} != patch_size {patch_size}"
                )
            sig_pred = rearrange(pred_f, "... s p -> ... (s p)")
            sig_gt = rearrange(target_f, "... s p -> ... (s p)")
            valid_per_pos = (
                rearrange(mask, "... s p -> ... (s p)") if mask is not None else None
            )
        else:
            sig_pred = pred_f
            sig_gt = target_f
            valid_per_pos = mask
    else:
        # Per-patch FFT — keep the trailing P axis.
        sig_pred = pred_f
        sig_gt = target_f
        valid_per_pos = mask

    # ----- FFT (orthonormal so Parseval holds: Σ|x|² = Σ|α|²) ----------
    fft_pred = torch.fft.rfft(sig_pred, dim=-1, norm="ortho")
    fft_gt = torch.fft.rfft(sig_gt, dim=-1, norm="ortho")
    n_freq = fft_pred.shape[-1]

    # ----- per-frequency amplitudes / cross / PSD ---------------------
    amp_x = fft_pred.abs()       # |α_x_k|
    amp_y = fft_gt.abs()         # |α_y_k|
    psd_x = amp_x * amp_x        # PSD_k(x)
    psd_y = amp_y * amp_y        # PSD_k(y)
    cross_re = (fft_pred * fft_gt.conj()).real

    # Coherence Coh_k = Re(α_x α_y*) / (|α_x|·|α_y|).  Bound to [-1, 1]
    # to absorb numerical noise — the analytic value is always in this
    # range for real signals but rounding can produce slight overshoot.
    denom = (amp_x * amp_y).clamp_min(eps)
    coh = (cross_re / denom).clamp(-1.0, 1.0)

    # ----- AMSE per frequency ------------------------------------------
    amp_term = (amp_x - amp_y).pow(2)                            # (√PSD_x − √PSD_y)²
    coh_term = 2.0 * torch.maximum(psd_x, psd_y) * (1.0 - coh)   # 2·max·(1−Coh)

    # Optional EEG-style spectral weighting.
    if spectral_weight_exponent != 0.0:
        w = _spectral_weights(
            n_freq,
            exponent=float(spectral_weight_exponent),
            device=pred_f.device,
            dtype=pred_f.dtype,
        )
        amp_term = amp_term * w
        coh_term = coh_term * w

    per_freq = amp_term + coh_term            # (..., n_freq)
    per_signal = per_freq.sum(dim=-1)         # sum_k → per-signal AMSE

    # ----- per-signal validity reduction ------------------------------
    # When concat_patches=True: per_signal has shape (..., ); we drop
    # signals where every position is masked.
    # When concat_patches=False: per_signal has shape (..., S); we drop
    # patches where every position is masked.
    if valid_per_pos is not None:
        # Boolean: True if at least one observed position contributes.
        valid = valid_per_pos.any(dim=-1)                 # (..., ) or (..., S)
        valid_w = valid.to(per_signal.dtype)
        denom_n = valid_w.sum().clamp_min(eps)
        amse = (per_signal * valid_w).sum() / denom_n
    else:
        amse = per_signal.mean()

    if not return_diagnostics:
        return amse

    with torch.no_grad():
        # All diagnostics are pre-aggregation (mean over freq + signal axes)
        # so they are insensitive to the validity mask.  This keeps the
        # numbers comparable across batches with different observation
        # patterns.
        diags: Dict[str, torch.Tensor] = {
            "amse_amp_term": amp_term.mean(),
            "amse_coh_term": coh_term.mean(),
            "amse_coh_mean": coh.mean(),
            # Predicted-to-truth amplitude ratio per frequency, averaged
            # across signals.  ~1.0 means the spectrum is right; <1
            # means the model is smoothing (the classic MSE pathology
            # AMSE is designed to fix), >1 means it is over-sharpening.
            "amse_amp_ratio": (amp_x.mean() / amp_y.mean().clamp_min(eps)),
            # Number of frequency bins (helps interpret the magnitudes
            # above — they are pre-sum across freq).
            "amse_n_freq": torch.tensor(float(n_freq), device=amse.device, dtype=amse.dtype),
        }
    return amse, diags


class AMSELoss(nn.Module):
    """Module wrapper around :func:`amse_loss_1d` with cached config.

    Stateless apart from the configuration knobs — no learnable parameters,
    no buffers worth registering.  Wrapping in ``nn.Module`` lets the
    Lightning module own it as a submodule and makes the configuration
    discoverable via ``state_dict()`` for checkpoint introspection.

    Parameters
    ----------
    concat_patches
        See :func:`amse_loss_1d`.  ``True`` by default (recommended for EEG).
    spectral_weight_exponent
        See :func:`amse_loss_1d`.  ``0.0`` by default (uniform weighting).
    """

    def __init__(
        self,
        *,
        concat_patches: bool = True,
        spectral_weight_exponent: float = 0.0,
    ) -> None:
        super().__init__()
        self.concat_patches = bool(concat_patches)
        self.spectral_weight_exponent = float(spectral_weight_exponent)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        patch_size: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        return amse_loss_1d(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            concat_patches=self.concat_patches,
            spectral_weight_exponent=self.spectral_weight_exponent,
            return_diagnostics=return_diagnostics,
        )

    def extra_repr(self) -> str:
        return (
            f"concat_patches={self.concat_patches}, "
            f"spectral_weight_exponent={self.spectral_weight_exponent}"
        )
