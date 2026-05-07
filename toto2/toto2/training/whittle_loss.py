# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

r"""Univariate Whittle pseudo-likelihood loss for Toto 2.0 EEG pre-training.

Implements the frequency-domain Gaussian likelihood (Whittle 1953) adapted as
a self-supervised loss for EEG foundation model training.  Per-channel PSD
prediction: the model's median quantile prediction implicitly defines a
predicted power spectral density whose fit to the observed periodogram is
evaluated under the Whittle NLL.

Mathematical formulation
------------------------

For a stationary Gaussian process with PSD :math:`S(\omega)`, the Whittle
pseudo-likelihood's negative log-likelihood is (Whittle 1953, Hannan 1973):

.. math::

    -\log L_{\text{Whittle}} = \sum_{k=1}^{K}
        \bigl[\log S(\omega_k) + I(\omega_k) / S(\omega_k)\bigr]

where:

* :math:`I(\omega_k) = |\hat{x}(\omega_k)|^2` is the **periodogram** of the
  observed data (target).
* :math:`S(\omega_k)` is the **model-predicted PSD** — here derived from
  the model's median prediction via :math:`S(\omega_k) = |\hat{y}(\omega_k)|^2`.

The Whittle NLL achieves its minimum when :math:`S(\omega_k) = I(\omega_k)`
at every frequency bin.  This is a KL divergence between empirical and
predicted spectral densities (Hannan 1973).

Why this is the right loss for EEG SSL
--------------------------------------

1. **Spectrally weighted by** :math:`1/S(\omega)`.  High-power frequencies
   (alpha, beta — where the signal lives) and low-power frequencies (gamma,
   drift — where noise lives) are normalised by their own variance.  The
   causal-scaler instability that pinball rewarded becomes irrelevant.

2. **The trunk has to learn the spectrum to win.**  There is no degenerate
   "smart causal scaler + identity residual" path: to minimise
   :math:`\log S + I/S`, the FFN must encode the actual frequency-domain
   structure (alpha-band power, beta-gamma coupling, etc.).  This is the
   structural fix to the v2/v3 trunk-collapse problem.

3. **Phase-agnostic on autospectra.**  The univariate Whittle depends only
   on :math:`|FFT|^2`, discarding within-channel phase.  For EEG, this is
   mostly fine — within a single channel, phase is only meaningful relative
   to a reference, and EEG references are arbitrary.

4. **Information-theoretic interpretation.**  Minimising the Whittle NLL
   minimises :math:`D_{KL}(\hat{S} \| S_\theta)` between empirical and
   predicted spectral densities — a representation-learning objective
   by construction.

5. **Operationally validated.**  Yu et al. ICML 2021 ("Whittle Networks"),
   UAI 2022 ("Predictive Whittle Networks"), Liu et al. 2025 (gravitational
   waves) have demonstrated differentiability and gradient stability of
   Whittle-based deep learning objectives.

Numerical stability
-------------------

* FFT is computed in fp32 (bf16 cuFFT not supported on all backends).
* PSD values are floored at ``psd_floor`` (default 1e-8) before log/division
  to avoid log(0) and division-by-zero.
* PSD is computed from ``real² + imag²`` directly (not through ``|z|``)
  to avoid the complex absolute-value singularity at z=0.
* Periodogram is computed via ``rfft`` with ``norm="ortho"`` so Parseval's
  theorem holds: :math:`\sum |x_t|^2 = \sum |\alpha_k|^2`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange


__all__ = [
    "whittle_loss_1d",
    "WhittleLoss",
]


def whittle_loss_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    patch_size: Optional[int] = None,
    concat_patches: bool = True,
    spectral_weight_exponent: float = 0.0,
    normalize_by_freq: bool = True,
    psd_floor: float = 1e-8,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    r"""Univariate Whittle pseudo-likelihood loss for 1-D signals / patched sequences.

    Parameters
    ----------
    pred
        Predicted point values (median quantile).  Shape ``(B, V, S, P)``
        (per-patch layout, same as Toto 2.0's quantile head outputs) or
        any ``(*lead, T)`` shape if ``concat_patches=False``.
    target
        Ground-truth values.  Same shape as ``pred``.
    mask
        Optional observation mask, same shape as ``pred``.  ``True`` = observed.
        Masked-out positions are zeroed before FFT.
    patch_size
        Required when ``concat_patches=True`` and ``pred`` has the
        ``(B, V, S, P)`` 4-D layout — used solely to assert the trailing
        axis matches.
    concat_patches
        If ``True`` (default), concatenate patches into a single 1-D signal
        per recording before the FFT, giving full frequency-range coverage
        (delta through gamma for HBN-EEG).  ``False`` runs per-patch FFT.
    spectral_weight_exponent
        Optional per-frequency weighting ``w(k) = (1+k)^exp``, normalised
        to unit mean.  ``0.0`` (default) = uniform.  ``2/3`` = EEG 1/f
        bias correction.
    normalize_by_freq
        If ``True`` (default), divide by number of frequency bins so the
        loss scale is comparable to pinball / MSE per element.  Required
        for the u-μP-balanced ``base_lr=0.3`` to transfer.
    psd_floor
        Minimum PSD value (prevents ``log(0)`` and ``I/0``).
    return_diagnostics
        If ``True``, also return a dict of summary diagnostics.

    Returns
    -------
    torch.Tensor
        Scalar Whittle NLL loss.
    dict, optional
        If ``return_diagnostics=True``, a diagnostic dict with keys:
        ``whittle_log_ratio``, ``whittle_psd_ratio``, ``whittle_n_freq``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match pred shape {tuple(pred.shape)}"
        )

    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        m = mask.float()
        pred_f = pred_f * m
        target_f = target_f * m

    if concat_patches:
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
        sig_pred = pred_f
        sig_gt = target_f
        valid_per_pos = mask

    # FFT (orthonormal so Parseval holds)
    fft_pred = torch.fft.rfft(sig_pred, dim=-1, norm="ortho")
    fft_gt = torch.fft.rfft(sig_gt, dim=-1, norm="ortho")
    n_freq = fft_pred.shape[-1]

    # PSD via real²+imag² (avoids complex |z| singularity at z=0)
    psd_pred = fft_pred.real.pow(2) + fft_pred.imag.pow(2)  # S(ω_k)
    psd_target = fft_gt.real.pow(2) + fft_gt.imag.pow(2)    # I(ω_k)

    # Floor PSD to prevent log(0) and division-by-zero
    floor = float(psd_floor)
    psd_pred_safe = psd_pred.clamp_min(floor)
    psd_target_safe = psd_target.clamp_min(floor)

    # Whittle NLL per frequency: log S(ω_k) + I(ω_k) / S(ω_k)
    log_term = torch.log(psd_pred_safe)
    ratio_term = psd_target_safe / psd_pred_safe
    per_freq = log_term + ratio_term

    # Optional spectral weighting
    if spectral_weight_exponent != 0.0:
        w = _spectral_weights(
            n_freq,
            exponent=float(spectral_weight_exponent),
            device=pred_f.device,
            dtype=pred_f.dtype,
        )
        per_freq = per_freq * w

    if normalize_by_freq:
        per_signal = per_freq.mean(dim=-1)
    else:
        per_signal = per_freq.sum(dim=-1)

    # Per-signal validity reduction
    if valid_per_pos is not None:
        valid = valid_per_pos.any(dim=-1)
        valid_w = valid.to(per_signal.dtype)
        denom_n = valid_w.sum().clamp_min(1e-8)
        loss = (per_signal * valid_w).sum() / denom_n
    else:
        loss = per_signal.mean()

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        # log(S_pred / I_target) averaged: should be ~0 at convergence
        log_ratio = (torch.log(psd_pred_safe) - torch.log(psd_target_safe)).mean()

        # S_pred / I_target averaged: should be ~1 at convergence
        psd_ratio = (psd_pred_safe / psd_target_safe).mean()

        # Per-band diagnostics for EEG (assuming sfreq=500, concat_patches=True)
        # Band boundaries in terms of frequency bin indices
        total_samples = sig_pred.shape[-1]
        freq_resolution = 1.0  # with ortho norm, freq bins are uniformly spaced
        # Just report the overall mean ratio per the low/high halves
        half = n_freq // 2
        low_freq_ratio = (psd_pred_safe[..., :half] / psd_target_safe[..., :half]).mean()
        high_freq_ratio = (psd_pred_safe[..., half:] / psd_target_safe[..., half:]).mean()

        diags: Dict[str, torch.Tensor] = {
            "whittle_log_ratio": log_ratio,
            "whittle_psd_ratio": psd_ratio,
            "whittle_low_freq_ratio": low_freq_ratio,
            "whittle_high_freq_ratio": high_freq_ratio,
            "whittle_n_freq": torch.tensor(float(n_freq), device=loss.device, dtype=loss.dtype),
            "whittle_log_term_mean": log_term.mean(),
            "whittle_ratio_term_mean": ratio_term.mean(),
        }
    return loss, diags


def _spectral_weights(
    n_freq: int,
    *,
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-frequency weighting ``w(k) = (1+k)^exponent``, normalised to mean 1."""
    if exponent == 0.0:
        return torch.ones(n_freq, device=device, dtype=dtype)
    k = torch.arange(n_freq, device=device, dtype=dtype)
    w = (1.0 + k).pow(exponent)
    return w / w.mean().clamp_min(torch.finfo(dtype).eps)


class WhittleLoss(nn.Module):
    """Module wrapper around :func:`whittle_loss_1d` with cached config.

    Stateless apart from the configuration knobs — no learnable parameters.

    Parameters
    ----------
    concat_patches
        See :func:`whittle_loss_1d`.  ``True`` by default.
    spectral_weight_exponent
        See :func:`whittle_loss_1d`.  ``0.0`` by default (uniform).
    normalize_by_freq
        See :func:`whittle_loss_1d`.  ``True`` by default.
    psd_floor
        Minimum PSD value for numerical stability.  Default ``1e-8``.
    """

    def __init__(
        self,
        *,
        concat_patches: bool = True,
        spectral_weight_exponent: float = 0.0,
        normalize_by_freq: bool = True,
        psd_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        self.concat_patches = bool(concat_patches)
        self.spectral_weight_exponent = float(spectral_weight_exponent)
        self.normalize_by_freq = bool(normalize_by_freq)
        self.psd_floor = float(psd_floor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        patch_size: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        return whittle_loss_1d(
            pred=pred,
            target=target,
            mask=mask,
            patch_size=patch_size,
            concat_patches=self.concat_patches,
            spectral_weight_exponent=self.spectral_weight_exponent,
            normalize_by_freq=self.normalize_by_freq,
            psd_floor=self.psd_floor,
            return_diagnostics=return_diagnostics,
        )

    def extra_repr(self) -> str:
        return (
            f"concat_patches={self.concat_patches}, "
            f"spectral_weight_exponent={self.spectral_weight_exponent}, "
            f"normalize_by_freq={self.normalize_by_freq}, "
            f"psd_floor={self.psd_floor}"
        )
