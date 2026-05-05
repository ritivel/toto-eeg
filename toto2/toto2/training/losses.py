# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Quantile (pinball) loss for Toto 2.0 training.

Toto 2.0 produces fixed-knot quantile predictions (default 9 knots:
[0.1, 0.2, …, 0.9]). The natural objective is the average pinball loss across
those knots, which is the discrete approximation of CRPS used to evaluate the
model on BOOM and GIFT-Eval (parkCRPS).

Pinball loss for quantile level ``tau`` and error ``e = y - y_hat``::

    L_tau(e) = max(tau * e, (tau - 1) * e)

We optionally smooth the kink at ``e = 0`` with a Huber-style transition
(``huber_kappa > 0``); this improves numerical conditioning at very small
errors without changing the asymptotic behaviour.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    huber_kappa: float = 0.0,
    reduction: str = "mean_quantiles",
) -> torch.Tensor:
    """Pinball / quantile loss.

    Parameters
    ----------
    predictions
        Quantile predictions with shape ``(Q, *batch_shape)`` — the same layout
        produced by :class:`toto2.model.QuantileKnotsOutputHead`.
    targets
        Ground-truth values with shape ``(*batch_shape,)``.
    quantile_levels
        1-D tensor of length ``Q`` with the quantile levels in ``(0, 1)``.
    huber_kappa
        If positive, replace the kink at ``e = 0`` with a quadratic of width
        ``2 * huber_kappa`` (Huber smoothing). ``0.0`` recovers the exact
        pinball loss.
    reduction
        - ``"none"``: returns the per-element loss with shape ``(Q, *batch)``.
        - ``"mean_quantiles"``: average across the leading quantile axis;
          returns shape ``(*batch,)``. This is the recommended default and
          matches mean weighted quantile loss / discrete CRPS.
        - ``"mean"``: average over every dimension; returns a scalar.

    Returns
    -------
    torch.Tensor
        Loss tensor whose shape depends on ``reduction``.
    """
    if predictions.shape[0] != quantile_levels.numel():
        raise ValueError(
            f"predictions axis 0 has length {predictions.shape[0]} but quantile_levels "
            f"has length {quantile_levels.numel()}; they must match."
        )
    if predictions.shape[1:] != targets.shape:
        raise ValueError(
            f"Trailing shape mismatch: predictions[1:]={tuple(predictions.shape[1:])} "
            f"vs targets={tuple(targets.shape)}."
        )

    # Broadcast quantile levels to (Q, 1, ..., 1)
    levels = quantile_levels.to(predictions.dtype).to(predictions.device)
    view_shape = (-1,) + (1,) * (predictions.ndim - 1)
    levels = levels.view(view_shape)

    errors = targets.unsqueeze(0) - predictions  # (Q, *batch)

    if huber_kappa > 0.0:
        # Huberized pinball: smooth quadratic for |e| <= kappa, linear outside.
        abs_e = errors.abs()
        kappa = errors.new_tensor(float(huber_kappa))
        quadratic = 0.5 * errors * errors / kappa
        linear = abs_e - 0.5 * kappa
        smoothed_abs = torch.where(abs_e <= kappa, quadratic, linear)
        # Sign-aware tilt: tau * e for e > 0, (1 - tau) * |e| for e < 0
        tilt = torch.where(errors >= 0, levels, 1.0 - levels)
        loss = tilt * (2.0 * smoothed_abs)  # 2 * smoothed_abs equals |e| in linear region
    else:
        loss = torch.maximum(levels * errors, (levels - 1.0) * errors)

    if reduction == "none":
        return loss
    if reduction == "mean_quantiles":
        return loss.mean(dim=0)
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unknown reduction: {reduction!r}")


class QuantileLoss(torch.nn.Module):
    """Module wrapper around :func:`quantile_loss` with cached quantile levels.

    Parameters
    ----------
    quantile_levels
        Iterable of quantile levels in ``(0, 1)`` (must match the model head).
    huber_kappa
        See :func:`quantile_loss`.
    """

    def __init__(
        self,
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        huber_kappa: float = 0.0,
    ):
        super().__init__()
        if not all(0.0 < q < 1.0 for q in quantile_levels):
            raise ValueError("All quantile levels must lie strictly in (0, 1).")
        self.register_buffer(
            "quantile_levels",
            torch.tensor(list(quantile_levels), dtype=torch.float32),
            persistent=False,
        )
        self.huber_kappa = float(huber_kappa)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the (optionally weighted) average pinball loss.

        Parameters
        ----------
        predictions
            Quantile predictions ``(Q, *batch)``.
        targets
            Ground-truth ``(*batch,)``.
        weights
            Optional non-negative mask / weight tensor broadcastable to
            ``targets``. Positions with zero weight are excluded from the
            average. If ``None``, an unweighted mean is returned.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        per_element = quantile_loss(
            predictions,
            targets,
            self.quantile_levels,
            huber_kappa=self.huber_kappa,
            reduction="mean_quantiles",
        )

        if weights is None:
            return per_element.mean()

        weights = weights.to(per_element.dtype)
        if weights.shape != per_element.shape:
            weights = weights.expand_as(per_element)
        eps = torch.finfo(per_element.dtype).eps
        total = (per_element * weights).sum()
        denom = weights.sum().clamp_min(eps)
        return total / denom
