# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""World-size-aware unit-scaled functional operations.

Mirrors ``unit_scaling.functional`` but accounts for DDP/FSDP world size
and gradient accumulation when computing batch-dependent scale factors.
Also contains compile-friendly reimplementations of ``residual_split`` and
``residual_add``.

Usage::

    from .dd_unit_scaling import functional as U
    U.silu(x)          # upstream passthrough
    U.linear(x, w)     # our world-size-aware version (overrides upstream)
"""

# Re-export everything from upstream so ``U.silu``, ``U.softmax``, etc. work.
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import unit_scaling.functional as _U_upstream
from unit_scaling.constraints import apply_constraint
from unit_scaling.functional import *  # noqa: F401,F403

from .scale import scale_bwd, scale_fwd


# Global gradient accumulation steps — set once at script startup.
GRAD_ACCUMULATION_STEPS = 1

# Cached world size — defaults to 1 (single-GPU). Call
# init_world_size_cache() before torch.compile in distributed
# settings so that _get_effective_batch_multiplier is a pure-int
# function that dynamo can trace without graph breaks.
_CACHED_WORLD_SIZE: int = 1


def set_grad_accumulation_steps(steps: int) -> None:
    """Set the gradient accumulation steps for unit scaling.

    The effective global batch size is:
        local_batch_size * world_size * accumulate_grad_batches

    Call this once at training startup, before creating the model.
    """
    global GRAD_ACCUMULATION_STEPS
    if steps < 1:
        raise ValueError(f"accumulate_grad_batches must be >= 1, got {steps}")
    GRAD_ACCUMULATION_STEPS = steps


def init_world_size_cache(world_size: int = 1) -> None:
    """Set the cached distributed world size.

    Must be called **before** ``torch.compile`` in distributed settings
    so that ``_get_effective_batch_multiplier`` is a pure-int function
    that dynamo can trace without graph breaks.

    Args:
        world_size: The data-parallel world size. For HSDP, pass the
            replicate dimension size (not total world size).
    """
    global _CACHED_WORLD_SIZE
    _CACHED_WORLD_SIZE = world_size


def _get_effective_batch_multiplier() -> int:
    """Get the multiplier for GLOBAL batch.

    VERIFIED: Both this AND loss correction (* world_size) are needed:
    - This: Makes batch-based scale factors (1/√batch) consistent across GPUs
    - Loss: Undoes DDP's gradient averaging

    For HSDP, uses only the replicate dimension (data-parallel) size.
    The world size is cached (default 1) to avoid graph breaks inside
    torch.compile.  Call ``init_world_size_cache()`` before
    ``torch.compile`` in distributed settings.
    """
    return GRAD_ACCUMULATION_STEPS * _CACHED_WORLD_SIZE


# -------------------------------------------------------------------------
# Residual helpers (compile-friendly drop-ins for unit_scaling.functional)
# -------------------------------------------------------------------------


def residual_split(
    input: torch.Tensor, tau: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split into (residual, skip) with τ-weighted backward scaling.

    Compile-friendly drop-in for ``unit_scaling.functional.residual_split``.
    """
    denom = (1 + tau**2) ** 0.5
    residual = scale_bwd(input, tau / denom)
    skip = scale_bwd(input, 1 / denom)
    return residual, skip


def residual_add(
    residual: torch.Tensor, skip: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """Combine residual + skip with τ-weighted forward scaling.

    Compile-friendly drop-in for ``unit_scaling.functional.residual_add``.
    """
    denom = (1 + tau**2) ** 0.5
    residual = scale_fwd(residual, tau / denom)
    skip = scale_fwd(skip, 1 / denom)
    return residual + skip


# -------------------------------------------------------------------------
# Magnitude-preserving residual (Karras et al., EDM2 / SDXL CONFIG D-G)
# -------------------------------------------------------------------------
#
# Activation-RMS-preserving residual stream: y = (skip + α·residual) /
# sqrt(1 + α²).  Unlike u-μP's τ-rule (which only normalizes *gradient*
# magnitudes via scale_bwd), this normalizes *forward* activation
# magnitudes — the FFN/attention sub-block can have any output magnitude
# and the residual stream RMS stays bounded.
#
# Why we want it for the EEG/Toto2 trunk-collapse problem
# -------------------------------------------------------
# In exp24/25 we observed FFN weights collapsing to fp32 underflow
# even though val_loss kept improving — the model relied entirely on the
# scaler/output-head and the trunk drifted toward zero unchecked.
# Magnitude-preserving residual gives the trunk a finite "magnitude
# budget": small FFN weights translate to small residual contribution
# but never zero out the stream.
#
# Notes
# -----
# * No learnable parameters are introduced — α is a fixed scalar
#   (default 1.0, matching the symmetric SDXL choice).  Karras EDM2
#   uses α=0.3 in encoder/decoder blocks, α=0.5 in the embedding;
#   we expose ``alpha`` per-call so callers can sweep.
# * Equivalent at α=1 to ``residual_add(residual, skip, tau=1)`` in
#   the *forward* pass — but the gradient flow is different: here the
#   gradient is just ``1/sqrt(1+α²)`` w.r.t. both legs (computed by
#   autograd through the divide), with no explicit scale_bwd.  This
#   removes one source of u-μP × FFN-shrinkage interaction.


def mp_residual_split(
    input: torch.Tensor, alpha: float = 1.0  # noqa: ARG001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity split for magnitude-preserving residual streams.

    Returns ``(residual, skip)`` both equal to ``input``. The
    activation-magnitude bookkeeping happens entirely in
    :func:`mp_residual_add`. We expose this as a function (instead of
    inlining the `x_main, x_skip = x, x` line) for symmetry with
    :func:`residual_split` and so callers can swap implementations
    via a single config flag.
    """
    return input, input


def mp_residual_add(
    residual: torch.Tensor, skip: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    r"""Karras-style magnitude-preserving residual combine.

    .. math::
        y = \frac{\text{skip} + \alpha \cdot \text{residual}}
                 {\sqrt{1 + \alpha^2}}

    Keeps ``RMS(y) ≈ RMS(skip)`` regardless of the magnitude of
    ``residual``, on the assumption that ``skip`` and ``residual`` are
    approximately uncorrelated and each has unit RMS.
    """
    denom = (1.0 + alpha * alpha) ** 0.5
    return (skip + alpha * residual) / denom


# -------------------------------------------------------------------------
# Compile-friendly gated activations
# -------------------------------------------------------------------------


def _unscaled_silu(x: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    if mult == 1.0:
        return F.silu(x)
    return x * F.sigmoid(x * mult)


def silu_glu(
    input: torch.Tensor, gate: torch.Tensor, mult: float = 1.0
) -> torch.Tensor:
    """Unit-scaled gated linear unit: ``input * silu(gate)``.

    Compile-friendly reimplementation of ``unit_scaling.functional.silu_glu``
    using our ``scale_fwd`` / ``scale_bwd``.
    """
    alpha = 1.0 / (1.0 + 1.0 / (mult**2.0))
    scale = math.exp(alpha * math.log(2.0**0.5) + (1.0 - alpha) * math.log(2.0))
    input = scale_bwd(input, scale)
    gate = scale_bwd(gate, scale)
    output = input * _unscaled_silu(gate, mult=mult)
    return scale_fwd(output, scale)


# -------------------------------------------------------------------------
# World-size-aware functional operations
# -------------------------------------------------------------------------


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    constraint: Optional[str] = "to_output_scale",
    scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """World-size-aware unit-scaled linear transformation.

    Uses the original unit_scaling pattern:
    - output_scale = 1/fan_in^scale_power[0] (forward scaling)
    - grad_input_scale = 1/fan_out^scale_power[1] (counteracts sqrt(fan_out) amplification)
    - grad_weight_scale = 1/batch_size^scale_power[2]

    With constraint="to_output_scale", grad_input_scale is overridden to match output_scale.
    """
    fan_out, fan_in = weight.shape
    effective_multiplier = _get_effective_batch_multiplier()
    global_numel = input.numel() * effective_multiplier
    batch_size = global_numel // fan_in

    output_scale = 1.0 / fan_in ** scale_power[0]
    grad_input_scale = 1.0 / fan_out ** scale_power[1]  # Uses fan_OUT per unit_scaling
    grad_weight_scale = grad_bias_scale = 1.0 / batch_size ** scale_power[2]

    # Apply constraint if specified
    if constraint is not None:
        output_scale, grad_input_scale = apply_constraint(
            constraint, output_scale, grad_input_scale
        )

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.linear(input, weight, bias)
    return scale_fwd(output, output_scale)


def rms_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Accumulation-aware unit-scaled RMS normalization."""
    if weight is not None:
        effective_multiplier = _get_effective_batch_multiplier()
        global_numel = input.numel() * effective_multiplier
        scale = math.sqrt(math.prod(normalized_shape) / global_numel)
        weight = scale_bwd(weight, scale)
    return _U_upstream._unscaled_rms_norm(input, normalized_shape, weight, eps=eps)


def softplus(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    constraint: Optional[str] = None,
) -> torch.Tensor:
    """Unit-scaled softplus.

    Empirically calibrated so that for standard-normal input the forward
    and backward scales are ~1.  The constants below were measured at
    ``mult=1`` (see ``toto/model/util.py`` for derivation).
    """
    y_scale = 1.0 / 0.52103
    grad_input_scale = 1.0 / 0.20833444

    if constraint is not None:
        y_scale, grad_input_scale = apply_constraint(
            constraint, y_scale, grad_input_scale
        )

    x = scale_bwd(x, grad_input_scale)
    output = F.softplus(x, beta=beta, threshold=threshold)
    return scale_fwd(output, y_scale)


def per_dim_scale(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Accumulation-aware unit-scaled per-dimension scaling.

    Elementwise-multiplies ``input`` by ``weight`` (broadcast along the last
    dim), with batch-dependent gradient scaling on ``weight`` — same pattern
    as RMSNorm weight scaling.

    The output and grad_input scales compensate for the unit-scaled softplus
    forward scaling so that per_dim_scale is identity at init (params=0).

    Args:
        input: Activation tensor, typically (B, H, S, D).
        weight: Per-dimension scale factors, shape (D,).
    """
    # 0.52103 = 1 / y_scale from uu.softplus, compensating so that
    # softplus(0)/log(2) * output_scale = 1.0 (identity at init).
    output_scale = 0.52103
    grad_input_scale = 0.52103
    effective_multiplier = _get_effective_batch_multiplier()
    global_numel = input.numel() * effective_multiplier
    grad_scale = math.sqrt(input.shape[-1] / global_numel)
    weight = scale_bwd(weight, grad_scale)
    input = scale_bwd(input, grad_input_scale)
    return scale_fwd(input * weight, output_scale)
