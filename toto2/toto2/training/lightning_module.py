# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""PyTorch Lightning module for Toto 2.0 pre-training.

Implements next-patch quantile prediction with the same supervision used in
the Toto paper (Cohen et al., 2025) but adapted to Toto 2.0's quantile head:

- **Inputs** at every patch position pass through ``PatchedCausalStdScaler``
  ➜ ``asinh`` ➜ patch projection ➜ decoder stack.
- **Targets** are the same window shifted forward by one ``patch_size``
  (``series[patch_size : context_length + patch_size]``).
- **Loss** is the average pinball loss across the 9 quantile knots,
  computed in *scaled / asinh* space for stability — the same space the
  output head predicts in.

The module is u-μP-aware: it caches MuP fan values before optimizer
construction (and before any FSDP wrapping or ``torch.compile`` in the
training script) so that the ``dd_unit_scaling.AdamW`` / ``Dion2`` /
``NorMuon`` optimizers can apply the correct per-parameter LR scaling.

Both *from-scratch pre-training* and *continued pre-training* are supported
via two construction paths:

- ``Toto2ForTraining(config=...)`` — builds a fresh ``Toto2Model`` (random
  init).
- ``Toto2ForTraining.from_pretrained(model_id=...)`` — loads a published
  Toto 2.0 checkpoint and resumes training.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Sequence

import lightning as L
import torch
from einops import rearrange, reduce
from lightning.pytorch.utilities import grad_norm

import dd_unit_scaling as uu

from ..configuration import Toto2ModelConfig
from ..model import Toto2Model
from .losses import QuantileLoss
from .scheduler import WarmupStableDecayLR


OptimizerName = Literal["adamw", "normuon", "dion2"]


def _build_residual_attn_ratio(
    config_dict: Dict[str, Any],
    context_length: int,
) -> Dict[str, Any]:
    """Compute ``residual_attn_ratio`` from context length if not provided.

    Toto 2.0 requires this field to balance attention/MLP variance when the
    SDPA call is unscaled. See ``dd_unit_scaling/README.md`` for details.
    """
    cfg = dict(config_dict)
    if cfg.get("residual_attn_ratio") is None:
        cfg["residual_attn_ratio"] = Toto2ModelConfig.compute_residual_attn_ratio(
            context_length=int(context_length),
            patch_size=int(cfg["patch_size"]),
        )
    return cfg


class Toto2ForTraining(L.LightningModule):
    """Lightning wrapper around :class:`toto2.model.Toto2Model` for training.

    Parameters
    ----------
    config
        Either a :class:`Toto2ModelConfig` instance or a dict of model
        hyperparameters. If a dict is passed and ``residual_attn_ratio`` is
        ``None``, it is computed automatically from ``context_length`` and
        ``patch_size``.
    pretrained_model
        Optional Toto 2.0 ``Toto2Model`` instance to start training from
        (used for continued pre-training). When provided, ``config`` is
        ignored and the model's own config is used.
    context_length
        Number of input timesteps fed to the model. Must be a multiple of
        ``patch_size``. Used both for ``residual_attn_ratio`` derivation
        and for slicing batches.
    base_lr
        Peak learning rate at the end of warmup. With u-μP this transfers
        across model widths, so the same value works at all sizes.
    min_lr
        Floor learning rate at the end of decay (and during the post-decay
        plateau).
    warmup_steps / stable_steps / decay_steps
        Phase lengths of the WSD schedule, measured in optimizer steps.
    weight_decay
        AdamW weight decay (independent — applied before LR scaling).
    betas
        ``(beta1, beta2)`` for the AdamW / NorMuon / Dion2 update rules.
    optimizer_name
        ``"adamw"`` (recommended baseline), ``"normuon"`` or ``"dion2"``
        (Muon-family; require ``dion``).
    quantile_levels
        Iterable of quantile levels matching the Toto 2.0 head. Default
        ``[0.1, …, 0.9]`` (the 9 knots used by the public checkpoints).
    huber_kappa
        Huber-smoothing width for the pinball loss. ``0.0`` recovers the
        exact pinball; ``0.05`` is a good default for noisy domains like
        EEG to suppress kink-driven gradient noise at very small errors.
    log_grad_norm
        If ``True``, log the global L2 grad norm before each optimizer step.
    """

    def __init__(
        self,
        config: Optional[Toto2ModelConfig | Dict[str, Any]] = None,
        *,
        pretrained_model: Optional[Toto2Model] = None,
        context_length: int = 4096,
        base_lr: float = 5e-4,
        min_lr: float = 1e-5,
        warmup_steps: int = 1000,
        stable_steps: int = 50_000,
        decay_steps: int = 5_000,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer_name: OptimizerName = "adamw",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        huber_kappa: float = 0.0,
        log_grad_norm: bool = True,
    ) -> None:
        super().__init__()

        # Resolve / build the underlying Toto 2.0 model
        if pretrained_model is not None:
            self.model = pretrained_model
        elif config is not None:
            if isinstance(config, dict):
                cfg = _build_residual_attn_ratio(config, context_length)
                model_config = Toto2ModelConfig(**cfg)
            elif isinstance(config, Toto2ModelConfig):
                model_config = config
            else:
                raise TypeError(f"Unsupported config type: {type(config)!r}")
            self.model = Toto2Model(model_config)
        else:
            raise ValueError("Pass either `config=...` or `pretrained_model=...`.")

        self.context_length = int(context_length)
        if self.context_length % self.model.config.patch_size != 0:
            raise ValueError(
                "context_length must be a multiple of patch_size; "
                f"got context_length={self.context_length}, patch_size={self.model.config.patch_size}."
            )

        # Quantile loss (matches the head's quantile knots)
        if list(quantile_levels) != list(self.model.output_head.knots):
            raise ValueError(
                "quantile_levels must match the model output head knots: "
                f"got {list(quantile_levels)}, head expects {self.model.output_head.knots}."
            )
        self.loss_fn = QuantileLoss(quantile_levels=quantile_levels, huber_kappa=huber_kappa)

        # Optimizer / schedule hyperparameters
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.weight_decay = float(weight_decay)
        self.betas = (float(betas[0]), float(betas[1]))
        self.optimizer_name = optimizer_name
        self.log_grad_norm = bool(log_grad_norm)

        # Save hyperparameters (Lightning checkpoint metadata). We deliberately
        # avoid serializing the pre-built model object.
        hparams = {
            "context_length": self.context_length,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "stable_steps": self.stable_steps,
            "decay_steps": self.decay_steps,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "optimizer_name": self.optimizer_name,
            "huber_kappa": float(huber_kappa),
            "quantile_levels": list(quantile_levels),
            "model_config": asdict(self.model.config),
        }
        self.save_hyperparameters(hparams)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        series_ids: torch.Tensor,
        cpm_mask: Optional[torch.Tensor] = None,
        num_return_steps: Optional[int] = None,
    ):
        """Run a single training/eval forward pass.

        Parameters
        ----------
        target
            ``(B, V, T)`` raw values fed to the patch-causal scaler. ``T``
            should equal ``context_length`` for the standard supervision
            recipe in :meth:`_step`.
        target_mask
            ``(B, V, T)`` boolean mask — ``True`` indicates an observed
            value, ``False`` indicates a missing/pad position.
        series_ids
            ``(B, V)`` integer ids that group variates for variate-axis
            attention. Padding variates carry ``-1``.
        cpm_mask
            Optional causal-patch mask. If ``None`` we use ``target_mask``,
            i.e., the scaler accumulates statistics over every observed
            position — the standard auto-regressive next-patch recipe.
        num_return_steps
            Optional restriction to the trailing ``num_return_steps`` patch
            tokens (mainly useful for evaluation; ``None`` returns all).

        Returns
        -------
        toto2.model.Toto2ModelOutputs
            ``(quantiles, loc, scale)``.
        """
        if cpm_mask is None:
            cpm_mask = target_mask
        return self.model(
            target=target,
            target_mask=target_mask,
            cpm_mask=cpm_mask,
            series_ids=series_ids,
            num_return_steps=num_return_steps,
        )

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute next-patch quantile loss on a collated batch.

        The recipe mirrors LLM-style next-token training, lifted to patches:
        the input is ``series[0 : context_length]`` and the target at patch
        token ``i`` is ``series[(i+1)*P : (i+2)*P]``. The model's quantile
        outputs at each token directly predict the *next* patch.

        Loss is evaluated in scaled + asinh space — the same space the
        output head predicts in — so we reproduce the inference-time
        un-asinh / un-scale operations only at evaluation, never during
        training (matching ``forecast()``'s ``block_q.sinh() * scale + loc``).
        """
        target = batch["target"]              # (B, V, context_length + patch_size)
        target_mask = batch["target_mask"]    # (B, V, context_length + patch_size)
        series_ids = batch["series_ids"]      # (B, V)

        P = self.model.config.patch_size
        if target.shape[-1] != self.context_length + P:
            raise ValueError(
                f"Expected window of length context_length + patch_size "
                f"= {self.context_length + P}, but got {target.shape[-1]}."
            )

        # Split into model input (length context_length) and shifted ground truth
        input_target = target[..., : self.context_length]
        input_mask = target_mask[..., : self.context_length]
        gt_next = target[..., P : self.context_length + P]
        gt_mask = target_mask[..., P : self.context_length + P]

        # Re-key padding variates (series_id == -1) so loss isn't computed on them.
        valid_variate = (series_ids != -1).unsqueeze(-1)  # (B, V, 1)
        loss_mask = gt_mask & valid_variate

        # Run the model on the *input* portion only — this is the same
        # forward pass that produces quantile predictions during inference.
        outputs = self.forward(
            target=input_target,
            target_mask=input_mask,
            series_ids=series_ids,
        )
        quantiles = outputs.quantiles  # (Q, B, V, n_patches, P)
        loc = outputs.loc              # (B, V, context_length)
        scale = outputs.scale          # (B, V, context_length)

        # Normalize ground-truth next patches with the same loc/scale used
        # by the model on the input. ``loc`` / ``scale`` repeat across each
        # patch (last-step pooling in the scaler), so we can index any
        # position within a patch — we choose the first.
        loc_per_patch = rearrange(loc, "b v (s p) -> b v s p", p=P)[..., 0]
        scale_per_patch = rearrange(scale, "b v (s p) -> b v s p", p=P)[..., 0]
        gt_per_patch = rearrange(gt_next, "b v (s p) -> b v s p", p=P)
        gt_mask_per_patch = rearrange(loss_mask, "b v (s p) -> b v s p", p=P)

        eps = torch.finfo(gt_per_patch.dtype).eps
        scaled_gt = (gt_per_patch - loc_per_patch.unsqueeze(-1)) / (scale_per_patch.unsqueeze(-1) + eps)
        scaled_gt = torch.where(gt_mask_per_patch, scaled_gt, torch.zeros_like(scaled_gt))
        asinh_gt = torch.asinh(scaled_gt)  # supervision lives in asinh space

        # The model's num_output_patches multiplies the patch dimension of
        # the output head; we keep training simple by requiring nop=1.
        nop = self.model.config.num_output_patches
        if nop != 1:
            raise NotImplementedError(
                "This training module currently supports num_output_patches=1. "
                "Multi-step output heads can be added later by aligning targets accordingly."
            )

        # Compute pinball loss only over patches that have any observed
        # target steps; further weight by per-step observation mask.
        # quantiles[..., n_patches, P] aligns 1:1 with asinh_gt[..., n_patches, P].
        loss = self.loss_fn(quantiles, asinh_gt, weights=gt_mask_per_patch.to(quantiles.dtype))

        # ---- diagnostics ----
        observed_frac = gt_mask_per_patch.float().mean()
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        self.log(
            f"{stage}_observed_frac",
            observed_frac,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    # ------------------------------------------------------------------
    # Optimizer / schedule (u-μP-aware)
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Cache u-μP fan values before optimizer construction.

        ``dd_unit_scaling`` requires ``cache_fan_values`` to be called *before*
        FSDP wrapping (which replaces parameter tensors with DTensors and
        loses the original fan-in). Calling it here in ``setup`` runs after
        DDP/FSDP rank initialisation but before
        :meth:`configure_optimizers`.
        """
        super().setup(stage)
        try:
            uu.cache_fan_values(self.model.named_parameters())
        except Exception:
            # If running without unit-scaling-tagged params, optimizer falls
            # back to standard scaling. We tolerate this for compatibility.
            pass

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_name == "adamw":
            optimizer = uu.AdamW(
                params,
                lr=self.base_lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                independent_weight_decay=True,
                allow_non_unit_scaling_params=True,
            )
        elif self.optimizer_name == "normuon":
            optimizer = uu.NorMuon(  # type: ignore[call-arg]
                params,
                lr=self.base_lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                allow_non_unit_scaling_params=True,
            )
        elif self.optimizer_name == "dion2":
            optimizer = uu.Dion2(  # type: ignore[call-arg]
                params,
                lr=self.base_lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                allow_non_unit_scaling_params=True,
            )
        else:
            raise ValueError(f"Unknown optimizer_name: {self.optimizer_name!r}")

        scheduler = WarmupStableDecayLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            stable_steps=self.stable_steps,
            decay_steps=self.decay_steps,
            min_lr=self.min_lr,
            base_lr=self.base_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.log_grad_norm:
            try:
                norms = grad_norm(self.model, norm_type=2)
                if "grad_2.0_norm_total" in norms:
                    self.log(
                        "grad_norm",
                        norms["grad_2.0_norm_total"],
                        prog_bar=False,
                        on_step=True,
                        sync_dist=False,
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        context_length: int = 4096,
        map_location: str = "cpu",
        **trainer_kwargs: Any,
    ) -> "Toto2ForTraining":
        """Load a Toto 2.0 checkpoint and wrap it for continued pre-training.

        Parameters
        ----------
        model_id
            Either a HuggingFace repo id (e.g. ``"Datadog/Toto-2.0-22m"``) or
            a local path containing ``config.json`` plus a ``safetensors``
            checkpoint.
        context_length
            Context length to use for training (must be compatible with the
            checkpoint's RoPE max length, which is 8192 for the public
            Toto 2.0 family).
        map_location
            Device to load weights onto initially.
        trainer_kwargs
            Forwarded to :meth:`__init__` (LR, schedule, optimizer, …).
        """
        model = Toto2Model.from_pretrained(model_id, map_location=map_location)
        # Continue training requires gradients enabled (HF ``from_pretrained``
        # leaves ``requires_grad=True`` by default but we restore train mode).
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        return cls(pretrained_model=model, context_length=context_length, **trainer_kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_input_patches(self) -> int:
        """Number of patch tokens emitted for an input window of ``context_length``."""
        return self.context_length // self.model.config.patch_size

    def patch_token_observed_fraction(self, target_mask: torch.Tensor) -> torch.Tensor:
        """Fraction of observed steps within each patch token (``B, V, S``)."""
        P = self.model.config.patch_size
        return reduce(target_mask.float(), "b v (s p) -> b v s", "mean", p=P)
