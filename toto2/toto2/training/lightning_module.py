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

Constructed via ``Toto2ForTraining(config=...)`` which builds a fresh
``Toto2Model`` (random init). Continue-pretraining from a published
checkpoint is no longer supported — use the HF inference loader
(``Toto2Model.from_pretrained``) for zero-shot evaluation only.
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
from .amse_loss import AMSELoss
from .auxiliary_losses import (
    JEPAHead,
    MultiResolutionSTFTAuxLoss,
    PARSHead,
    amplitude_aware_cpm_mask,
    online_denoise_target,
    phase_aware_fourier_loss,
)
from .losses import QuantileLoss
from .scheduler import WarmupStableDecayLR


OptimizerName = Literal["adamw", "normuon", "dion2"]
LossType = Literal["pinball", "amse"]


# =====================================================================
# exp43 AMSE loss configuration helpers
# =====================================================================
#
# AMSE (Subich et al. ICML 2025, arXiv:2501.19374) replaces the pinball
# loss with a spectrally-decomposed alternative that separates amplitude
# error from phase / coherence error.  See ``amse_loss.py`` for the
# math; the lightning module just plumbs the config through.
#
# The hypothesis we test is whether the v2/v3 trunk-collapse pathology
# (FFN weights underflow, val_pinball stays low because the head learns
# a smart causal-scaler shortcut) is driven by pinball's inability to
# distinguish amplitude smoothing from phase error.  AMSE makes the two
# error sources independent so the gradient cannot win by smoothing.


_DEFAULT_AMSE = {
    "concat_patches": True,           # FFT over the full S·P sequence
    "spectral_weight_exponent": 0.0,  # 0 = uniform; 2/3 = EEG 1/f comp.
    "pinball_calibration_weight": 0.0,  # optional small pinball term to
                                        # keep the off-median knots
                                        # calibrated; 0 = pure AMSE
                                        # (recommended for the principled
                                        # exp43 ablation).
}


def _merge_amse_config(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(_DEFAULT_AMSE)
    if not user:
        return out
    unknown = set(user) - set(out)
    if unknown:
        raise ValueError(
            f"Unknown amse keys {sorted(unknown)!r}; expected one of {list(out)}."
        )
    out.update(user)
    return out


# =====================================================================
# exp27 auxiliary supervision configuration helpers
# =====================================================================
#
# The lightning module accepts a single ``auxiliary`` dict that
# enumerates which probes are active and the per-probe hyperparameters.
# This lets a single yaml config switch any combination on/off without
# touching the python — exactly the pattern exp24/25/26 used for the
# trunk fixes.  Defaults are conservative (probe disabled).


_DEFAULT_AUX = {
    "jepa": {
        "enabled": False,
        "weight": 0.5,
        "sigreg_weight": 0.02,
        "proj_dim": 256,
        "ema_tau": 0.999,
        "sigreg_num_slices": 1024,
        "sigreg_n_quad_points": 17,
    },
    "aamp": {
        "enabled": False,
        "mask_ratio_range": (0.20, 0.40),
    },
    "pars": {
        "enabled": False,
        "weight": 0.1,
        "num_pairs": 12,
        "head_dim": 64,
        "mlp_hidden": 512,
    },
    "phase": {
        "enabled": False,
        "weight": 0.1,
    },
    "mrstft": {
        "enabled": False,
        "weight": 0.1,
        "fft_sizes": (32, 128, 512),
        "hop_sizes": (8, 32, 128),
        "win_sizes": None,  # defaults to fft_sizes
    },
    "denoise": {
        "enabled": False,
        "low_hz": 1.0,
        "high_hz": 50.0,
        "sfreq": 500.0,
        "clip_sigma": 6.0,
        "detrend": True,
    },
}


def _merge_aux_config(user: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in _DEFAULT_AUX.items()}
    if not user:
        return out
    for k, v in user.items():
        if k not in out:
            raise ValueError(
                f"Unknown auxiliary key {k!r}; expected one of {list(out)}."
            )
        out[k].update(v)
    return out


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
    loss_type
        Primary supervision objective.

        * ``"pinball"`` (default, Toto-2 baseline): mean pinball loss
          across the 9 quantile knots in asinh-scaled space.
        * ``"amse"`` (exp43): :class:`AMSELoss` on the *median* quantile
          prediction (deterministic point forecast in asinh-scaled
          space).  See :data:`_DEFAULT_AMSE` for the configurable knobs
          and ``amse_loss.py`` for the math.  When ``loss_type="amse"``
          the off-median quantile knots receive no direct supervision
          unless ``amse.pinball_calibration_weight > 0``.

        Pinball is always *computed* and logged as ``{stage}_pinball``
        regardless of ``loss_type`` so we can compare new AMSE runs
        against the prior pinball-trained baselines on the same scale.
    amse
        Optional dict overriding :data:`_DEFAULT_AMSE`.  Keys:

        * ``concat_patches`` (bool): concatenate the ``S`` predicted
          patches into a single 1-D signal per recording before FFT.
          ``True`` (default) gives full frequency-range coverage; the
          alternative ``False`` runs per-patch FFT and matches pinball's
          per-patch granularity but loses delta/theta on EEG.
        * ``spectral_weight_exponent`` (float): per-frequency weight
          ``S(k) = (1+k)^exp``.  ``0.0`` (default) is uniform; ``2/3``
          is the EEG 1/f bias correction discussed in the abandoned
          exp43 spec.
        * ``pinball_calibration_weight`` (float): if > 0, add a small
          pinball term on the side knots to keep them calibrated.
          ``0.0`` (default) is the cleanest experimental control.
    """

    def __init__(
        self,
        config: Toto2ModelConfig | Dict[str, Any],
        *,
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
        auxiliary: Optional[Dict[str, Any]] = None,
        loss_type: LossType = "pinball",
        amse: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if isinstance(config, dict):
            cfg = _build_residual_attn_ratio(config, context_length)
            model_config = Toto2ModelConfig(**cfg)
        elif isinstance(config, Toto2ModelConfig):
            model_config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")
        self.model = Toto2Model(model_config)

        self.context_length = int(context_length)
        if self.context_length % self.model.config.patch_size != 0:
            raise ValueError(
                "context_length must be a multiple of patch_size; "
                f"got context_length={self.context_length}, patch_size={self.model.config.patch_size}."
            )

        # Quantile loss (matches the head's quantile knots) — kept around even
        # when ``loss_type == "amse"`` so we can always log ``{stage}_pinball``
        # for cross-comparison with prior pinball-trained baselines.
        if list(quantile_levels) != list(self.model.output_head.knots):
            raise ValueError(
                "quantile_levels must match the model output head knots: "
                f"got {list(quantile_levels)}, head expects {self.model.output_head.knots}."
            )
        self.loss_fn = QuantileLoss(quantile_levels=quantile_levels, huber_kappa=huber_kappa)

        # ----- exp43 AMSE loss (optional) -----
        if loss_type not in ("pinball", "amse"):
            raise ValueError(
                f"loss_type must be 'pinball' or 'amse'; got {loss_type!r}"
            )
        self.loss_type = loss_type
        self.amse_cfg = _merge_amse_config(amse)
        if self.loss_type == "amse":
            self.amse_loss_fn = AMSELoss(
                concat_patches=bool(self.amse_cfg["concat_patches"]),
                spectral_weight_exponent=float(self.amse_cfg["spectral_weight_exponent"]),
            )
        else:
            self.amse_loss_fn = None

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

        # ----- exp27 auxiliary heads / config -----
        self.aux_cfg = _merge_aux_config(auxiliary)

        d_model = self.model.config.d_model
        if self.aux_cfg["jepa"]["enabled"]:
            self.jepa_head = JEPAHead(
                d_model=d_model,
                proj_dim=int(self.aux_cfg["jepa"]["proj_dim"]),
                ema_tau=float(self.aux_cfg["jepa"]["ema_tau"]),
                sigreg_num_slices=int(self.aux_cfg["jepa"]["sigreg_num_slices"]),
                sigreg_n_quad_points=int(self.aux_cfg["jepa"]["sigreg_n_quad_points"]),
            )
        else:
            self.jepa_head = None

        if self.aux_cfg["pars"]["enabled"]:
            self.pars_head = PARSHead(
                d_model=d_model,
                num_pairs=int(self.aux_cfg["pars"]["num_pairs"]),
                head_dim=int(self.aux_cfg["pars"]["head_dim"]),
                mlp_hidden=int(self.aux_cfg["pars"]["mlp_hidden"]),
            )
        else:
            self.pars_head = None

        if self.aux_cfg["mrstft"]["enabled"]:
            mrcfg = self.aux_cfg["mrstft"]
            self.mrstft_loss = MultiResolutionSTFTAuxLoss(
                fft_sizes=tuple(mrcfg["fft_sizes"]),
                hop_sizes=tuple(mrcfg["hop_sizes"]),
                win_sizes=tuple(mrcfg["win_sizes"]) if mrcfg.get("win_sizes") is not None else None,
            )
        else:
            self.mrstft_loss = None

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
            "auxiliary": self.aux_cfg,
            "loss_type": self.loss_type,
            "amse": self.amse_cfg,
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

    # ------------------------------------------------------------------
    # Trunk-representation diagnostics (val-only)
    # ------------------------------------------------------------------

    def _capture_trunk(self):
        """Context manager that records the **pre-out_norm** trunk activation.

        Hooked on ``self.model.transformer.out_norm`` via a *pre-forward*
        hook so we capture the residual-stream tensor *before* the final
        RMSNorm (whose output is unit-RMS by construction and would
        therefore make ``val_trunk_rms`` a tautology).

        Why we want this
        ----------------
        exp24/25 showed val_loss can keep improving even as the trunk
        weights underflow to fp32-zero — the model fell back to its
        causal-scaler + output-head, and the transformer became a
        no-op pass-through.  Effective rank, anisotropy, and **raw
        residual-stream RMS** of the trunk output directly measure
        whether the encoder is using its 384 dimensions or has
        collapsed to a low-rank / low-magnitude subspace.

        Returned object is a context manager; the captured tensor (or
        ``None`` if no forward pass happened) is in
        ``capture.trunk_act`` after exit.
        """

        class _Capture:
            def __init__(self, parent_module: torch.nn.Module):
                self._parent = parent_module
                self._handle = None
                self.trunk_act: Optional[torch.Tensor] = None

            def _pre_hook(self, _module, inputs):
                # ``inputs`` is the tuple ``(residual_stream,)`` about to
                # be fed into out_norm — exactly the post-trunk pre-norm
                # tensor we want.  We only keep the first call so
                # multi-batch decode loops don't overwrite.
                if self.trunk_act is None and inputs:
                    t = inputs[0]
                    if torch.is_tensor(t):
                        self.trunk_act = t.detach()

            def __enter__(self):
                self._handle = self._parent.register_forward_pre_hook(self._pre_hook)
                return self

            def __exit__(self, *exc):
                if self._handle is not None:
                    self._handle.remove()
                    self._handle = None
                return False

        return _Capture(self.model.transformer.out_norm)

    @torch.no_grad()
    def _log_trunk_diagnostics(self, trunk_act: torch.Tensor, batch_size: int) -> None:
        """Effective rank, anisotropy, and RMS of the trunk output.

        Parameters
        ----------
        trunk_act
            Trunk output of shape ``(*lead, V, S, D)``. We flatten to
            ``(N, D)`` where ``N = prod(lead) * V * S``.
        batch_size
            For Lightning's ``log(... batch_size=batch_size)`` accounting.
        """
        x = trunk_act.float()
        if x.ndim < 2:
            return
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]
        if N < 2:
            return

        # Subsample if the batch is huge (eff-rank's SVD is O(N D^2)).
        max_n = 4096
        if N > max_n:
            idx = torch.randperm(N, device=x_flat.device)[:max_n]
            x_flat = x_flat[idx]
            N = max_n

        # Trunk RMS — direct readout of "is the encoder outputting anything?"
        rms = x_flat.pow(2).mean().sqrt()

        # Effective rank (Roy & Vetterli 2007): exp(H(s/Σs)) where s are
        # singular values.  Bounded in [1, D]; we report the ratio to D.
        # Center first to avoid the mean-direction dominating.
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        try:
            sv = torch.linalg.svdvals(x_centered)
        except Exception:
            return
        sv = sv.clamp_min(1e-12)
        p = sv / sv.sum()
        # Shannon entropy in nats; eff-rank = exp(H).
        h = -(p * p.log()).sum()
        eff_rank = h.exp()

        # Anisotropy — average pairwise cosine similarity of L2-normalized
        # tokens (Ethayarajh 2019).  ~0 = uniform on the sphere; ~1 = all
        # tokens collapse onto one direction.
        x_norm = x_flat / x_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # Sample a few random pairs to keep this O(N) instead of O(N^2).
        n_pairs = min(2048, N // 2)
        perm_a = torch.randperm(N, device=x_flat.device)[:n_pairs]
        perm_b = torch.randperm(N, device=x_flat.device)[:n_pairs]
        # Avoid trivial self-pairs by shifting if collisions happen.
        same = perm_a == perm_b
        perm_b = torch.where(same, (perm_b + 1) % N, perm_b)
        cos = (x_norm[perm_a] * x_norm[perm_b]).sum(dim=-1).mean()

        for name, value in [
            ("val_trunk_rms", rms),
            ("val_trunk_eff_rank", eff_rank),
            ("val_trunk_eff_rank_ratio", eff_rank / float(D)),
            ("val_trunk_anisotropy", cos),
        ]:
            self.log(
                name,
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute next-patch quantile loss + auxiliary supervision (exp27 suite).

        The recipe mirrors LLM-style next-token training, lifted to patches:
        the input is ``series[0 : context_length]`` and the target at patch
        token ``i`` is ``series[(i+1)*P : (i+2)*P]``. The model's quantile
        outputs at each token directly predict the *next* patch.

        Loss is evaluated in scaled + asinh space — the same space the
        output head predicts in — so we reproduce the inference-time
        un-asinh / un-scale operations only at evaluation, never during
        training (matching ``forecast()``'s ``block_q.sinh() * scale + loc``).

        On top of pinball, exp27 layers in optional auxiliary supervision
        controlled by ``self.aux_cfg`` (see :data:`_DEFAULT_AUX`):

        * **B-AAMP** (``aamp``) — modify ``cpm_mask`` to drop top-K%
          amplitude patches before the scaler / patch_proj see them.
        * **F-Denoise** (``denoise``) — replace the pinball *target*
          (not the input) with a band-passed, sigma-clipped, detrended
          version so the model has to denoise as it predicts.
        * **A-JEPA** (``jepa``) — predict next-patch latents of an EMA
          target encoder; SIGReg the projected context.
        * **C-PARS** (``pars``) — predict the antisymmetric NxN matrix
          of pairwise relative shifts on a random subset of patches.
        * **D-Phase** (``phase``) — Fourier-domain log-amp + sin/cos-φ
          MSE on the median-knot prediction.
        * **E-MRSTFT** (``mrstft``) — multi-resolution STFT loss on the
          median-knot prediction.
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

        # ----- B-AAMP: modify the causal-patch mask to drop high-amplitude patches -----
        # The base ``input_mask`` still defines what counts as observed;
        # AAMP only changes what the **scaler / patch_proj** see, by
        # dropping a randomly-sized fraction of the highest-amplitude
        # patches.  Pinball is still computed on every observed gt
        # position.  The ``no_grad`` decorator on ``amplitude_aware_cpm_mask``
        # ensures gradients don't flow through the percentile thresholding.
        if stage == "train" and self.aux_cfg["aamp"]["enabled"]:
            cpm_mask = amplitude_aware_cpm_mask(
                input_target,
                input_mask,
                patch_size=P,
                mask_ratio_range=tuple(self.aux_cfg["aamp"]["mask_ratio_range"]),
            )
        else:
            cpm_mask = input_mask

        # Re-key padding variates (series_id == -1) so loss isn't computed on them.
        valid_variate = (series_ids != -1).unsqueeze(-1)  # (B, V, 1)
        loss_mask = gt_mask & valid_variate

        # Run the model on the *input* portion only — this is the same
        # forward pass that produces quantile predictions during inference.
        # We always capture the post-trunk activation when JEPA or PARS
        # is active (their auxiliary losses operate on the trunk
        # output).  At validation we capture for the trunk-health
        # diagnostics regardless.
        capture_trunk = (
            stage == "val"
            or (stage == "train" and (self.aux_cfg["jepa"]["enabled"] or self.aux_cfg["pars"]["enabled"]))
        )
        if capture_trunk:
            with self._capture_trunk() as cap:
                outputs = self.forward(
                    target=input_target,
                    target_mask=input_mask,
                    series_ids=series_ids,
                    cpm_mask=cpm_mask,
                )
            trunk_act = cap.trunk_act
        else:
            outputs = self.forward(
                target=input_target,
                target_mask=input_mask,
                series_ids=series_ids,
                cpm_mask=cpm_mask,
            )
            trunk_act = None
        quantiles = outputs.quantiles  # (Q, B, V, n_patches, P)
        loc = outputs.loc              # (B, V, context_length)
        scale = outputs.scale          # (B, V, context_length)

        # Normalize ground-truth next patches with the same loc/scale used
        # by the model on the input. ``loc`` / ``scale`` repeat across each
        # patch (last-step pooling in the scaler), so we can index any
        # position within a patch — we choose the first.
        loc_per_patch = rearrange(loc, "b v (s p) -> b v s p", p=P)[..., 0]
        scale_per_patch = rearrange(scale, "b v (s p) -> b v s p", p=P)[..., 0]

        # ----- F-Denoise: replace the pinball target with a denoised version -----
        # (input is unchanged; only the target the loss compares against
        # is run through the simple online filter chain in
        # online_denoise_target.)
        if stage == "train" and self.aux_cfg["denoise"]["enabled"]:
            dcfg = self.aux_cfg["denoise"]
            # Concat input + gt_next so the IIR sees a continuous signal,
            # then slice off the gt portion afterwards.  This avoids a
            # transient at the seam.
            concat = torch.cat([input_target, gt_next], dim=-1)
            concat_mask = torch.cat([input_mask, gt_mask], dim=-1)
            denoised = online_denoise_target(
                concat,
                concat_mask,
                sfreq=float(dcfg["sfreq"]),
                low_hz=float(dcfg["low_hz"]),
                high_hz=float(dcfg["high_hz"]),
                clip_sigma=float(dcfg["clip_sigma"]),
                detrend=bool(dcfg["detrend"]),
            )
            denoise_gt_next = denoised[..., self.context_length :]
        else:
            denoise_gt_next = gt_next

        gt_per_patch = rearrange(denoise_gt_next, "b v (s p) -> b v s p", p=P)
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
        # We always compute pinball, even under loss_type="amse", so it can be
        # logged for cross-comparison with prior baselines (it just doesn't
        # backprop in the AMSE case unless pinball_calibration_weight > 0).
        if self.loss_type == "amse":
            with torch.no_grad():
                pinball_loss = self.loss_fn(
                    quantiles,
                    asinh_gt,
                    weights=gt_mask_per_patch.to(quantiles.dtype),
                )
        else:
            pinball_loss = self.loss_fn(
                quantiles,
                asinh_gt,
                weights=gt_mask_per_patch.to(quantiles.dtype),
            )

        # ----- exp43 AMSE primary loss (when enabled) ---------------------
        # Replaces pinball with a spectrally-decomposed loss (Subich et al.,
        # ICML 2025) on the *median* quantile prediction.  See ``amse_loss.py``
        # for the math; in short:
        #   AMSE(x, y) = Σ_k (√PSD_k(x) − √PSD_k(y))² + 2·max(PSD_k(x), PSD_k(y))·(1 − Coh_k)
        # which separates amplitude error from phase error so the model can't
        # win by smoothing.  Tests the hypothesis that pinball's amplitude/
        # phase entanglement drives the v2/v3 trunk-collapse pathology.
        amse_log: Dict[str, torch.Tensor] = {}
        if self.loss_type == "amse":
            median_idx = self._knot_indices_for(0.5)[0]
            median_pred = quantiles[median_idx]  # (B, V, S, P) in asinh-scaled space
            amse_loss_value, amse_diags = self.amse_loss_fn(
                pred=median_pred,
                target=asinh_gt,
                mask=gt_mask_per_patch,
                patch_size=P,
                return_diagnostics=True,
            )
            total_loss = amse_loss_value
            calib_w = float(self.amse_cfg["pinball_calibration_weight"])
            if calib_w > 0.0:
                # Re-compute pinball with grad so the side-knot calibration term
                # contributes to the optimizer step.  Replaces the no_grad call
                # above for the "hybrid AMSE + small pinball" recipe.
                pinball_loss_with_grad = self.loss_fn(
                    quantiles,
                    asinh_gt,
                    weights=gt_mask_per_patch.to(quantiles.dtype),
                )
                pinball_loss = pinball_loss_with_grad
                total_loss = total_loss + calib_w * pinball_loss_with_grad
            amse_log["amse"] = amse_loss_value.detach()
            for k, v in amse_diags.items():
                amse_log[k] = v
        else:
            total_loss = pinball_loss

        # ---- exp27 auxiliary losses ----
        # Each is conditionally applied if its config flag is enabled.
        # Their per-loss values + diagnostics are logged with `stage` prefix
        # for direct comparison to pinball.
        aux_log: Dict[str, torch.Tensor] = {}
        if stage == "train":
            if self.jepa_head is not None:
                # ``trunk_act`` is captured above; pass through JEPA head.
                # The JEPA target tower runs internally inside JEPAHead.
                # ``valid_seq_mask`` is True at positions where the gt patch
                # is observed AND the variate is real.
                valid_seq = (gt_mask_per_patch.float().mean(dim=-1) > 0) & valid_variate
                jepa_l, sigreg_l, jepa_metrics = self.jepa_head(
                    context_trunk_act=trunk_act,
                    target_input=input_target,
                    target_input_mask=input_mask,
                    target_input_cpm_mask=cpm_mask,
                    series_ids=series_ids,
                    patch_size=P,
                    global_step=int(self.global_step),
                    valid_loss_mask=valid_seq,
                )
                jepa_w = float(self.aux_cfg["jepa"]["weight"])
                sigreg_w = float(self.aux_cfg["jepa"]["sigreg_weight"])
                total_loss = total_loss + jepa_w * jepa_l + sigreg_w * sigreg_l
                aux_log["jepa_loss"] = jepa_l.detach()
                aux_log["sigreg_loss"] = sigreg_l.detach()
                for k, v in jepa_metrics.items():
                    aux_log[k] = v

            if self.pars_head is not None:
                valid_seq = (gt_mask_per_patch.float().mean(dim=-1) > 0) & valid_variate
                pars_l, pars_metrics = self.pars_head(
                    trunk_act,
                    valid_seq_mask=valid_seq,
                )
                pars_w = float(self.aux_cfg["pars"]["weight"])
                total_loss = total_loss + pars_w * pars_l
                aux_log["pars_loss"] = pars_l.detach()
                for k, v in pars_metrics.items():
                    aux_log[k] = v

            if self.aux_cfg["phase"]["enabled"]:
                phase_l, phase_metrics = phase_aware_fourier_loss(
                    quantiles=quantiles,
                    asinh_gt=asinh_gt,
                    weights=gt_mask_per_patch,
                    median_idx=self._knot_indices_for(0.5)[0],
                )
                phase_w = float(self.aux_cfg["phase"]["weight"])
                total_loss = total_loss + phase_w * phase_l
                aux_log["phase_loss"] = phase_l.detach()
                for k, v in phase_metrics.items():
                    aux_log[k] = v

            if self.mrstft_loss is not None:
                stft_l, stft_metrics = self.mrstft_loss(
                    quantiles=quantiles,
                    asinh_gt=asinh_gt,
                    weights=gt_mask_per_patch,
                    median_idx=self._knot_indices_for(0.5)[0],
                )
                stft_w = float(self.aux_cfg["mrstft"]["weight"])
                total_loss = total_loss + stft_w * stft_l
                aux_log["mrstft_loss"] = stft_l.detach()
                for k, v in stft_metrics.items():
                    aux_log[k] = v

        # ---- diagnostics ----
        observed_frac = gt_mask_per_patch.float().mean()
        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        self.log(
            f"{stage}_pinball",
            pinball_loss,
            prog_bar=False,
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

        # Auxiliary diagnostics (train only — they all condition on `stage == 'train'`).
        for k, v in aux_log.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=target.shape[0],
            )

        # AMSE diagnostics — logged every step when AMSE is enabled (both train
        # and val).  These let us see at a glance whether the spectral
        # decomposition is producing the intended signal (amp_term ≈ 0 means
        # the model has the right amplitudes; coh_term ≈ 0 means it has the
        # right phases; amp_ratio ≈ 1 means the spectrum magnitude matches).
        for k, v in amse_log.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(k == "amse"),
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=target.shape[0],
            )

        # Quantile-head health: spread across the 9 knots. If this collapses to ~0
        # the head has degenerated to a constant predictor and pinball loss
        # plateaus near 0.33–0.40 even though the trunk could carry more signal.
        # Cheap to compute (one std over the 9-knot axis), surfaces collapse
        # before a full epoch elapses.
        with torch.no_grad():
            knot_spread = quantiles.std(dim=0).mean()
            self.log(
                f"{stage}_knot_spread",
                knot_spread,
                prog_bar=False,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=target.shape[0],
            )

            # ---------------- richer probabilistic-forecast metrics ------------
            # Only at validation: these add ~no overhead but give us a clean
            # signal even when val_loss is barely moving.  Motivated by the
            # post-exp25 finding that val_loss kept improving as the FFN
            # trunk drifted to fp32 underflow — we need metrics that can
            # tell apart "model is genuinely calibrating" from "model is
            # collapsing onto its causal-scaler+head fallback".
            if stage == "val":
                self._log_probabilistic_metrics(
                    quantiles=quantiles,
                    asinh_gt=asinh_gt,
                    weights=gt_mask_per_patch,
                    batch_size=target.shape[0],
                )
                if trunk_act is not None:
                    self._log_trunk_diagnostics(
                        trunk_act=trunk_act,
                        batch_size=target.shape[0],
                    )
        return total_loss

    # ------------------------------------------------------------------
    # Probabilistic-forecast diagnostics (val-only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_probabilistic_metrics(
        self,
        quantiles: torch.Tensor,
        asinh_gt: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Log CRPS, quantile coverage, interval coverage, MASE-proxy.

        All metrics are computed in *asinh / scaled* space, the same space
        the model is trained in.  This is faster than reversing the asinh
        and `static_loc/scale` and — crucially — keeps the metrics on a
        bounded scale (raw asinh-EEG can have outliers > 13 σ which dominate
        unscaled MAE/MSE; we already saw this in v3 smoke).

        We compute:
          • ``val_crps_q``         — quantile-CRPS estimator
            (``2/Q · Σ_τ pinball(y, ŷ_τ)``).  Same number as
            GluonTS' ``mean_weighted_sum_quantile_loss``; close to but
            not identical to true CRPS for finite Q.
          • ``val_cov_50``         — empirical coverage of the 50%
            interval ``[ŷ_0.25_lin_interp, ŷ_0.75_lin_interp]``.
          • ``val_cov_80``         — empirical coverage of the 80%
            interval ``[ŷ_0.1, ŷ_0.9]``.  Should be ~0.80 if the model
            is well-calibrated; <0.80 = over-confident, >0.80 =
            under-confident.
          • ``val_q_cov_<τ>``      — proportion of targets below the
            τ-quantile prediction (should ≈ τ).  Logged for τ ∈
            ``{0.1, 0.5, 0.9}``.
          • ``val_mae_p50``        — MAE at the median quantile,
            scaled-asinh space.  Comparable across runs.
          • ``val_quant_xings``    — fraction of patch positions where
            the predicted quantiles are *not* monotone in τ.  Should
            be ~0; a non-zero value means the head isn't learning a
            valid CDF.
        """
        # quantiles: (Q, B, V, S, P), asinh_gt: (B, V, S, P), weights: (B, V, S, P)
        Q = quantiles.shape[0]
        levels = self.loss_fn.quantile_levels.to(quantiles.dtype).to(quantiles.device)
        levels_view = levels.view(Q, *([1] * (quantiles.ndim - 1)))
        w = weights.to(quantiles.dtype)
        eps = torch.finfo(quantiles.dtype).eps
        denom = w.sum().clamp_min(eps)

        # ----- quantile-CRPS (2/Q · weighted-sum pinball) --------------------
        errors = asinh_gt.unsqueeze(0) - quantiles  # (Q, B, V, S, P)
        pinball = torch.maximum(levels_view * errors, (levels_view - 1.0) * errors)
        # (2/Q) · Σ_τ pinball ≡ "weighted_sum_quantile_loss" used by GIFT-Eval.
        per_pos_crps = pinball.sum(dim=0) * (2.0 / float(Q))  # (B, V, S, P)
        crps = (per_pos_crps * w).sum() / denom

        # ----- per-knot empirical coverage (P(y < ŷ_τ)) -----------------------
        # Linearly comparable to τ if the model is calibrated.
        idx_lo, idx_med, idx_hi = self._knot_indices_for(0.1, 0.5, 0.9)
        q_lo = quantiles[idx_lo]
        q_med = quantiles[idx_med]
        q_hi = quantiles[idx_hi]
        cov_q10 = ((asinh_gt < q_lo).to(quantiles.dtype) * w).sum() / denom
        cov_q50 = ((asinh_gt < q_med).to(quantiles.dtype) * w).sum() / denom
        cov_q90 = ((asinh_gt < q_hi).to(quantiles.dtype) * w).sum() / denom

        # ----- interval coverage (P(y ∈ [ŷ_lo, ŷ_hi])) ------------------------
        in_80 = ((asinh_gt >= q_lo) & (asinh_gt <= q_hi)).to(quantiles.dtype)
        cov_80 = (in_80 * w).sum() / denom
        # 50% interval requires interpolating between 0.4 and 0.6 knots
        # (or 0.3/0.7).  We use 0.3/0.7 to keep things on actual knots.
        idx_30, idx_70 = self._knot_indices_for(0.3, 0.7)
        q_30 = quantiles[idx_30]
        q_70 = quantiles[idx_70]
        in_40 = ((asinh_gt >= q_30) & (asinh_gt <= q_70)).to(quantiles.dtype)
        cov_40 = (in_40 * w).sum() / denom

        # ----- MAE @ p50 in asinh-scaled space --------------------------------
        mae_p50 = ((asinh_gt - q_med).abs() * w).sum() / denom

        # ----- quantile crossings ---------------------------------------------
        # Fraction of position-axes where ŷ_τ is *not* non-decreasing in τ.
        # A trivially-collapsed head will have many crossings; a healthy
        # head will have ~0.
        diffs = quantiles[1:] - quantiles[:-1]  # (Q-1, B, V, S, P)
        crossings = (diffs < 0).any(dim=0).to(quantiles.dtype)  # (B, V, S, P)
        cross_frac = (crossings * w).sum() / denom

        for name, value in [
            ("val_crps_q", crps),
            ("val_cov_q10", cov_q10),
            ("val_cov_q50", cov_q50),
            ("val_cov_q90", cov_q90),
            ("val_cov_80", cov_80),
            ("val_cov_40", cov_40),
            ("val_mae_p50", mae_p50),
            ("val_quant_xings", cross_frac),
        ]:
            self.log(
                name,
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _knot_indices_for(self, *targets: float) -> tuple[int, ...]:
        """Return indices into ``self.loss_fn.quantile_levels`` for each target.

        Targets must coincide with one of the quantile knots used by the
        model (default ``[0.1, 0.2, …, 0.9]``).  Raises if any target is
        outside the available knots so config drift surfaces immediately.
        """
        levels = self.loss_fn.quantile_levels.detach().cpu().tolist()
        out: list[int] = []
        for t in targets:
            idx = min(range(len(levels)), key=lambda i: abs(levels[i] - t))
            if abs(levels[idx] - t) > 1e-6:
                raise ValueError(
                    f"Knot {t!r} is not available; have {levels}. "
                    "Probabilistic-forecast metrics need exact knot matches."
                )
            out.append(idx)
        return tuple(out)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    # ------------------------------------------------------------------
    # Optimizer / schedule (u-μP-aware)
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Cache u-μP fan values before optimizer construction; build JEPA target.

        ``dd_unit_scaling`` requires ``cache_fan_values`` to be called *before*
        FSDP wrapping (which replaces parameter tensors with DTensors and
        loses the original fan-in). Calling it here in ``setup`` runs after
        DDP/FSDP rank initialisation but before
        :meth:`configure_optimizers`.

        We also instantiate the JEPA target tower here — :class:`JEPAHead`
        deep-copies the online scaler / patch_proj / trunk so the target's
        weights are independent and only updated via :meth:`on_train_batch_end`'s
        EMA.  Doing this in ``setup`` (rather than ``__init__``) ensures the
        copy sees any post-init mutations (e.g., :class:`SigmaReparamLinear`
        γ buffers).
        """
        super().setup(stage)
        try:
            uu.cache_fan_values(self.model.named_parameters())
            # The auxiliary heads are u-μP-aware (uu.Linear / uu.RMSNorm)
            # so they need fan-value caching too — otherwise the optimiser
            # would fall back to fan-naive LR scaling for their params.
            if self.jepa_head is not None:
                uu.cache_fan_values(self.jepa_head.named_parameters())
            if self.pars_head is not None:
                uu.cache_fan_values(self.pars_head.named_parameters())
        except Exception:
            # If running without unit-scaling-tagged params, optimizer falls
            # back to standard scaling. We tolerate this for compatibility.
            pass

        if self.jepa_head is not None and getattr(self.jepa_head, "_target_trunk", None) is None:
            self.jepa_head.setup_target(
                scaler=self.model.scaler,
                patch_proj=self.model.patch_proj,
                trunk=self.model.transformer,
            )

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the JEPA target tower's EMA weights after each training step."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.jepa_head is not None:
            self.jepa_head.update_ema(
                scaler=self.model.scaler,
                patch_proj=self.model.patch_proj,
                trunk=self.model.transformer,
            )

    def configure_optimizers(self):
        # Trainable parameters live under (a) the trunk model, (b) the
        # auxiliary heads' *online* sides (projector, predictor, PARS, MR-STFT
        # — each only present if its config flag is enabled).  The JEPA
        # *target* tower lives inside ``jepa_head`` but its params are
        # ``requires_grad=False`` so the filter naturally excludes them.
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.jepa_head is not None:
            params.extend(p for p in self.jepa_head.parameters() if p.requires_grad)
        if self.pars_head is not None:
            params.extend(p for p in self.pars_head.parameters() if p.requires_grad)
        if self.mrstft_loss is not None:
            params.extend(p for p in self.mrstft_loss.parameters() if p.requires_grad)

        if self.optimizer_name == "adamw":
            optimizer = uu.AdamW(
                params,
                lr=self.base_lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                independent_weight_decay=True,
                allow_non_unit_scaling_params=True,
                # NOTE: ``fused=True`` would give a small (~5%) speedup but
                # Lightning's AMP precision plugin rejects gradient clipping
                # with fused optimizers (see lightning.pytorch.plugins.
                # precision.amp.MixedPrecisionPlugin.clip_gradients). We can
                # re-enable when we either drop gradient_clip_val or move to
                # bf16-true / a manual clipping hook.
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
    # Utilities
    # ------------------------------------------------------------------

    def num_input_patches(self) -> int:
        """Number of patch tokens emitted for an input window of ``context_length``."""
        return self.context_length // self.model.config.patch_size

    def patch_token_observed_fraction(self, target_mask: torch.Tensor) -> torch.Tensor:
        """Fraction of observed steps within each patch token (``B, V, S``)."""
        P = self.model.config.patch_size
        return reduce(target_mask.float(), "b v (s p) -> b v s", "mean", p=P)
