"""
Adapter wrapping Toto2's pretrained backbone for open-eeg-bench.

open-eeg-bench expects a PyTorch model that:
  - Accepts input of shape  (batch, n_chans, n_times)
  - Returns output of shape (batch, n_features)
  - Has a named module `self.final_layer` for the classification head

This adapter loads a Toto2 checkpoint, strips the quantile output head,
runs raw EEG through the scaler -> patch_proj -> transformer backbone,
and pools the resulting embeddings into a fixed-length feature vector.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce

from toto2.configuration import Toto2ModelConfig
from toto2.model import Toto2Model

DEFAULT_CONTEXT_LENGTH = 4096


def _coords_from_chs_info(chs_info: Optional[list]) -> Optional[np.ndarray]:
    """Best-effort extraction of unit-sphere positions from MNE ``info['chs']``.

    open-eeg-bench passes a list of MNE channel-info dicts at build time
    when the underlying dataset has electrode positions.  Each dict has
    ``loc`` which is a 12-element vector whose first three components
    are the (x, y, z) position in metres.  We normalise each row to
    unit norm so the output is directly usable by ``CoordPE``.

    Returns ``None`` if ``chs_info`` is missing or any channel lacks a
    valid position.
    """
    if not chs_info:
        return None
    positions = []
    for ch in chs_info:
        loc = None
        if isinstance(ch, dict):
            loc = ch.get("loc")
        else:
            loc = getattr(ch, "loc", None)
        if loc is None:
            return None
        loc_arr = np.asarray(loc, dtype=np.float64).reshape(-1)
        if loc_arr.size < 3:
            return None
        xyz = loc_arr[:3]
        if not np.all(np.isfinite(xyz)) or np.linalg.norm(xyz) < 1e-12:
            return None
        positions.append(xyz)
    arr = np.asarray(positions, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


class Toto2EEGBenchModel(nn.Module):
    """Toto2 backbone adapted for EEG classification benchmarks.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local path to a PyTorch Lightning ``.ckpt`` file or a
        ``safetensors``/``state_dict`` file.
    d_model : int
        Embedding dimension of the Toto2 backbone (must match checkpoint).
    patch_size : int
        Patch size used during pretraining (must match checkpoint).
    n_chans : int
        Number of EEG channels the benchmark will provide. Each channel
        is treated as an independent variate (series).
    pool : str
        Pooling strategy over (channels, time_patches): ``"mean"`` or ``"cls"``.
    config_overrides : dict, optional
        Extra overrides passed to ``Toto2ModelConfig``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        d_model: int = 384,
        patch_size: int = 64,
        n_chans: int = 64,
        num_layers: int = 12,
        num_heads: int = 6,
        pool: str = "mean",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        config_overrides: Optional[dict] = None,
        electrode_coords: Optional[np.ndarray] = None,
        # open-eeg-bench passes these at build time
        n_times: Optional[int] = None,
        n_outputs: Optional[int] = None,
        sfreq: Optional[float] = None,
        chs_info: Optional[list] = None,
        **extra_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_chans = n_chans
        self.pool = pool

        attn_ratio = Toto2ModelConfig.compute_residual_attn_ratio(context_length, patch_size)

        cfg_kwargs = dict(
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_group_size=num_layers,
            num_variate_layers_per_group=1,
            variate_layer_first=False,
            num_output_patches=1,
            qk_norm=True,
            norm_include_weight=False,
            per_dim_scale=False,
            use_xpos=False,
            residual_mult=0.75,
            residual_attn_ratio=attn_ratio,
        )
        if config_overrides:
            cfg_kwargs.update(config_overrides)

        config = Toto2ModelConfig(**cfg_kwargs)
        self._toto = Toto2Model(config)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.final_layer = nn.Identity()

        # exp49 — resolve electrode coords for downstream eval.  Three
        # accepted sources, in order of preference:
        #   1. Explicit ``electrode_coords`` arg (np.ndarray of shape
        #      (n_chans, 3), unit-sphere normalised).
        #   2. ``chs_info`` from open-eeg-bench (MNE channel-info dicts
        #      with 'loc'); we auto-derive unit-sphere positions.
        #   3. Neither — only OK if the underlying model does not enable
        #      coord_pe, in which case coords are ignored.
        coords_buffer: Optional[torch.Tensor] = None
        if electrode_coords is not None:
            arr = np.asarray(electrode_coords, dtype=np.float32)
            if arr.ndim != 2 or arr.shape != (n_chans, 3):
                raise ValueError(
                    f"electrode_coords must have shape ({n_chans}, 3); "
                    f"got {arr.shape}."
                )
            coords_buffer = torch.from_numpy(arr.copy())
        else:
            inferred = _coords_from_chs_info(chs_info)
            if inferred is not None and inferred.shape == (n_chans, 3):
                coords_buffer = torch.from_numpy(inferred.copy())
        if coords_buffer is not None:
            self.register_buffer("electrode_coords", coords_buffer)
        else:
            self.electrode_coords = None  # type: ignore[assignment]
            if self._toto.coord_pe is not None:
                raise ValueError(
                    "Toto2EEGBenchModel: model uses use_coord_pe=True but no "
                    "electrode coordinates were provided.  Either pass "
                    "``electrode_coords`` directly or rely on ``chs_info`` "
                    "carrying MNE-style channel positions ('loc')."
                )

    def _load_checkpoint(self, path: Union[str, Path]):
        """Load weights from a Lightning .ckpt or plain state_dict file."""
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" in state:
            raw = state["state_dict"]
        elif "model_state_dict" in state:
            raw = state["model_state_dict"]
        else:
            raw = state

        model_sd = {}
        for k, v in raw.items():
            clean = k.removeprefix("model.").removeprefix("_toto.")
            model_sd[clean] = v

        self._toto.load_state_dict(model_sd, strict=False)

    def _pad_to_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Pad time dimension to a multiple of patch_size."""
        n_times = x.shape[-1]
        remainder = n_times % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = torch.nn.functional.pad(x, (0, pad_len), value=0.0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, n_chans, n_times)
            Raw EEG input at 100 Hz (from open-eeg-bench datasets).

        Returns
        -------
        features : Tensor of shape (batch, d_model)
            Pooled backbone embeddings.
        """
        batch, n_chans, n_times = x.shape

        x_padded = self._pad_to_patch(x)
        n_times_padded = x_padded.shape[-1]

        target = x_padded
        target_mask = torch.ones_like(target, dtype=torch.bool)
        if n_times_padded != n_times:
            target_mask[..., n_times:] = False

        cpm_mask = target_mask
        series_ids = torch.arange(n_chans, device=x.device).unsqueeze(0).expand(batch, -1)

        scaled, loc, scale = self._toto.scaler(target, target_mask & cpm_mask)
        scaled = scaled.asinh()

        # exp49 — feed coords through ``_embed_patches`` when CoordPE is on.
        # ``self.electrode_coords`` is a buffer of shape (n_chans, 3) that
        # we broadcast over the batch axis; if the model was trained
        # without coord_pe then it is None and we pass None straight through.
        embed_kwargs = {}
        if self.electrode_coords is not None:
            embed_kwargs["electrode_coords"] = (
                self.electrode_coords.to(scaled.dtype)
                .unsqueeze(0)
                .expand(batch, -1, -1)
            )
        patches = self._toto._embed_patches(
            scaled,
            target_mask & cpm_mask,
            self.patch_size,
            **embed_kwargs,
        )

        group_ids = series_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-2]).clone()
        mask_per_patch = reduce(
            target_mask, "... (seq patch) -> ... seq", "sum", patch=self.patch_size
        )
        cpm_per_patch = reduce(
            cpm_mask, "... (seq patch) -> ... seq", "prod", patch=self.patch_size
        )
        group_ids[(mask_per_patch == 0) & (cpm_per_patch == 1)] = -1

        embeddings = self._toto.transformer(patches, group_ids=group_ids)

        if self.pool == "mean":
            patch_mask = (mask_per_patch > 0).unsqueeze(-1).float()
            masked_emb = embeddings * patch_mask
            features = masked_emb.sum(dim=(-3, -2)) / patch_mask.sum(dim=(-3, -2)).clamp(min=1)
        else:
            features = embeddings[..., -1, :].mean(dim=-2)

        return self.final_layer(features)
