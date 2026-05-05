# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Reference dataset builder for EEG pre-training of Toto 2.0.

This builder is intentionally simple — it expects the user to have already
exported each EEG recording to an ``.npz`` file containing a 2-D float array
``(channels, time_steps)`` under a configurable key (default ``"data"``).
Files matching ``data.eeg.train_glob`` go into the training set; files
matching ``data.eeg.val_glob`` (optional) go into the validation set.

To plug your own pipeline (e.g. MNE-Python, BIDS, HDF5, S3-streamed parquet)
in, write a callable with the same signature and pass its dotted path via
``--dataset-builder my_pkg.my_module:my_callable``.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from toto2.training import (
    ArrayTimeSeriesDataset,
    SlidingWindowConfig,
)


# ----------------------------------------------------------------------
# Per-channel transforms
# ----------------------------------------------------------------------


def standardize_per_channel(window: torch.Tensor) -> torch.Tensor:
    """Z-score each channel of a ``(C, T)`` window using its own statistics."""
    mean = window.mean(dim=-1, keepdim=True)
    std = window.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (window - mean) / std


def standardize_robust_per_channel(window: torch.Tensor) -> torch.Tensor:
    """Robust z-score using median / MAD per channel (better for spiky EEG)."""
    median = window.median(dim=-1, keepdim=True).values
    mad = (window - median).abs().median(dim=-1, keepdim=True).values.clamp_min(1e-6)
    return (window - median) / (1.4826 * mad)


_TRANSFORM_REGISTRY: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": lambda x: x,
    "standardize_per_channel": standardize_per_channel,
    "standardize_robust_per_channel": standardize_robust_per_channel,
}


def _resolve_transform(spec: Optional[Dict[str, Any]]) -> Optional[Callable]:
    if not spec:
        return None
    name = spec.get("type", "none")
    if name not in _TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform {name!r}; available: {list(_TRANSFORM_REGISTRY)}."
        )
    return _TRANSFORM_REGISTRY[name]


# ----------------------------------------------------------------------
# .npz loader
# ----------------------------------------------------------------------


def _load_npz_recordings(
    paths: Sequence[str],
    *,
    array_key: str,
    expected_channels: Optional[int],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as f:
            if array_key not in f.files:
                continue
            arr = np.asarray(f[array_key], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"{path}: expected 2-D (channels, time) array under {array_key!r}, "
                    f"got shape {arr.shape}."
                )
            if expected_channels is not None and arr.shape[0] != expected_channels:
                # Skip recordings with the wrong montage rather than failing hard,
                # and emit a single line for visibility.
                print(
                    f"[eeg_builder] Skipping {path}: {arr.shape[0]} channels, "
                    f"expected {expected_channels}."
                )
                continue
            out.append(arr)
    return out


def _list_files(root: str, pattern: Optional[str]) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------


def build_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
    """Build (train_dataset, val_dataset) for Toto 2.0 EEG pre-training.

    Reads the ``data:`` block of the parsed YAML config — specifically the
    ``data.eeg`` sub-block:

    .. code-block:: yaml

        data:
          context_length: 4096
          eeg:
            data_root: /path/to/eeg
            train_glob: "*train*.npz"
            val_glob: "*val*.npz"
            array_key: data
            sfreq: 256
            expected_channels: 19
            sliding_stride: 2048
            min_observed_frac: 0.8
            transform: { type: standardize_per_channel }
    """
    data_cfg = config.get("data", {})
    eeg_cfg = data_cfg.get("eeg", {})
    if not eeg_cfg:
        raise ValueError("Config is missing data.eeg block.")

    data_root = str(eeg_cfg.get("data_root"))
    if not data_root or not Path(data_root).is_dir():
        raise FileNotFoundError(f"data.eeg.data_root does not exist: {data_root!r}")

    train_paths = _list_files(data_root, eeg_cfg.get("train_glob"))
    val_paths = _list_files(data_root, eeg_cfg.get("val_glob"))

    if not train_paths:
        raise FileNotFoundError(
            f"No training files matched in {data_root} (glob={eeg_cfg.get('train_glob')!r})."
        )

    # If no explicit val glob, hold out a deterministic fraction of subject
    # directories for validation so train/val are subject-disjoint (matches
    # the Evaluation Suite's release-based split convention).
    if not val_paths:
        val_fraction = float(eeg_cfg.get("val_fraction", 0.0))
        if val_fraction > 0.0:
            seed = int(config.get("training", {}).get("seed", 42))
            subject_dirs = sorted({os.path.dirname(p) for p in train_paths})
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(subject_dirs))
            n_val = max(1, int(len(subject_dirs) * val_fraction))
            val_dirs = {subject_dirs[i] for i in order[:n_val]}
            val_paths = [p for p in train_paths if os.path.dirname(p) in val_dirs]
            train_paths = [p for p in train_paths if os.path.dirname(p) not in val_dirs]
            print(
                f"[eeg_builder] Holding out {len(val_dirs)} of {len(subject_dirs)} "
                f"subjects ({val_fraction:.0%}) for validation; "
                f"train={len(train_paths)} val={len(val_paths)} files.",
                flush=True,
            )

    array_key = str(eeg_cfg.get("array_key", "data"))
    expected_channels = eeg_cfg.get("expected_channels")
    if expected_channels is not None:
        expected_channels = int(expected_channels)

    transform = _resolve_transform(eeg_cfg.get("transform"))

    # Pull patch_size from the model section if present, else fall back to a default.
    patch_size = int(config.get("model", {}).get("patch_size", 64))
    context_length = int(data_cfg.get("context_length", 4096))

    sliding_cfg_train = SlidingWindowConfig(
        context_length=context_length,
        patch_size=patch_size,
        stride=int(eeg_cfg.get("sliding_stride", context_length)),
        random=True,
        seed=int(config.get("training", {}).get("seed", 42)),
        min_observed_frac=float(eeg_cfg.get("min_observed_frac", 0.5)),
    )
    sliding_cfg_val = SlidingWindowConfig(
        context_length=context_length,
        patch_size=patch_size,
        stride=int(eeg_cfg.get("sliding_stride", context_length)),
        random=False,
        seed=int(config.get("training", {}).get("seed", 42)),
        min_observed_frac=float(eeg_cfg.get("min_observed_frac", 0.5)),
    )

    train_recordings = _load_npz_recordings(
        train_paths,
        array_key=array_key,
        expected_channels=expected_channels,
    )
    train_ds = ArrayTimeSeriesDataset(
        train_recordings,
        sliding_cfg_train,
        transform=transform,
    )

    val_ds: Optional[Dataset] = None
    if val_paths:
        val_recordings = _load_npz_recordings(
            val_paths,
            array_key=array_key,
            expected_channels=expected_channels,
        )
        if val_recordings:
            val_ds = ArrayTimeSeriesDataset(
                val_recordings,
                sliding_cfg_val,
                transform=transform,
            )

    return train_ds, val_ds
