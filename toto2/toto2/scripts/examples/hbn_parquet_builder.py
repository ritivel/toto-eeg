"""HBN parquet dataset builder for Toto 2.0 training.

Reads the ``hbn_minimal_500hz`` (or ``hbn_v2_clean_250hz``) parquet files
produced by the eegfm preprocessing pipeline and converts them into Toto 2.0
training windows.

The HBN parquet layout is **one row = one (channel, window)** with columns:

    subject_id, channel_idx, channel_name, window_idx, window_start_s,
    sample_rate_hz, n_samples, signal (list<float16>), ...

This builder **reconstructs multi-channel windows** by grouping rows with
the same ``(subject_id, recording_id, window_idx)`` and stacking all
channels along the variate axis. The result is a ``(n_channels, T)`` tensor
per window — one Toto 2.0 training sample.

Since each window is 4 s @ 500 Hz = 2000 samples, and Toto's default
patch_size=64 requires ``context_length`` to be a multiple of 64, we set
``context_length = 1984`` (31 patches) and ``window_len = 1984 + 64 = 2048``.
The remaining 2000 - 1984 = 16 samples at the start are discarded (or the
window is left-aligned with the first 2048 samples used).

For longer context, multiple consecutive windows can be concatenated — the
``concat_windows`` parameter controls this (default 1).

Usage:

    python -m toto2.scripts.train_toto2 \\
        --config toto2/scripts/configs/pretrain_eeg_from_scratch.yaml \\
        --dataset-builder toto2.scripts.examples.hbn_parquet_builder:build_datasets
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from toto2.training import SlidingWindowConfig


class HBNParquetDataset(Dataset):
    """Dataset that reads HBN parquet files and produces multi-channel windows.

    Each sample is a dict with:
    - target: (n_channels, window_len)   float32
    - target_mask: (n_channels, window_len)  bool
    - series_ids: (n_channels,)  long — same id for all channels of one recording
    """

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        patch_size: int = 64,
        context_length: int = 1984,
        concat_windows: int = 1,
        channel_subset: Optional[list[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        random_offset: bool = True,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.context_length = context_length
        self.window_len = context_length + patch_size
        self.concat_windows = concat_windows
        self.channel_subset = set(channel_subset) if channel_subset else None
        self.transform = transform
        self.random_offset = random_offset
        self._rng = np.random.default_rng(seed)

        self._index: list[dict[str, Any]] = []
        self._data: dict[str, dict] = {}

        for path in parquet_paths:
            self._load_parquet(path)

    def _load_parquet(self, path: str) -> None:
        table = pq.read_table(path)
        subject_ids = table.column("subject_id").to_pylist()
        recording_ids = table.column("recording_id").to_pylist()
        channel_idxs = table.column("channel_idx").to_pylist()
        channel_names = table.column("channel_name").to_pylist()
        window_idxs = table.column("window_idx").to_pylist()
        signals = table.column("signal")
        sfreqs = table.column("sample_rate_hz").to_pylist()
        n_samples_col = table.column("n_samples").to_pylist()

        groups: dict[tuple[str, str, int], dict] = {}
        for i in range(len(subject_ids)):
            ch_name = channel_names[i]
            if self.channel_subset and ch_name not in self.channel_subset:
                continue

            key = (subject_ids[i], recording_ids[i], window_idxs[i])
            if key not in groups:
                groups[key] = {
                    "subject_id": subject_ids[i],
                    "recording_id": recording_ids[i],
                    "window_idx": window_idxs[i],
                    "sfreq": sfreqs[i],
                    "n_samples": n_samples_col[i],
                    "channels": {},
                }
            groups[key]["channels"][channel_idxs[i]] = np.array(
                signals[i].as_py(), dtype=np.float32
            )

        for key, group in groups.items():
            rec_key = f"{group['subject_id']}_{group['recording_id']}"
            if rec_key not in self._data:
                self._data[rec_key] = {
                    "windows": {},
                    "n_channels": 0,
                    "sfreq": group["sfreq"],
                    "n_samples": group["n_samples"],
                }
            self._data[rec_key]["windows"][group["window_idx"]] = group["channels"]
            self._data[rec_key]["n_channels"] = max(
                self._data[rec_key]["n_channels"], len(group["channels"])
            )

        for rec_key, rec_data in self._data.items():
            sorted_wins = sorted(rec_data["windows"].keys())
            n_concat = self.concat_windows
            for start in range(0, len(sorted_wins) - n_concat + 1, n_concat):
                self._index.append({
                    "rec_key": rec_key,
                    "win_ids": sorted_wins[start : start + n_concat],
                })

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self._index[idx]
        rec = self._data[entry["rec_key"]]
        n_ch = rec["n_channels"]
        n_samp = rec["n_samples"]
        win_ids = entry["win_ids"]

        total_samples = n_samp * len(win_ids)
        all_channels: list[np.ndarray] = []

        for ch_idx in range(n_ch):
            ch_data = []
            for wid in win_ids:
                win_channels = rec["windows"].get(wid, {})
                if ch_idx in win_channels:
                    ch_data.append(win_channels[ch_idx])
                else:
                    ch_data.append(np.zeros(n_samp, dtype=np.float32))
            all_channels.append(np.concatenate(ch_data))

        multi_ch = np.stack(all_channels)  # (n_ch, total_samples)

        if total_samples < self.window_len:
            pad = self.window_len - total_samples
            multi_ch = np.pad(multi_ch, ((0, 0), (pad, 0)), constant_values=0.0)
            mask = np.ones_like(multi_ch, dtype=bool)
            mask[:, :pad] = False
        else:
            if self.random_offset and total_samples > self.window_len:
                max_start = total_samples - self.window_len
                start = int(self._rng.integers(0, max_start + 1))
            else:
                start = 0
            multi_ch = multi_ch[:, start : start + self.window_len]
            mask = np.isfinite(multi_ch)

        target = torch.from_numpy(multi_ch).to(torch.float32)
        target_mask = torch.from_numpy(mask) & torch.isfinite(target)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform is not None:
            target = self.transform(target)
            target_mask = target_mask & torch.isfinite(target)
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        series_id_val = hash(entry["rec_key"]) % (2**31)
        series_ids = torch.full((target.shape[0],), series_id_val, dtype=torch.long)

        return {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }


def standardize_per_channel(window: torch.Tensor) -> torch.Tensor:
    mean = window.mean(dim=-1, keepdim=True)
    std = window.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (window - mean) / std


def build_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
    """Build (train_dataset, val_dataset) for Toto 2.0 HBN pre-training.

    Reads ``data.eeg`` block from the config. Expected layout on disk:

        <data_root>/derived/<pipeline>/sub-<NDAR...>/*.parquet

    Example config block:

    .. code-block:: yaml

        data:
          context_length: 1984
          eeg:
            data_root: /opt/dlami/nvme/eeg
            pipeline: hbn_minimal_500hz
            val_fraction: 0.1
            concat_windows: 1
            channel_subset: null   # or list of channel names
            transform: standardize_per_channel
    """
    data_cfg = config.get("data", {})
    eeg_cfg = data_cfg.get("eeg", {})

    data_root = eeg_cfg.get("data_root", "/opt/dlami/nvme/eeg")
    pipeline = eeg_cfg.get("pipeline", "hbn_minimal_500hz")
    derived_path = os.path.join(data_root, "derived", pipeline)

    if not Path(derived_path).is_dir():
        raise FileNotFoundError(
            f"Derived data not found at {derived_path}. "
            "Run rclone copy from S3 first."
        )

    all_parquets = sorted(glob.glob(os.path.join(derived_path, "**", "*.parquet"), recursive=True))
    if not all_parquets:
        raise FileNotFoundError(f"No parquet files found in {derived_path}")

    print(f"[hbn_parquet_builder] Found {len(all_parquets)} parquet files in {derived_path}")

    subject_dirs = sorted(set(os.path.dirname(p) for p in all_parquets))
    val_frac = float(eeg_cfg.get("val_fraction", 0.1))
    n_val = max(1, int(len(subject_dirs) * val_frac))
    seed = int(config.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(subject_dirs))
    val_dirs = set(subject_dirs[i] for i in indices[:n_val])
    train_dirs = set(subject_dirs[i] for i in indices[n_val:])

    train_parquets = [p for p in all_parquets if os.path.dirname(p) in train_dirs]
    val_parquets = [p for p in all_parquets if os.path.dirname(p) in val_dirs]

    print(f"[hbn_parquet_builder] Train: {len(train_parquets)} files ({len(train_dirs)} subjects)")
    print(f"[hbn_parquet_builder] Val: {len(val_parquets)} files ({len(val_dirs)} subjects)")

    patch_size = int(config.get("model", {}).get("patch_size", 64))
    context_length = int(data_cfg.get("context_length", 1984))
    concat_windows = int(eeg_cfg.get("concat_windows", 1))
    channel_subset = eeg_cfg.get("channel_subset")

    transform_name = eeg_cfg.get("transform", "standardize_per_channel")
    transform_fn = None
    if transform_name == "standardize_per_channel":
        transform_fn = standardize_per_channel
    elif transform_name and transform_name != "none":
        raise ValueError(f"Unknown transform: {transform_name}")

    train_ds = HBNParquetDataset(
        train_parquets,
        patch_size=patch_size,
        context_length=context_length,
        concat_windows=concat_windows,
        channel_subset=channel_subset,
        transform=transform_fn,
        random_offset=True,
        seed=seed,
    )

    val_ds = None
    if val_parquets:
        val_ds = HBNParquetDataset(
            val_parquets,
            patch_size=patch_size,
            context_length=context_length,
            concat_windows=concat_windows,
            channel_subset=channel_subset,
            transform=transform_fn,
            random_offset=False,
            seed=seed,
        )

    return train_ds, val_ds
