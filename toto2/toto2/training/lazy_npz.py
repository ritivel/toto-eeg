# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Lazy, file-backed sliding-window dataset for ``.npz`` recordings.

:class:`ArrayTimeSeriesDataset` keeps every recording resident in RAM, which
becomes impractical past a few tens of GiB of audio/EEG data. This module
provides a drop-in alternative that opens each ``.npz`` only when sampled,
reads exactly the window slice required, and never holds more than one
recording in memory per worker.

A small in-process LRU cache softens the cost of repeatedly returning
windows from the same recording within a single worker (common when stride
is much smaller than recording length); set ``cache_size=0`` to disable.

Each sample is the same dictionary that :class:`ArrayTimeSeriesDataset`
yields, so it composes with :func:`toto2.training.collate_timeseries` and
:class:`TimeSeriesDataModule` unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import SlidingWindowConfig


@dataclass
class _RecordingMeta:
    """Per-file metadata used to plan windows without opening the file."""

    path: str
    n_channels: int
    n_samples: int
    series_id: int


def _probe_npz(path: str, array_key: str) -> tuple[int, int]:
    """Return ``(n_channels, n_samples)`` for an .npz array without copying."""
    with np.load(path, allow_pickle=False) as f:
        if array_key not in f.files:
            raise KeyError(f"{path}: missing {array_key!r}")
        # ``f[array_key]`` returns a NpzFile-resident array. We only want the
        # shape, which is cheap to read out of the metadata header.
        arr = f[array_key]
        if arr.ndim == 1:
            return 1, int(arr.shape[0])
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 1-D or 2-D; got {arr.shape}.")
        return int(arr.shape[0]), int(arr.shape[1])


class _LRUCache:
    """Tiny LRU keyed by integer recording index, returning numpy arrays."""

    def __init__(self, max_entries: int):
        self.max_entries = int(max(0, max_entries))
        self._store: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if self.max_entries == 0:
            return None
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: int, value: np.ndarray) -> None:
        if self.max_entries == 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


class LazyNpzTimeSeriesDataset(Dataset):
    """Sliding-window dataset that reads ``.npz`` files on demand.

    Parameters
    ----------
    paths
        Paths to ``.npz`` files. Each file must contain a 1-D or 2-D array
        keyed by ``array_key`` representing one recording.
    config
        :class:`SlidingWindowConfig` controlling window size, stride, and
        whether sampling is random.
    array_key
        Key under which the EEG array lives in the ``.npz``. Default
        ``"data"`` matches ``convert_hbn_to_npz.py``'s output.
    expected_channels
        Optional sanity check; recordings whose first dim differs are
        excluded from the index.
    transform
        Optional callable applied per-window after slicing. Useful for
        per-channel z-scoring etc.
    nan_to_num
        Replace ``NaN``/``Inf`` with zero in the value tensor. The mask
        records those positions as unobserved regardless of this flag.
    cache_size
        Number of recordings to keep loaded per worker. ``0`` disables
        caching entirely; ``2`` is a good default for sliding windows.
    series_id_offset
        Added to the per-file integer id assigned to all variates of one
        recording — useful when concatenating multiple datasets.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        config: SlidingWindowConfig,
        *,
        array_key: str = "data",
        expected_channels: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        nan_to_num: bool = True,
        cache_size: int = 2,
        series_id_offset: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("LazyNpzTimeSeriesDataset received an empty path list.")
        self.config = config
        self.array_key = array_key
        self.expected_channels = expected_channels
        self.transform = transform
        self.nan_to_num = nan_to_num
        self._cache = _LRUCache(cache_size)
        self._rng = np.random.default_rng(config.seed)

        self._meta: list[_RecordingMeta] = []
        skipped = 0
        for i, p in enumerate(sorted(str(x) for x in paths)):
            try:
                n_ch, n_samp = _probe_npz(p, array_key)
            except (OSError, ValueError, KeyError) as e:
                print(f"[lazy_npz] skip {p}: {e}", flush=True)
                skipped += 1
                continue
            if expected_channels is not None and n_ch != expected_channels:
                skipped += 1
                continue
            if n_samp < config.window_len:
                skipped += 1
                continue
            self._meta.append(
                _RecordingMeta(
                    path=p,
                    n_channels=n_ch,
                    n_samples=n_samp,
                    series_id=series_id_offset + i,
                )
            )
        if not self._meta:
            raise ValueError(
                "No recording was long enough or matched expected_channels. "
                f"Skipped {skipped} of {len(paths)}."
            )

        # Plan: deterministic (recording_idx, start) pairs covering the
        # corpus once at the configured stride. Random sampling reuses the
        # plan length but draws a fresh random offset per call.
        plan: list[tuple[int, int]] = []
        for rec_idx, meta in enumerate(self._meta):
            max_start = meta.n_samples - config.window_len
            for start in range(0, max_start + 1, config.stride):
                plan.append((rec_idx, start))
        if not plan:
            raise ValueError("Sliding plan empty; check stride / window_len.")
        self._plan = plan
        if skipped:
            print(
                f"[lazy_npz] indexed {len(self._meta)} recordings "
                f"({len(plan)} windows); skipped {skipped} files.",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._plan)

    def _read_recording(self, rec_idx: int) -> np.ndarray:
        cached = self._cache.get(rec_idx)
        if cached is not None:
            return cached
        meta = self._meta[rec_idx]
        with np.load(meta.path, allow_pickle=False) as f:
            arr = np.asarray(f[self.array_key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self._cache.put(rec_idx, arr)
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, det_start = self._plan[idx]
        meta = self._meta[rec_idx]

        if self.config.random:
            # Derive randomness per-call from (worker_id, idx) so multiple
            # DataLoader workers don't all draw the same sequence after fork.
            worker_info = torch.utils.data.get_worker_info()
            wid = worker_info.id if worker_info is not None else 0
            seed_payload = (self.config.seed, wid, idx)
            local_rng = np.random.default_rng(hash(seed_payload) & 0xFFFFFFFF)
            max_start = meta.n_samples - self.config.window_len
            start = int(local_rng.integers(0, max_start + 1)) if max_start > 0 else 0
        else:
            start = det_start

        recording = self._read_recording(rec_idx)
        window = recording[:, start : start + self.config.window_len]

        if self.transform is not None:
            window_t = torch.from_numpy(window.copy()).to(torch.float32)
            window_t = self.transform(window_t)
            window_np = window_t.numpy()
        else:
            window_np = window.copy()

        finite = np.isfinite(window_np)
        if self.nan_to_num and not finite.all():
            window_np = np.where(finite, window_np, 0.0)

        target = torch.from_numpy(window_np).to(torch.float32)
        target_mask = torch.from_numpy(finite)
        series_ids = torch.full(
            (target.shape[0],), meta.series_id, dtype=torch.long,
        )
        return {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }
