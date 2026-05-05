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
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import SlidingWindowConfig


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

    The constructor only stores the path list — it never opens any file.
    Length is reported as ``len(paths) * windows_per_file``, and each
    ``__getitem__`` picks one (file, random offset) pair, decompresses the
    file once (cached), and returns a window slice. This means the dataset
    initialises in O(N) over filesystem listings — *not* over file
    contents — so 100k+ files initialise in well under a second.

    Parameters
    ----------
    paths
        Paths to ``.npz`` files. Each file must contain a 1-D or 2-D array
        keyed by ``array_key`` representing one recording.
    config
        :class:`SlidingWindowConfig` controlling window size, stride, and
        whether sampling is random. ``stride`` is ignored in random mode;
        in deterministic mode we yield one window per file at a fixed
        offset (``stride * window_idx % T``-style sampling is not implemented
        here — use :class:`ArrayTimeSeriesDataset` for full coverage).
    array_key
        Key under which the EEG array lives in the ``.npz``. Default
        ``"data"`` matches ``convert_hbn_to_npz.py``'s output.
    expected_channels
        Optional sanity check; windows from recordings with the wrong
        channel count are dropped at access time and a different file is
        re-sampled (best-effort retry up to 8 times).
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
    windows_per_file
        Number of "logical" windows reported per file in ``len(self)``.
        Together with ``train_batch_size`` this controls one epoch's
        nominal length. Each window is a fresh random offset into a
        randomly-chosen file when ``config.random`` is True.
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
        windows_per_file: int = 4,
    ) -> None:
        if not paths:
            raise ValueError("LazyNpzTimeSeriesDataset received an empty path list.")
        self.config = config
        self.array_key = array_key
        self.expected_channels = expected_channels
        self.transform = transform
        self.nan_to_num = nan_to_num
        self.series_id_offset = int(series_id_offset)
        self.windows_per_file = max(1, int(windows_per_file))
        self._cache = _LRUCache(cache_size)

        self._paths: list[str] = sorted(str(p) for p in paths)
        # Worker-process-local cache of (n_channels, n_samples) discovered
        # at first read. Skips files that turn out to be unusable.
        self._known_shape: dict[int, tuple[int, int]] = {}

        print(
            f"[lazy_npz] indexed {len(self._paths)} files "
            f"(reporting {len(self._paths) * self.windows_per_file} logical windows)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._paths) * self.windows_per_file

    def _load_recording(self, file_idx: int) -> Optional[np.ndarray]:
        cached = self._cache.get(file_idx)
        if cached is not None:
            return cached
        path = self._paths[file_idx]
        try:
            with np.load(path, allow_pickle=False) as f:
                if self.array_key not in f.files:
                    return None
                arr = np.asarray(f[self.array_key], dtype=np.float32)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            print(f"[lazy_npz] skip {path}: {e}", flush=True)
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            return None
        if self.expected_channels is not None and arr.shape[0] != self.expected_channels:
            return None
        if arr.shape[1] < self.config.window_len:
            return None
        self._known_shape[file_idx] = (int(arr.shape[0]), int(arr.shape[1]))
        self._cache.put(file_idx, arr)
        return arr

    def _sample_window(
        self, file_idx: int, rng: np.random.Generator,
    ) -> Optional[dict[str, torch.Tensor]]:
        recording = self._load_recording(file_idx)
        if recording is None:
            return None

        n_ch, n_samp = recording.shape
        max_start = n_samp - self.config.window_len
        if self.config.random:
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        else:
            start = 0
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
            (n_ch,), self.series_id_offset + file_idx, dtype=torch.long,
        )
        return {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n_files = len(self._paths)

        # Per-call RNG seeded by (config.seed, worker_id, idx) so multiple
        # DataLoader workers don't draw the same sequence after fork.
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        seed_payload = (self.config.seed, wid, idx)
        rng = np.random.default_rng(hash(seed_payload) & 0xFFFFFFFF)

        if self.config.random:
            file_idx = int(rng.integers(0, n_files))
        else:
            file_idx = idx % n_files

        # Try the chosen file; on failure, walk forward up to 8 times.
        for attempt in range(8):
            sample = self._sample_window(file_idx, rng)
            if sample is not None:
                return sample
            file_idx = (file_idx + 1) % n_files
        raise RuntimeError(
            f"LazyNpzTimeSeriesDataset: 8 consecutive files failed at idx={idx}; "
            "check the source data for corruption / wrong channel counts."
        )
