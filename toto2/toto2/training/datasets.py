# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""PyTorch ``Dataset`` implementations for Toto 2.0 pre-training.

Toto 2.0 expects each training sample to be a contiguous window with shape
``(n_var, window_len)``, plus auxiliary masks. We provide two flexible
producers that can be used directly or sub-classed:

- :class:`ArrayTimeSeriesDataset` — wraps an in-memory list of NumPy /
  ``torch.Tensor`` recordings (one ``(n_var, T)`` array per recording) and
  samples sliding windows. Suitable for EEG datasets that fit in RAM.
- :class:`HFTimeSeriesDataset` — adapter for HuggingFace ``datasets.Dataset``
  rows. Each row supplies one or more ``(T,)`` target series; the adapter
  stacks them along the variate axis and slides windows.

Both producers emit dictionaries with the keys consumed by
:func:`toto2.training.collate.collate_timeseries`:

- ``target`` : ``Float[V, window_len + patch_size]``
- ``target_mask`` : ``Bool[V, window_len + patch_size]`` (1 = observed)
- ``series_ids`` : ``Int[V]`` (group id per variate)

The ``+ patch_size`` is the one-patch shift required for next-patch
supervision, so the LightningModule can split into "input" and "shifted
target" without re-loading data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding-window sampling.

    Attributes
    ----------
    context_length
        Number of timesteps in the *input* portion of each window. Must be a
        multiple of ``patch_size`` to align cleanly with patch tokenization.
    patch_size
        Patch size of the Toto 2.0 model being trained.
    stride
        Step between successive deterministic windows in
        :meth:`ArrayTimeSeriesDataset.iter_deterministic_windows`. Defaults to
        ``context_length`` (non-overlapping windows).
    random
        If ``True``, ``__getitem__`` picks a random valid start offset for
        each call (data augmentation). If ``False``, windows are produced
        deterministically (one per ``(recording, start)`` pair).
    seed
        RNG seed for reproducible random sampling.
    min_observed_frac
        Drop windows where fewer than this fraction of timesteps carry valid
        observations across all variates (helps avoid mostly-NaN windows).
    """

    context_length: int
    patch_size: int
    stride: Optional[int] = None
    random: bool = True
    seed: int = 0
    min_observed_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.context_length <= 0 or self.context_length % self.patch_size != 0:
            raise ValueError(
                "context_length must be a positive multiple of patch_size; "
                f"got context_length={self.context_length}, patch_size={self.patch_size}."
            )
        if self.stride is None:
            self.stride = self.context_length
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if not 0.0 <= self.min_observed_frac <= 1.0:
            raise ValueError("min_observed_frac must lie in [0, 1].")

    @property
    def window_len(self) -> int:
        """Total length of each window — input plus the one-patch target shift."""
        return self.context_length + self.patch_size


def _to_tensor_2d(array: Any) -> torch.Tensor:
    """Coerce a recording to a ``(n_var, T)`` ``torch.float32`` tensor."""
    if isinstance(array, torch.Tensor):
        t = array.to(dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array; got shape {tuple(t.shape)}.")
    return t.contiguous()


class ArrayTimeSeriesDataset(Dataset):
    """Sliding-window dataset over a list of in-memory recordings.

    Parameters
    ----------
    recordings
        Iterable of array-likes. Each entry is one recording with shape
        ``(n_var, T)`` (or ``(T,)`` for univariate). Different recordings
        may have different ``n_var`` *and* ``T``; they will be padded /
        masked appropriately by :func:`collate_timeseries`.
    config
        Sampling configuration (window length, stride, etc.).
    series_ids
        Optional per-recording group ids for variates (length ``n_var``).
        If ``None``, all variates of recording ``i`` get id ``i`` so they
        attend to each other but not to other recordings (block-diagonal
        attention pattern, matching Toto's variate-id mask).
    transform
        Optional callable applied to each sampled window
        ``(n_var, window_len)`` *before* mask generation. Useful for
        per-recording z-scoring, band-pass filtering, etc.
    nan_to_num
        Replace ``NaN`` / ``Inf`` with zero in the value tensor. The mask
        will record those positions as unobserved regardless of this flag.
    electrode_coords
        Optional ``(n_var, 3)`` array (or per-recording sequence of
        such arrays) of unit-sphere-normalised electrode positions.
        When set, every emitted sample carries an ``electrode_coords``
        tensor that the exp49 ``CoordPE`` module consumes.  Pass
        ``None`` (default) to fall back to v3 / exp48 behaviour where
        only ``series_ids`` provide variate identity.
    """

    def __init__(
        self,
        recordings: Sequence[Any],
        config: SlidingWindowConfig,
        *,
        series_ids: Optional[Sequence[Sequence[int]]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        nan_to_num: bool = True,
        electrode_coords: Optional[Any] = None,
    ) -> None:
        if len(recordings) == 0:
            raise ValueError("Dataset received no recordings.")
        self.recordings: list[torch.Tensor] = [_to_tensor_2d(r) for r in recordings]
        self.config = config
        self.transform = transform
        self.nan_to_num = nan_to_num

        if series_ids is None:
            self.series_ids = [
                torch.full((rec.shape[0],), fill_value=i, dtype=torch.long)
                for i, rec in enumerate(self.recordings)
            ]
        else:
            if len(series_ids) != len(self.recordings):
                raise ValueError("series_ids must have one entry per recording.")
            self.series_ids = [torch.as_tensor(sid, dtype=torch.long) for sid in series_ids]
            for i, (sid, rec) in enumerate(zip(self.series_ids, self.recordings)):
                if sid.numel() != rec.shape[0]:
                    raise ValueError(
                        f"series_ids[{i}] has length {sid.numel()} but recording has "
                        f"{rec.shape[0]} variates."
                    )

        # exp49 — optional electrode_coords (1 per variate, 3-D each).
        self.electrode_coords: Optional[list[torch.Tensor]]
        if electrode_coords is None:
            self.electrode_coords = None
        else:
            arr = np.asarray(electrode_coords, dtype=np.float32)
            if arr.ndim == 2:
                # Single shared montage broadcast over every recording.
                if arr.shape[1] != 3:
                    raise ValueError(
                        f"electrode_coords must have shape (n_var, 3); got {arr.shape}."
                    )
                self.electrode_coords = []
                for rec in self.recordings:
                    if arr.shape[0] != rec.shape[0]:
                        raise ValueError(
                            f"Recording with {rec.shape[0]} variates does not match "
                            f"electrode_coords with {arr.shape[0]} positions."
                        )
                    self.electrode_coords.append(torch.from_numpy(arr.copy()))
            elif arr.ndim == 3:
                # Per-recording montage list.
                if arr.shape[2] != 3:
                    raise ValueError(
                        f"electrode_coords must have last dim 3; got {arr.shape}."
                    )
                if arr.shape[0] != len(self.recordings):
                    raise ValueError(
                        "Per-recording electrode_coords length does not match "
                        f"#recordings ({arr.shape[0]} vs {len(self.recordings)})."
                    )
                self.electrode_coords = [torch.from_numpy(row.copy()) for row in arr]
            else:
                raise ValueError(
                    "electrode_coords must be (n_var, 3) or (n_recordings, n_var, 3); "
                    f"got shape {arr.shape}."
                )

        # Per-recording valid start offsets (T - window_len + 1)
        self._max_starts = [
            max(0, rec.shape[1] - config.window_len + 1) for rec in self.recordings
        ]

        # Pre-compute deterministic index plan
        plan: list[tuple[int, int]] = []
        for rec_idx, max_start in enumerate(self._max_starts):
            if max_start == 0:
                continue
            for start in range(0, max_start, config.stride):
                plan.append((rec_idx, start))
            # Always include the final aligned window so the last samples are seen
            last = max_start - 1
            if last > 0 and (last % config.stride) != 0:
                plan.append((rec_idx, last))
        if not plan:
            raise ValueError(
                "No recording is long enough to fit a single window of length "
                f"{config.window_len}. Check context_length / patch_size or pad your data."
            )
        self._plan = plan
        self._rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return len(self._plan)

    def _sample_window(self, rec_idx: int, start: int) -> dict[str, torch.Tensor]:
        rec = self.recordings[rec_idx]
        end = start + self.config.window_len
        window = rec[:, start:end]

        if self.transform is not None:
            window = self.transform(window)
            if window.shape != (rec.shape[0], self.config.window_len):
                raise ValueError(
                    "transform must preserve shape; expected "
                    f"({rec.shape[0]}, {self.config.window_len}), got {tuple(window.shape)}."
                )

        # Build observed-value mask (1 = real measurement, 0 = NaN / Inf / pad)
        finite = torch.isfinite(window)
        if self.nan_to_num:
            window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        sample: dict[str, torch.Tensor] = {
            "target": window,
            "target_mask": finite,
            "series_ids": self.series_ids[rec_idx],
        }
        if self.electrode_coords is not None:
            sample["electrode_coords"] = self.electrode_coords[rec_idx]
        return sample

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec_idx, det_start = self._plan[index]
        if self.config.random and self._max_starts[rec_idx] > 0:
            start = int(self._rng.integers(0, self._max_starts[rec_idx]))
        else:
            start = det_start

        sample = self._sample_window(rec_idx, start)
        # Reject windows with too few valid points; resample once to keep batch shapes stable.
        observed_frac = sample["target_mask"].float().mean().item()
        if observed_frac < self.config.min_observed_frac and self.config.random:
            for _ in range(8):
                start = int(self._rng.integers(0, self._max_starts[rec_idx]))
                sample = self._sample_window(rec_idx, start)
                if sample["target_mask"].float().mean().item() >= self.config.min_observed_frac:
                    break
        return sample

    def iter_deterministic_windows(self) -> "ArrayTimeSeriesDataset":
        """Return a sibling dataset with random sampling disabled (for validation)."""
        sibling_config = SlidingWindowConfig(
            context_length=self.config.context_length,
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            random=False,
            seed=self.config.seed,
            min_observed_frac=self.config.min_observed_frac,
        )
        sibling = ArrayTimeSeriesDataset.__new__(ArrayTimeSeriesDataset)
        sibling.recordings = self.recordings
        sibling.config = sibling_config
        sibling.transform = self.transform
        sibling.nan_to_num = self.nan_to_num
        sibling.series_ids = self.series_ids
        sibling.electrode_coords = self.electrode_coords
        sibling._max_starts = self._max_starts
        sibling._plan = self._plan
        sibling._rng = np.random.default_rng(self.config.seed)
        return sibling


@dataclass
class HFTimeSeriesDatasetSpec:
    """Specification for converting a HuggingFace ``Dataset`` row into a recording.

    Attributes
    ----------
    target_fields
        Column names whose contents are stacked along the variate axis. Each
        column is expected to be either 1-D (length ``T``) or 2-D
        (``(n_subvariates, T)``).
    """

    target_fields: Sequence[str] = field(default_factory=lambda: ["target"])


class HFTimeSeriesDataset(Dataset):
    """Sliding-window dataset over a HuggingFace ``datasets.Dataset``.

    The underlying HF dataset can be arbitrarily large (memory-mapped Arrow);
    we materialize each row only when sampled.

    Parameters
    ----------
    hf_dataset
        A ``datasets.Dataset`` instance (must support ``__getitem__`` /
        ``__len__``). Each row contains one or more time-series fields.
    config
        Sampling configuration shared with :class:`ArrayTimeSeriesDataset`.
    spec
        Field-mapping specification. See :class:`HFTimeSeriesDatasetSpec`.
    transform
        Optional callable applied to each sampled window.
    nan_to_num
        See :class:`ArrayTimeSeriesDataset`.
    """

    def __init__(
        self,
        hf_dataset: Any,
        config: SlidingWindowConfig,
        *,
        spec: Optional[HFTimeSeriesDatasetSpec] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        nan_to_num: bool = True,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.config = config
        self.spec = spec or HFTimeSeriesDatasetSpec()
        self.transform = transform
        self.nan_to_num = nan_to_num
        self._rng = np.random.default_rng(config.seed)

        # Cache row lengths so we know which rows are long enough.
        # We cap inspection to the first 10k rows to keep init cheap; rows
        # discovered to be too short at __getitem__ are resampled.
        self._row_indices: list[int] = []
        probe = min(len(hf_dataset), 10_000)
        for i in range(probe):
            row = hf_dataset[i]
            length = self._row_length(row)
            if length >= config.window_len:
                self._row_indices.append(i)
        if probe < len(hf_dataset):
            # Remaining rows are tentatively included; filter at sample time.
            self._row_indices.extend(range(probe, len(hf_dataset)))
        if not self._row_indices:
            raise ValueError(
                f"No HF dataset row is long enough for window_len={config.window_len}."
            )

    def _row_length(self, row: dict[str, Any]) -> int:
        first = row[self.spec.target_fields[0]]
        arr = np.asarray(first)
        return arr.shape[-1] if arr.ndim > 0 else 0

    def _row_to_tensor(self, row: dict[str, Any]) -> torch.Tensor:
        stacks: list[torch.Tensor] = []
        for field_name in self.spec.target_fields:
            arr = np.asarray(row[field_name], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                raise ValueError(
                    f"Field {field_name!r} has unsupported shape {arr.shape}; "
                    "expected 1-D or 2-D."
                )
            stacks.append(torch.from_numpy(arr))
        return torch.cat(stacks, dim=0)  # (n_var_total, T)

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        attempts = 0
        while True:
            row_idx = self._row_indices[index % len(self._row_indices)]
            row = self.hf_dataset[int(row_idx)]
            recording = self._row_to_tensor(row)
            T = recording.shape[1]
            if T < self.config.window_len:
                attempts += 1
                index = int(self._rng.integers(0, len(self._row_indices)))
                if attempts > 16:
                    raise RuntimeError("Failed to find a row long enough; check your data.")
                continue

            if self.config.random:
                start = int(self._rng.integers(0, T - self.config.window_len + 1))
            else:
                start = 0
            window = recording[:, start : start + self.config.window_len]

            if self.transform is not None:
                window = self.transform(window)

            finite = torch.isfinite(window)
            if self.nan_to_num:
                window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            return {
                "target": window,
                "target_mask": finite,
                "series_ids": torch.full((recording.shape[0],), row_idx, dtype=torch.long),
            }


def make_eeg_dataset_from_npz(
    npz_paths: Sequence[str],
    config: SlidingWindowConfig,
    *,
    array_key: str = "data",
    sfreq_key: Optional[str] = None,
    expected_sfreq: Optional[float] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> ArrayTimeSeriesDataset:
    """Convenience builder for EEG recordings stored as ``.npz`` files.

    Each ``.npz`` is expected to contain an array under ``array_key`` of shape
    ``(channels, time_steps)``. If ``sfreq_key`` and ``expected_sfreq`` are
    both provided, recordings whose sampling frequency disagrees with the
    expectation are skipped (with a logged warning).
    """
    import logging

    log = logging.getLogger("toto2.training")
    recordings: list[np.ndarray] = []
    for path in npz_paths:
        with np.load(path) as f:
            if array_key not in f.files:
                log.warning("%s missing key %r; skipping.", path, array_key)
                continue
            arr = np.asarray(f[array_key], dtype=np.float32)
            if sfreq_key is not None and expected_sfreq is not None:
                if sfreq_key in f.files and not math.isclose(
                    float(f[sfreq_key]), float(expected_sfreq), rel_tol=1e-3
                ):
                    log.warning(
                        "%s has sfreq=%s; expected %s. Skipping.",
                        path,
                        float(f[sfreq_key]),
                        expected_sfreq,
                    )
                    continue
            recordings.append(arr)
    if not recordings:
        raise FileNotFoundError("No usable EEG recordings found.")
    return ArrayTimeSeriesDataset(recordings, config, transform=transform)
