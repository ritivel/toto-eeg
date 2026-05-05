# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""LightningDataModule for Toto 2.0 pre-training and continued pre-training.

Wraps a pair of ``Dataset`` instances (train + optional validation) into a
Lightning-native pipeline with proper sharding-aware DataLoaders.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .collate import collate_timeseries


class TimeSeriesDataModule(LightningDataModule):
    """Generic ``LightningDataModule`` for Toto 2.0 training.

    Parameters
    ----------
    train_dataset
        PyTorch ``Dataset`` returning Toto-shaped samples (see
        :mod:`toto2.training.datasets`).
    val_dataset
        Optional validation dataset.
    train_batch_size
        Per-rank training batch size. The effective global batch is
        ``train_batch_size * world_size * accumulate_grad_batches`` and must
        be reflected in :func:`dd_unit_scaling.init_world_size_cache` /
        :func:`dd_unit_scaling.set_grad_accumulation_steps` to keep u-μP
        scale factors correct.
    val_batch_size
        Per-rank validation batch size.
    num_workers
        DataLoader workers per process.
    pin_memory
        Whether to pin host memory in the DataLoader.
    persistent_workers
        Keep DataLoader workers alive across epochs (recommended for long
        runs to avoid worker startup overhead).
    drop_last
        Drop the final partial batch in training. Strongly recommended for
        DDP/FSDP to avoid uneven batches that change u-μP scale factors.
    pad_series_id
        Series id assigned to padding variates (default ``-1``).
    multiprocessing_context
        Optional name of the ``multiprocessing`` start method for DataLoader
        workers (``"fork"``, ``"forkserver"``, ``"spawn"``). Default ``"fork"``
        when ``num_workers > 0``: forking inherits the DDP-wrapped model
        without re-pickling, which avoids the
        ``Default process group has not been initialized`` error you get
        from spawn'd workers under Lightning DDP. Set explicitly to
        ``"spawn"`` if your dataset holds CUDA tensors / non-fork-safe state.
    prefetch_factor
        Number of batches each worker prefetches.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        *,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = True,
        pad_series_id: int = -1,
        multiprocessing_context: Optional[str] = "fork",
        prefetch_factor: Optional[int] = 2,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.drop_last = bool(drop_last)
        self.pad_series_id = int(pad_series_id)
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor

    def _collate(self, samples):
        return collate_timeseries(samples, pad_series_id=self.pad_series_id)

    def _dataloader_kwargs(self) -> dict:
        kw: dict = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self._collate,
        }
        if self.num_workers > 0:
            # Verify the requested mp context exists; fall back to default
            # if not (e.g. ``fork`` on Windows).
            ctx_name = self.multiprocessing_context
            if ctx_name and ctx_name in mp.get_all_start_methods():
                kw["multiprocessing_context"] = mp.get_context(ctx_name)
            kw["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kw["prefetch_factor"] = int(self.prefetch_factor)
        return kw

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            **self._dataloader_kwargs(),
        )
