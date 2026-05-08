# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Variate-padding collation for Toto 2.0 training batches.

Toto 2.0 supports a variable number of variates per sample: the model
attends across variates with a block-diagonal id mask, where the value
``-1`` marks padding variates that are excluded from attention. This module
provides a collation function that stacks samples with potentially differing
``n_var`` into a single padded batch.

The time dimension is fixed (all samples share the same ``window_len``), so
no time padding is required.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


_DEFAULT_PAD_SERIES_ID = -1


def collate_timeseries(
    samples: Sequence[dict[str, torch.Tensor]],
    *,
    pad_series_id: int = _DEFAULT_PAD_SERIES_ID,
) -> dict[str, torch.Tensor]:
    """Collate a list of ``{target, target_mask, series_ids[, electrode_coords]}`` samples.

    Parameters
    ----------
    samples
        Output of an ``ArrayTimeSeriesDataset`` / ``HFTimeSeriesDataset`` /
        ``LazyNpzTimeSeriesDataset``.  Each sample is a dictionary with:

        - ``target`` of shape ``(n_var_i, window_len)``
        - ``target_mask`` of shape ``(n_var_i, window_len)``
        - ``series_ids`` of shape ``(n_var_i,)``
        - ``electrode_coords`` of shape ``(n_var_i, 3)`` (exp49, optional;
          must be present in either *every* sample or *no* sample of a
          batch — mixed batches are rejected so the model never sees a
          partial coord encoding).

    pad_series_id
        Series id used for padded variates. ``-1`` matches Toto 2.0's
        convention for "exclude from attention".

    Returns
    -------
    dict[str, torch.Tensor]
        Batched dictionary with:

        - ``target``: ``(B, n_var_max, window_len)``
        - ``target_mask``: ``(B, n_var_max, window_len)`` — also ``False``
          across the entire padding variate dimension.
        - ``series_ids``: ``(B, n_var_max)`` — padding variates carry
          ``pad_series_id``.
        - ``electrode_coords``: ``(B, n_var_max, 3)`` — only present
          when every input sample carried it. Padded variate rows
          contain zeros (origin); the trunk masks them out via
          ``series_ids == pad_series_id``.
        - ``num_variates``: ``(B,)`` — original ``n_var`` per sample.
    """
    if not samples:
        raise ValueError("Cannot collate an empty list of samples.")

    window_lens = {sample["target"].shape[-1] for sample in samples}
    if len(window_lens) != 1:
        raise ValueError(
            f"All samples must share the same window length; got {window_lens}."
        )
    window_len = window_lens.pop()

    n_var_max = max(sample["target"].shape[0] for sample in samples)
    batch_size = len(samples)

    target = torch.zeros(batch_size, n_var_max, window_len, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, n_var_max, window_len, dtype=torch.bool)
    series_ids = torch.full((batch_size, n_var_max), pad_series_id, dtype=torch.long)
    num_variates = torch.zeros(batch_size, dtype=torch.long)

    has_coords = ["electrode_coords" in s for s in samples]
    if any(has_coords) and not all(has_coords):
        raise ValueError(
            "electrode_coords must be present in either every sample of a "
            "batch or none — got a mixed batch.  Make sure your dataset "
            "consistently emits the field, or do not enable use_coord_pe."
        )
    electrode_coords: Optional[torch.Tensor] = None
    if all(has_coords):
        electrode_coords = torch.zeros(batch_size, n_var_max, 3, dtype=torch.float32)

    for b, sample in enumerate(samples):
        v = sample["target"].shape[0]
        target[b, :v] = sample["target"].to(torch.float32)
        target_mask[b, :v] = sample["target_mask"].to(torch.bool)
        series_ids[b, :v] = sample["series_ids"].to(torch.long)
        if electrode_coords is not None:
            ec = sample["electrode_coords"]
            if ec.shape != (v, 3):
                raise ValueError(
                    f"Sample {b} has electrode_coords shape {tuple(ec.shape)} "
                    f"but expected ({v}, 3)."
                )
            electrode_coords[b, :v] = ec.to(torch.float32)
        num_variates[b] = v

    out: dict[str, torch.Tensor] = {
        "target": target,
        "target_mask": target_mask,
        "series_ids": series_ids,
        "num_variates": num_variates,
    }
    if electrode_coords is not None:
        out["electrode_coords"] = electrode_coords
    return out
