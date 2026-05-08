# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Pre-computed EEG electrode montages used by Toto 2.0 training.

Montages live as JSON files alongside this module so the training pipeline
does not depend on MNE at runtime (MNE is heavy and pinning its version
across the AWS / Mumbai p5.48xlarge fleet is fragile).  Each JSON is
written once via ``scripts/dump_eeg_montage.py`` (see that script for the
exact MNE call) and then committed to the repo.

Two coordinate systems are stored per montage:

* ``positions_meters``      — raw output of ``mne.channels.make_standard_montage``,
  in metres relative to the head sphere centre.  Useful for diagnostic
  re-projection / debugging only.
* ``positions_unit_sphere`` — each row of ``positions_meters`` divided
  by its L2 norm so every electrode lives on the unit 2-sphere.
  This is the canonical input expected by :class:`toto2.model.CoordPE`
  (random Fourier features and spherical harmonics both assume unit-
  sphere geometry).

The :func:`load_montage` and :func:`load_unit_sphere_positions` helpers
expose this data without re-loading the JSON every call (a tiny in-
process LRU cache makes the lookup essentially free for DataLoader
workers).
"""

from __future__ import annotations

import functools as _functools
import json
from importlib.resources import files
from typing import TypedDict

import numpy as np


__all__ = [
    "MontageData",
    "load_montage",
    "load_unit_sphere_positions",
    "list_available_montages",
]


class MontageData(TypedDict):
    montage: str
    mne_version: str
    description: str
    channel_names: list[str]
    positions_unit_sphere: list[list[float]]
    positions_meters: list[list[float]]
    meters_norm_stats: dict[str, float]


@_functools.lru_cache(maxsize=8)
def load_montage(name: str) -> MontageData:
    """Load a precomputed EEG montage JSON shipped alongside this module.

    Parameters
    ----------
    name
        Montage identifier (without ``.json`` extension).  Currently
        ships ``"gsn_hydrocel_129"``.

    Returns
    -------
    MontageData
        TypedDict matching the shape of the on-disk JSON.

    Raises
    ------
    FileNotFoundError
        If ``name`` does not match a JSON file inside this package.
    """
    path = files("toto2.training.montages") / f"{name}.json"
    if not path.is_file():  # type: ignore[attr-defined]
        available = list_available_montages()
        raise FileNotFoundError(
            f"No montage {name!r}; available: {available}."
        )
    with path.open("r") as f:  # type: ignore[attr-defined]
        data: MontageData = json.load(f)
    expected_keys = {
        "montage",
        "channel_names",
        "positions_unit_sphere",
        "positions_meters",
    }
    missing = expected_keys - set(data.keys())
    if missing:
        raise ValueError(
            f"Montage JSON {name!r} is missing required keys {sorted(missing)!r}."
        )
    return data


def load_unit_sphere_positions(name: str) -> np.ndarray:
    """Return the ``(n_channels, 3)`` unit-sphere positions for a montage.

    The returned array is a fresh ``float32`` copy so callers can mutate
    it freely without poisoning the LRU cache.
    """
    data = load_montage(name)
    return np.asarray(data["positions_unit_sphere"], dtype=np.float32).copy()


def list_available_montages() -> list[str]:
    """List the montages packaged with this module."""
    package = files("toto2.training.montages")
    out: list[str] = []
    for entry in package.iterdir():  # type: ignore[attr-defined]
        if entry.name.endswith(".json"):
            out.append(entry.name[:-5])
    return sorted(out)
