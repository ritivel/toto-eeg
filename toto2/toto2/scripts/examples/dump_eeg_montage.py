#!/usr/bin/env python3
"""Dump a pre-computed EEG electrode montage JSON for runtime use.

The Toto 2.0 training pipeline does not depend on MNE at runtime (MNE is
heavy and version-fragile across the GPU fleet).  Instead the canonical
electrode positions for each supported montage are committed as JSON
alongside :mod:`toto2.training.montages`.  This script regenerates that
JSON from MNE's ``make_standard_montage`` so the data stays auditable.

Usage
-----

.. code-block:: bash

    python -m toto2.scripts.examples.dump_eeg_montage \\
        --montage GSN-HydroCel-129 \\
        --output toto2/toto2/training/montages/gsn_hydrocel_129.json

The default arguments produce the file shipped with the package; rerun
the script when MNE updates or when adding support for a new montage
(e.g., 10-20, 10-10, biosemi64).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _build_montage_payload(montage_name: str) -> dict:
    """Construct the JSON-serialisable payload for ``montage_name``."""
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()
    ch_names = list(montage.ch_names)
    xyz = np.array([positions["ch_pos"][n] for n in ch_names], dtype=np.float64)
    norms = np.linalg.norm(xyz, axis=1)
    if (norms == 0).any():
        raise ValueError(
            f"Montage {montage_name!r} has zero-norm electrode positions; "
            "cannot project onto the unit sphere."
        )
    unit_xyz = xyz / norms[:, None]
    return {
        "montage": montage_name,
        "mne_version": mne.__version__,
        "description": (
            f"Unit-sphere-normalised 3D positions for the {montage_name} montage. "
            "Generated once via mne.channels.make_standard_montage so training-"
            "time data loaders do not need to depend on MNE. Coordinates are "
            "right-handed: +X = right, +Y = anterior, +Z = superior."
        ),
        "channel_names": ch_names,
        "positions_unit_sphere": unit_xyz.tolist(),
        "positions_meters": xyz.tolist(),
        "meters_norm_stats": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dump an EEG montage JSON for Toto 2.0.")
    parser.add_argument(
        "--montage",
        default="GSN-HydroCel-129",
        help="MNE standard montage name (e.g. GSN-HydroCel-129, GSN-HydroCel-128, biosemi64).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON path (will be overwritten).",
    )
    args = parser.parse_args(argv)

    try:
        payload = _build_montage_payload(args.montage)
    except ImportError as e:
        print(f"error: {e}; install with `pip install mne`.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[dump_eeg_montage] wrote {args.output} ({len(payload['channel_names'])} channels, "
        f"mne={payload['mne_version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
