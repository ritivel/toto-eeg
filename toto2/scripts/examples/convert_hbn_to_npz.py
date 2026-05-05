#!/usr/bin/env python3
"""Convert raw HBN-EEG ``.set`` files into Toto 2.0-ready ``.npz`` recordings.

Reads a BIDS-formatted HBN release tree (one or more) at the source path,
loads each ``.set`` (and optional sibling ``.fdt``) via MNE, and writes a
compressed ``.npz`` per recording with:

    data: (n_channels, n_samples) float32 — raw EEG in volts (MNE native unit)
    sfreq: float — sampling rate (Hz)
    channels: array<str> — channel names (length n_channels)
    subject_id: str
    task: str
    release: str

No filtering, resampling, windowing, or normalization is applied here —
Toto 2.0's causal patch scaler + asinh transform handle normalization
internally during training, and the sliding-window dataset handles
windowing. Keeping the converter minimal lets the model see the raw signal
distribution and is faster (no scipy.signal calls per recording).

Layout
------
Input (BIDS, what ``aws s3 sync`` of fcp-indi produces)::

    <src>/cmi_bids_<release>/
        sub-<NDARxxx>/
            eeg/
                sub-<NDARxxx>_task-<task>[_run-N]_eeg.set
                sub-<NDARxxx>_task-<task>[_run-N]_eeg.fdt    # optional
                sub-<NDARxxx>_task-<task>[_run-N]_*.tsv/.json
        ...

Output (one .npz per recording)::

    <dst>/sub-<NDARxxx>/
        task-<task>[_run-N].npz

Usage
-----

.. code-block:: bash

    python convert_hbn_to_npz.py \\
        --src /opt/dlami/nvme/eeg/raw/hbn \\
        --dst /opt/dlami/nvme/eeg/npz \\
        --workers 16

By default, processes every ``.set`` in every release under ``--src``.
Use ``--release R1`` to limit to one release, or ``--max-recordings N`` to
cap total recordings (smoke testing).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


_SET_RE = re.compile(
    r"^sub-(?P<subject>NDAR[A-Z0-9]+)_task-(?P<task>[A-Za-z][A-Za-z0-9]*?)"
    r"(?P<run>_run-\d+)?_eeg\.set$"
)


def _list_set_files(src: Path, releases: Optional[list[str]] = None) -> list[Path]:
    """Walk the BIDS tree and return all .set files."""
    out: list[Path] = []
    if releases is not None:
        bids_dirs = [src / f"cmi_bids_{r}" for r in releases]
    else:
        bids_dirs = sorted(src.glob("cmi_bids_*"))
    for bids_dir in bids_dirs:
        if not bids_dir.is_dir():
            continue
        for set_path in sorted(bids_dir.glob("sub-*/eeg/sub-*_task-*_eeg.set")):
            out.append(set_path)
    return out


def _parse_set_path(set_path: Path) -> tuple[str, str, str]:
    """Return (release, subject_id, task_with_run) for a BIDS .set path."""
    m = _SET_RE.match(set_path.name)
    if not m:
        raise ValueError(f"Filename does not match BIDS pattern: {set_path.name}")
    subject = m.group("subject")
    task = m.group("task")
    run = m.group("run") or ""
    task_with_run = f"task-{task}{run}"
    release_dir = set_path.parents[2].name
    if not release_dir.startswith("cmi_bids_"):
        raise ValueError(f"Cannot infer release from path {set_path}")
    release = release_dir[len("cmi_bids_") :]
    return release, subject, task_with_run


def _convert_one(args: tuple[str, str, bool]) -> dict[str, object]:
    """Worker: load one .set via MNE and write the .npz.

    Returns a dict with ``ok`` plus diagnostics. Errors do not raise — they
    are returned as ``ok=False`` so a single bad recording does not abort
    the whole run.
    """
    set_path_s, dst_root_s, overwrite = args
    set_path = Path(set_path_s)
    dst_root = Path(dst_root_s)
    try:
        release, subject, task_run = _parse_set_path(set_path)
    except ValueError as e:
        return {"ok": False, "path": set_path_s, "error": f"parse: {e}"}

    out_dir = dst_root / f"sub-{subject}"
    out_path = out_dir / f"{task_run}.npz"
    if out_path.exists() and not overwrite:
        return {
            "ok": True,
            "path": set_path_s,
            "out": str(out_path),
            "skipped": True,
        }

    try:
        # MNE imports are deferred to the worker so the main process does
        # not have to import mne (which is heavy).
        import mne

        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        data = raw.get_data().astype(np.float32, copy=False)
        sfreq = float(raw.info["sfreq"])
        channels = np.asarray(list(raw.ch_names), dtype=object)
    except Exception as e:  # noqa: BLE001 — we intentionally swallow per-file errors
        return {
            "ok": False,
            "path": set_path_s,
            "error": f"mne_load: {e}\n{traceback.format_exc(limit=2)}",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        np.savez_compressed(
            tmp_path,
            data=data,
            sfreq=np.float32(sfreq),
            channels=channels,
            subject_id=np.str_(subject),
            task=np.str_(task_run),
            release=np.str_(release),
        )
        tmp_path.replace(out_path)
    except Exception as e:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return {"ok": False, "path": set_path_s, "error": f"write: {e}"}

    return {
        "ok": True,
        "path": set_path_s,
        "out": str(out_path),
        "shape": tuple(data.shape),
        "sfreq": sfreq,
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "size_bytes": out_path.stat().st_size,
        "skipped": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert HBN .set -> .npz for Toto 2.0.")
    parser.add_argument("--src", type=Path, required=True,
                        help="Source root containing cmi_bids_* directories.")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Destination root for sub-<NDAR...>/<task>.npz.")
    parser.add_argument("--release", action="append", default=None,
                        help="Restrict to one release (e.g. R1). Repeatable.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
                        help="Number of parallel worker processes.")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Cap to first N recordings (smoke test).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-write .npz even if it already exists.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Write a JSONL manifest of converted recordings.")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print a status line every N recordings.")
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"error: --src does not exist: {args.src}", file=sys.stderr)
        return 2

    set_files = _list_set_files(args.src, args.release)
    if args.max_recordings is not None:
        set_files = set_files[: args.max_recordings]
    if not set_files:
        print(f"error: no .set files found under {args.src}", file=sys.stderr)
        return 2

    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"[convert] {len(set_files)} .set files; workers={args.workers}; dst={args.dst}",
          flush=True)

    manifest_fh = open(args.manifest, "a") if args.manifest is not None else None
    n_ok = 0
    n_fail = 0
    n_skip = 0
    bytes_out = 0
    t0 = time.time()

    work = [(str(p), str(args.dst), args.overwrite) for p in set_files]

    def _handle(result: dict[str, object]) -> None:
        nonlocal n_ok, n_fail, n_skip, bytes_out
        if result.get("ok"):
            n_ok += 1
            if result.get("skipped"):
                n_skip += 1
            else:
                bytes_out += int(result.get("size_bytes", 0) or 0)
        else:
            n_fail += 1
            print(f"  [fail] {result.get('path')}: {result.get('error')}", flush=True)
        if manifest_fh is not None:
            manifest_fh.write(json.dumps(result) + "\n")
        n_seen = n_ok + n_fail
        if n_seen % args.log_every == 0:
            dt = max(1e-3, time.time() - t0)
            rate = n_seen / dt
            print(
                f"[convert] {n_seen}/{len(set_files)} "
                f"(ok={n_ok}, skip={n_skip}, fail={n_fail}) "
                f"{bytes_out / 2**30:.2f} GiB out, {rate:.1f} rec/s",
                flush=True,
            )

    if args.workers <= 1:
        for w in work:
            _handle(_convert_one(w))
    else:
        # spawn keeps mne import isolated to workers and avoids torch/cuda issues.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_convert_one, work, chunksize=1):
                _handle(result)

    if manifest_fh is not None:
        manifest_fh.close()

    dt = time.time() - t0
    print(
        f"\n[convert] DONE: ok={n_ok} (skip={n_skip}) fail={n_fail} "
        f"in {dt:.1f}s; wrote {bytes_out / 2**30:.2f} GiB",
        flush=True,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
