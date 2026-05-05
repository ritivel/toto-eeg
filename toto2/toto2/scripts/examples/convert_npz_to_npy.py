#!/usr/bin/env python3
"""One-shot ``.npz`` -> ``.npy`` converter for fast memory-mapped EEG loading.

The training pipeline reads each ``.npz`` (a ZIP of per-key DEFLATE-compressed
.npy buffers) every iteration. Even with our lazy + cached loader, the
per-step decompression dominates wall-time on Hopper GPUs because:

- DEFLATE is single-threaded inside one ``np.load`` call
- The decompressed array (~50 MB float32) is allocated and copied each access
- We cannot ``np.memmap`` a compressed ZIP entry

This script materialises one ``.npy`` per recording on NVMe so the training
loop can ``np.memmap`` the files: zero-copy reads of the requested time slice,
no decompression. Tradeoff: disk usage roughly doubles (no compression) but
NVMe is plentiful (28 TB total on a p5.48xlarge) and read throughput goes
from ~50 MB/s (CPU bound on DEFLATE) to ~3-5 GB/s (raw NVMe).

Layout::

    --src /opt/dlami/nvme/eeg/npz/sub-NDARxxx/task-RestingState.npz
    --dst /opt/dlami/nvme/eeg/npy/sub-NDARxxx/task-RestingState.npy

Each ``.npy`` contains the ``(n_channels, n_samples)`` float32 array; channel
names / sfreq / metadata are dropped because the training loop does not need
them once each file is one fixed-rate recording. If you need them later, add
a sibling ``.json`` write to ``_convert_one``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _convert_one(args: tuple[str, str, bool, str]) -> dict[str, object]:
    """Worker: read one .npz, write one .npy."""
    src_s, dst_s, overwrite, array_key = args
    src = Path(src_s)
    dst = Path(dst_s)
    if dst.exists() and not overwrite:
        return {"ok": True, "src": src_s, "skipped": True}
    try:
        with np.load(src, allow_pickle=False) as f:
            if array_key not in f.files:
                return {"ok": False, "src": src_s, "error": f"missing key {array_key!r}"}
            arr = np.asarray(f[array_key], dtype=np.float32)
    except (OSError, ValueError, KeyError) as e:
        return {"ok": False, "src": src_s, "error": f"load: {e}"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    # ``np.save`` auto-appends ``.npy`` to the path if missing, so we need
    # the temp file to NOT already end in ``.npy``. We pass a basename
    # ending in ``.part`` and the actual saved file lands at ``<base>.npy``.
    tmp_base = dst.with_suffix(".part")
    tmp_written = tmp_base.with_suffix(".part.npy")
    try:
        np.save(str(tmp_base), arr, allow_pickle=False)
        tmp_written.replace(dst)
    except Exception as e:  # noqa: BLE001
        for p in (tmp_base, tmp_written, dst):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        return {"ok": False, "src": src_s, "error": f"write: {e}"}

    return {
        "ok": True,
        "src": src_s,
        "dst": str(dst),
        "shape": tuple(arr.shape),
        "size_bytes": dst.stat().st_size,
        "skipped": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=".npz -> .npy converter for mmap loading.")
    parser.add_argument("--src", type=Path, required=True,
                        help="Root containing sub-*/task-*.npz")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output root for sub-*/task-*.npy")
    parser.add_argument("--array-key", default="data",
                        help="Key inside the .npz holding the EEG array.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap to first N files (smoke test).")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"error: --src does not exist: {args.src}", file=sys.stderr)
        return 2
    args.dst.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src.glob("**/*.npz"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(f"error: no .npz files under {args.src}", file=sys.stderr)
        return 2

    work: list[tuple[str, str, bool, str]] = []
    for src in files:
        rel = src.relative_to(args.src).with_suffix(".npy")
        work.append((str(src), str(args.dst / rel), args.overwrite, args.array_key))

    print(f"[npz2npy] {len(work)} files; workers={args.workers}; dst={args.dst}", flush=True)

    n_ok = n_fail = n_skip = 0
    bytes_out = 0
    t0 = time.time()

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
            print(f"  [fail] {result.get('src')}: {result.get('error')}", flush=True)
        seen = n_ok + n_fail
        if seen % args.log_every == 0:
            dt = max(1e-3, time.time() - t0)
            print(
                f"[npz2npy] {seen}/{len(work)} ok={n_ok} skip={n_skip} fail={n_fail} "
                f"{bytes_out / 2**30:.2f} GiB out, {seen / dt:.1f} files/s",
                flush=True,
            )

    if args.workers <= 1:
        for w in work:
            _handle(_convert_one(w))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_convert_one, work, chunksize=4):
                _handle(result)

    dt = time.time() - t0
    print(
        f"\n[npz2npy] DONE: ok={n_ok} (skip={n_skip}) fail={n_fail} in {dt:.1f}s; "
        f"wrote {bytes_out / 2**30:.2f} GiB",
        flush=True,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
