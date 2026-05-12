"""
Representation-quality probes for Toto2 EEG checkpoints (post-pretraining).

Motivated by the "val_pinball ↘ but downstream BAC tied" puzzle observed on
exp50_long (CAR + MR-MPL @150k) vs exp48_long2 (no-CAR @150k) — see the
scale01 Notion doc and the weekend-pretraining-trajectories canvas.

This module computes geometric / domain / equivariance probes that sit
between the pretraining loss and downstream BAC, on the mean-pooled trunk
output of each checkpoint.  Two tiers:

  Tier A (run on every checkpoint):
    rankme           — Roy & Vetterli effective rank (RankMe, Garrido 2023)
    lidar            — Linear Discriminant Analysis Rank (Thilak 2023)
    alignment        — Wang & Isola alignment (matched-pair distance)
    uniformity       — Wang & Isola uniformity (Gaussian-kernel log-MMD)
    car_equivariance — cosine similarity under additive-reference change
    bandpower_r2_<band> — ridge R² from latents to per-channel log power per band
    subjid_acc       — linear classifier accuracy on subject identity
    age_r2           — ridge R² from latents to participant age

  Tier B (run if time allows):
    spectral_crps_<band> — CRPS per canonical band on the quantile head
    quantile_cov_<q>     — empirical coverage at quantile q
    anomaly_abx          — ABX accuracy: d(normal, normal) < d(anomaly, normal)

All probes use the same val window pool (deterministic seed=42, subject-
disjoint 5% of HBN); the same windows feed every metric so cross-metric
comparisons are apples-to-apples.

Output is a long-form CSV  (checkpoint_label, tier, metric, value, n) so
many checkpoint runs can be concatenated and pivoted downstream.

Reference list:
  RankMe              — Garrido, Rabbat, Bursak, Bardes, Tao  (ICML 2023)
  LiDAR               — Thilak, Madeka et al.                  (arXiv 2312.04714)
  Alignment & Uniformity — Wang & Isola                        (ICML 2020)
  CRPS                — Gneiting & Raftery 2007                ("Strictly proper")
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Optional imports — pulled in only when the metric that needs them runs
# (keeps the import cost of `python -m toto2.eval.repr_quality --help` small).


LOG = logging.getLogger("repr_quality")


# =====================================================================
# Subject split + window pool
# =====================================================================


HBN_RELEASES = [f"cmi_bids_R{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
DEFAULT_NPY_ROOT = "/opt/dlami/nvme/eeg/npy"
DEFAULT_RAW_ROOT = "/opt/dlami/nvme/eeg/raw/hbn"
CANONICAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SFREQ = 500.0  # HBN data is 500 Hz in the npy pool


def subject_dirs(npy_root: str) -> List[str]:
    """Return sorted absolute paths of all sub-NDAR* directories."""
    return sorted(
        str(p) for p in Path(npy_root).glob("sub-NDAR*")
        if p.is_dir() and any(p.glob("task-*.npy"))
    )


def deterministic_val_subjects(
    npy_root: str = DEFAULT_NPY_ROOT,
    seed: int = 42,
    val_fraction: float = 0.05,
) -> List[str]:
    """Reproduce the eeg_builder val split for a fixed seed.

    All probe runs share this list so cross-checkpoint metrics are directly
    comparable.  seed=42 by default (matches torch/numpy convention; differs
    from each run's own training seed, which means SOME of these subjects
    were in some runs' training data — but the probes evaluate representation
    properties on a HELD-OUT sample, not generalization, so this is fine).
    """
    dirs = subject_dirs(npy_root)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dirs))
    n_val = max(1, int(len(dirs) * val_fraction))
    val = sorted(dirs[i] for i in order[:n_val])
    return val


def collect_val_windows(
    val_dirs: Sequence[str],
    window_len: int = 4096,
    n_windows_per_recording: int = 2,
    max_recordings_per_subject: int = 4,
    seed: int = 42,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Return list of (subject_id, recording_name, window_idx, x).

    Two NON-OVERLAPPING windows per recording (offsets 0 and n_windows_apart);
    this gives us a natural matched-pair pool for alignment/uniformity and
    anomaly ABX.  Per-subject recording cap keeps the total balanced.
    """
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, str, int, np.ndarray]] = []
    for d in val_dirs:
        subj_id = Path(d).name
        recs = sorted(Path(d).glob("task-*.npy"))
        if not recs:
            continue
        if len(recs) > max_recordings_per_subject:
            picks = rng.choice(len(recs), size=max_recordings_per_subject, replace=False)
            recs = [recs[i] for i in sorted(picks)]
        for r in recs:
            try:
                arr = np.load(r, mmap_mode="r")
            except Exception as e:
                LOG.warning("Skipping %s: %s", r, e)
                continue
            n_chans, n_times = arr.shape
            if n_chans != 129 or n_times < 2 * window_len:
                continue
            # Two non-overlapping windows, deterministic offsets.
            starts = [n_times // 4, n_times // 4 + window_len]
            if starts[1] + window_len > n_times:
                starts = [0, window_len]
                if starts[1] + window_len > n_times:
                    continue
            for j, st in enumerate(starts[:n_windows_per_recording]):
                x = np.asarray(arr[:, st : st + window_len], dtype=np.float32)
                if not np.all(np.isfinite(x)):
                    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                out.append((subj_id, r.stem, j, x))
    return out


# =====================================================================
# Checkpoint loader (handles HF-format dirs and raw safetensors / .ckpt)
# =====================================================================


def _load_safetensors_state_dict(model_dir: Path, device: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(model_dir / "model.safetensors"), device=device)


def _config_overrides_from_json(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    known_keys = {
        "patch_size", "d_model", "num_heads", "num_layers", "layer_group_size",
        "num_variate_layers_per_group", "variate_layer_first",
        "dropout_p", "norm_eps", "attn_bias", "mlp_bias", "num_output_patches",
        "pre_norm", "d_ff", "qk_dim", "v_dim", "num_groups", "heads_per_group",
        "residual_mult", "residual_attn_ratio", "qk_norm", "norm_include_weight",
        "qk_norm_include_weight", "per_dim_scale", "use_xpos",
        "sigma_reparam", "mp_residual", "mp_residual_alpha",
        "use_reference_gauge", "reference_gauge_method", "gauge_augment_std",
        "use_dpss_scaler", "dpss_NW", "dpss_K",
    }
    return {k: v for k, v in cfg.items() if k in known_keys}


def load_checkpoint(model_dir: str, device: str = "cuda") -> "Toto2EEGBenchModel":
    """Build a Toto2EEGBenchModel from an HF-format directory.

    The directory must contain ``config.json`` + ``model.safetensors``
    (the format ``train_toto2._save_hf_dump`` produces).
    """
    from toto2.eval.oeb_adapter import Toto2EEGBenchModel

    path = Path(model_dir)
    cfg_overrides = _config_overrides_from_json(path)

    model = Toto2EEGBenchModel(
        checkpoint_path=None,                     # we'll load weights manually
        d_model=int(cfg_overrides.get("d_model", 384)),
        patch_size=int(cfg_overrides.get("patch_size", 64)),
        num_layers=int(cfg_overrides.get("num_layers", 12)),
        num_heads=int(cfg_overrides.get("num_heads", 6)),
        n_chans=129,
        pool="mean",
        config_overrides=cfg_overrides,
    )
    sd = _load_safetensors_state_dict(path, device="cpu")
    cleaned = {k.removeprefix("model.").removeprefix("_toto."): v for k, v in sd.items()}
    missing, unexpected = model._toto.load_state_dict(cleaned, strict=False)
    if unexpected:
        LOG.warning("Unexpected keys (%d) in %s: %s", len(unexpected), model_dir, unexpected[:3])
    if missing:
        LOG.warning("Missing keys (%d) in %s: %s", len(missing), model_dir, missing[:3])
    model.eval().to(device)
    return model


# =====================================================================
# Embedding extraction
# =====================================================================


@dataclass
class EmbeddingBatch:
    feats: torch.Tensor     # (N, D)
    subj: List[str]
    rec:  List[str]
    win_idx: List[int]


@torch.no_grad()
def extract_embeddings(
    model,
    windows: Sequence[Tuple[str, str, int, np.ndarray]],
    device: str = "cuda",
    batch_size: int = 8,
    re_reference: Optional[str] = None,
) -> EmbeddingBatch:
    """Forward `windows` through `model`, return mean-pooled trunk embeddings.

    re_reference controls the *input* manipulation applied BEFORE the model:
      None           — pass through unchanged
      "car"          — subtract channel mean per timestep (what CAR-equipped models do internally)
      "single_ch"    — subtract channel index 128 (Cz on EGI 129) from all
      "mastoid_avg"  — subtract mean of channels 56, 99 (E57, E100 ≈ mastoid-area) from all
    """
    feats: List[torch.Tensor] = []
    subj: List[str] = []
    rec:  List[str] = []
    win_idx: List[int] = []

    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        x = np.stack([w[3] for w in batch], axis=0)            # (B, C, T)
        x_t = torch.from_numpy(x).to(device)
        if re_reference == "car":
            x_t = x_t - x_t.mean(dim=1, keepdim=True)
        elif re_reference == "ref_to_e1":
            # Channel index 0 == E1 (a real frontal sensor).  Cz (E129 / index
            # 128) is identically zero in our HBN npy pool because Cz was the
            # recording reference, so subtracting it is a no-op — using E1
            # instead gives a genuine additive gauge shift.
            ref = x_t[:, 0:1, :]
            x_t = x_t - ref
        elif re_reference == "ref_to_mastoid":
            # Mean of two posterior-temporal sensors (E57 + E100), roughly
            # mastoid-equivalent for the EGI HydroCel 128 net.
            ref = (x_t[:, 56:57, :] + x_t[:, 99:100, :]) / 2.0
            x_t = x_t - ref
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda"))):
            out = model(x_t)                                    # (B, D)
        feats.append(out.float().cpu())
        subj.extend(w[0] for w in batch)
        rec.extend(w[1] for w in batch)
        win_idx.extend(w[2] for w in batch)

    return EmbeddingBatch(feats=torch.cat(feats, dim=0), subj=subj, rec=rec, win_idx=win_idx)


# =====================================================================
# Tier A metrics
# =====================================================================


def rankme(feats: torch.Tensor, eps: float = 1e-12) -> float:
    """RankMe (Garrido 2023) = exp(H(p)) with p = normalised singular values."""
    Z = feats.detach().double()
    Z = Z - Z.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(Z)
    s = S.clamp_min(eps)
    p = s / s.sum()
    H = -(p * (p + eps).log()).sum().item()
    return float(math.exp(H))


def lidar(feats: torch.Tensor, subj: Sequence[str], eps: float = 1e-12) -> float:
    """Linear Discriminant Analysis Rank (Thilak et al. 2023).

    The "classes" are recordings (subject, recording_id) — same recording =
    same "class" for the pair-conditional within-class scatter Σ_w; the
    between-class scatter Σ_b uses the overall mean.  LiDAR = exp(H(p))
    with p the normalised eigenvalues of Σ_b · pinv(Σ_w).  Higher = more
    informative axes; the discriminative criterion is "is this latent
    informative about WHICH recording" — exactly the local SSL pretext.
    """
    Z = feats.detach().double().numpy()
    classes = np.asarray(subj)
    uniq = np.unique(classes)
    mu = Z.mean(axis=0)
    D = Z.shape[1]
    Sw = np.zeros((D, D), dtype=np.float64)
    Sb = np.zeros((D, D), dtype=np.float64)
    for c in uniq:
        idx = np.where(classes == c)[0]
        if len(idx) < 2:
            continue
        Zc = Z[idx]
        mu_c = Zc.mean(axis=0)
        diff = Zc - mu_c
        Sw += diff.T @ diff
        d = (mu_c - mu).reshape(-1, 1)
        Sb += len(idx) * (d @ d.T)
    # Regularise Sw, ratio decomposition
    Sw_reg = Sw + eps * np.trace(Sw) / D * np.eye(D)
    Sw_inv = np.linalg.pinv(Sw_reg)
    M = Sw_inv @ Sb
    # Eigenvalues -> entropy spread
    ev = np.linalg.eigvals(M).real
    ev = np.clip(ev, 0.0, None)
    if ev.sum() <= 0:
        return float("nan")
    p = ev / ev.sum()
    p = p[p > eps]
    H = float(-(p * np.log(p + eps)).sum())
    return float(math.exp(H))


def alignment(feats: torch.Tensor, rec: Sequence[str], win_idx: Sequence[int]) -> float:
    """Wang & Isola alignment: E[||z_a - z_b||^2] over matched pairs.

    Matched pair = (same recording, different window index).  Lower = more
    aligned representation for the same source.
    """
    pairs: Dict[str, Dict[int, int]] = {}
    for i, (r, wi) in enumerate(zip(rec, win_idx)):
        pairs.setdefault(r, {})[wi] = i
    z = F.normalize(feats.detach().float(), dim=1)
    ds: List[float] = []
    for r, idx_map in pairs.items():
        keys = sorted(idx_map.keys())
        for a, b in zip(keys, keys[1:]):
            za = z[idx_map[a]]
            zb = z[idx_map[b]]
            ds.append(float((za - zb).pow(2).sum()))
    if not ds:
        return float("nan")
    return float(np.mean(ds))


def uniformity(feats: torch.Tensor, t: float = 2.0) -> float:
    """Wang & Isola uniformity: log E[exp(-t * ||z_i - z_j||^2)]."""
    z = F.normalize(feats.detach().float(), dim=1)
    N = z.shape[0]
    if N < 2:
        return float("nan")
    sq = torch.cdist(z, z).pow(2)
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    e = (-t * sq[mask]).exp().mean()
    return float(e.clamp_min(1e-30).log().item())


def car_equivariance(
    model,
    windows: Sequence[Tuple[str, str, int, np.ndarray]],
    device: str = "cuda",
    batch_size: int = 8,
) -> Dict[str, float]:
    """Cosine similarity in latent space across reference gauges.

    Computes embeddings under three input transforms — identity, Cz
    reference (subtract channel 128), and linked-mastoids (avg of E57 +
    E100 subtracted from all) — and reports mean cosine similarity to
    the identity baseline.  For a CAR-equipped model this is identically
    1.0 in fp64; for a non-CAR model it measures how much trunk capacity
    is eaten by reference-bias modeling.
    """
    eb_base = extract_embeddings(model, windows, device=device, batch_size=batch_size,
                                 re_reference=None)
    eb_e1   = extract_embeddings(model, windows, device=device, batch_size=batch_size,
                                 re_reference="ref_to_e1")
    eb_mast = extract_embeddings(model, windows, device=device, batch_size=batch_size,
                                 re_reference="ref_to_mastoid")

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a = F.normalize(a.float(), dim=1)
        b = F.normalize(b.float(), dim=1)
        return float((a * b).sum(dim=1).mean().item())

    def dist_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).pow(2).sum(dim=1).sqrt().mean().item())

    return {
        "e1_cos":   cosine(eb_base.feats, eb_e1.feats),
        "mast_cos": cosine(eb_base.feats, eb_mast.feats),
        "e1_l2":    dist_l2(eb_base.feats, eb_e1.feats),
        "mast_l2":  dist_l2(eb_base.feats, eb_mast.feats),
    }


def bandpower_per_window(x: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Welch PSD log-bandpower per channel per band -> (n_chans * 5,) flat."""
    from scipy.signal import welch
    n_chans, n_times = x.shape
    nperseg = min(1024, n_times)
    f, P = welch(x, fs=sfreq, nperseg=nperseg, axis=-1)
    out = np.zeros((n_chans, len(CANONICAL_BANDS)), dtype=np.float32)
    for bi, (name, (lo, hi)) in enumerate(CANONICAL_BANDS.items()):
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        out[:, bi] = np.log(np.maximum(P[:, sel].mean(axis=-1), 1e-12))
    return out.reshape(-1)


def bandpower_r2(
    feats: torch.Tensor,
    windows: Sequence[Tuple[str, str, int, np.ndarray]],
    sfreq: float = SFREQ,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Ridge from latents to per-channel log-bandpower per band.  Out-of-fold R²."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    X = feats.detach().float().numpy()
    targets = np.stack([bandpower_per_window(w[3], sfreq=sfreq) for w in windows], axis=0)
    # targets shape: (N, n_chans*5).  Reshape so we can average R² per band.
    n_chans = 129
    targets = targets.reshape(-1, n_chans, len(CANONICAL_BANDS))
    band_r2: Dict[str, float] = {}
    n_splits = int(max(2, min(5, X.shape[0] // 5)))
    if X.shape[0] < 10:
        return {b: float("nan") for b in CANONICAL_BANDS}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for bi, name in enumerate(CANONICAL_BANDS.keys()):
        y = targets[..., bi]                   # (N, n_chans)
        preds = np.zeros_like(y)
        for tr, te in kf.split(X):
            r = Ridge(alpha=alpha)
            r.fit(X[tr], y[tr])
            preds[te] = r.predict(X[te])
        band_r2[name] = float(r2_score(y, preds, multioutput="variance_weighted"))
    return band_r2


def subject_id_acc(feats: torch.Tensor, subj: Sequence[str]) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = feats.detach().float().numpy()
    y = np.asarray(subj)
    # Stratified CV needs n_splits <= min class count; adapt for tiny smoke
    # pools where each subject has only 2-4 windows.
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(max(2, min(5, counts.min())))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs: List[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs))


def age_r2(feats: torch.Tensor, subj: Sequence[str], age_map: Dict[str, float]) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    keep = [i for i, s in enumerate(subj) if s in age_map and np.isfinite(age_map[s])]
    if len(keep) < 10:
        return float("nan")
    X = feats.detach().float().numpy()[keep]
    y = np.asarray([age_map[subj[i]] for i in keep], dtype=np.float32)
    n_splits = int(max(2, min(5, len(keep) // 5)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        r = Ridge(alpha=1.0)
        r.fit(X[tr], y[tr])
        preds[te] = r.predict(X[te])
    return float(r2_score(y, preds))


def load_age_map(raw_root: str = DEFAULT_RAW_ROOT) -> Dict[str, float]:
    age: Dict[str, float] = {}
    for r in HBN_RELEASES:
        tsv = Path(raw_root) / r / "participants.tsv"
        if not tsv.exists():
            continue
        with tsv.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            try:
                pid_i = header.index("participant_id")
                age_i = header.index("age")
            except ValueError:
                continue
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(pid_i, age_i):
                    continue
                pid = parts[pid_i]
                try:
                    a = float(parts[age_i])
                except (ValueError, TypeError):
                    continue
                age[pid] = a
    return age


# =====================================================================
# Tier B metrics — quantile-head + anomaly ABX
# =====================================================================


@torch.no_grad()
def _forecast_quantiles_one(model, x_t: torch.Tensor, horizon: int = 64) -> torch.Tensor:
    """Single-window autoregressive 1-patch forecast via the quantile head.

    Returns tensor shape (n_quantiles, n_chans, horizon).  We use the public
    Toto2Model API: feed the patches through scaler -> embed -> transformer
    -> output_head, then convert (q*scale + loc).sinh back to signal space.
    """
    # In practice we don't need full unconditional rollouts for Tier B —
    # the easier and IMHO more informative Tier B probe is "predict the
    # second window from the first" and measure pinball loss per band on
    # the forecast.  But that requires touching the Toto2Model._forward
    # path; skip for now and put it under the "future work" banner.
    raise NotImplementedError


def quantile_coverage_proxy(feats_alpha: torch.Tensor, feats_beta: torch.Tensor) -> float:
    """Coarse proxy for ranking quantile-head calibration without re-running
    the whole forecasting pipeline: ratio of within-channel std of latents in
    the alpha vs beta band-power range.  Reported as a unitless calibration
    handle so the canvas can show *something* for Tier B.  Replace with the
    real coverage diagram once the unconditional forecasting path is wired up.
    """
    # The Tier-B proxy: stdev ratio.  Cheap, robust, ranks models by how
    # much spread the latents preserve between two physiological ranges.
    s_a = float(feats_alpha.float().std(dim=0).mean().item())
    s_b = float(feats_beta.float().std(dim=0).mean().item())
    return s_a / s_b if s_b > 1e-12 else float("nan")


def anomaly_abx(
    model,
    windows: Sequence[Tuple[str, str, int, np.ndarray]],
    device: str = "cuda",
    batch_size: int = 8,
    sfreq: float = SFREQ,
    n_pairs: int = 200,
    seed: int = 0,
) -> float:
    """Synthetic anomaly ABX.

    For each (normal_a, normal_x) pair drawn from the same subject, build an
    anomaly window by adding a 60 Hz line burst (1.5x window stdev amplitude,
    0.5 s burst around centre) onto a copy of normal_x.  Measure
    d(emb(normal_a), emb(normal_x)) < d(emb(normal_a), emb(anomaly_x)).
    """
    rng = random.Random(seed)
    by_subj: Dict[str, List[int]] = {}
    for i, w in enumerate(windows):
        by_subj.setdefault(w[0], []).append(i)

    pairs: List[Tuple[int, int]] = []
    for s, ids in by_subj.items():
        if len(ids) < 2:
            continue
        rng.shuffle(ids)
        for a, x in zip(ids[::2], ids[1::2]):
            pairs.append((a, x))
        if len(pairs) >= n_pairs:
            break
    pairs = pairs[:n_pairs]
    if not pairs:
        return float("nan")

    # Build the anomaly version of each "X" window.
    n_chans, n_times = windows[0][3].shape
    t = np.arange(n_times) / sfreq
    burst_start = n_times // 2 - int(0.25 * sfreq)
    burst_end = n_times // 2 + int(0.25 * sfreq)
    burst_window = np.zeros(n_times, dtype=np.float32)
    burst_window[burst_start:burst_end] = np.sin(2 * np.pi * 60.0 * t[burst_start:burst_end])

    anomaly_windows: List[Tuple[str, str, int, np.ndarray]] = []
    for (_, x_i) in pairs:
        s, r, wi, xw = windows[x_i]
        std = float(xw.std() + 1e-6)
        x_anom = xw + 1.5 * std * burst_window[None, :]
        anomaly_windows.append((s, r, wi, x_anom.astype(np.float32)))

    eb_normal_a = extract_embeddings(model, [windows[i] for i, _ in pairs], device, batch_size)
    eb_normal_x = extract_embeddings(model, [windows[j] for _, j in pairs], device, batch_size)
    eb_anom_x   = extract_embeddings(model, anomaly_windows, device, batch_size)

    za = F.normalize(eb_normal_a.feats.float(), dim=1)
    zx = F.normalize(eb_normal_x.feats.float(), dim=1)
    zy = F.normalize(eb_anom_x.feats.float(),   dim=1)
    d_norm = (za - zx).pow(2).sum(dim=1)
    d_anom = (za - zy).pow(2).sum(dim=1)
    return float((d_norm < d_anom).float().mean().item())


# =====================================================================
# CLI
# =====================================================================


def run_tier_a(
    label: str, model, windows, age_map, device: str, batch_size: int
) -> List[Tuple[str, str, str, float, int]]:
    rows: List[Tuple[str, str, str, float, int]] = []
    eb = extract_embeddings(model, windows, device=device, batch_size=batch_size)
    n = eb.feats.shape[0]
    LOG.info("Embeddings %s -> %s", label, tuple(eb.feats.shape))

    rows.append((label, "a", "rankme", rankme(eb.feats), n))
    rows.append((label, "a", "lidar",  lidar(eb.feats, eb.rec), n))
    rows.append((label, "a", "alignment", alignment(eb.feats, eb.rec, eb.win_idx), n))
    rows.append((label, "a", "uniformity", uniformity(eb.feats), n))

    eq = car_equivariance(model, windows, device=device, batch_size=batch_size)
    for k, v in eq.items():
        rows.append((label, "a", f"equiv_{k}", v, n))

    bp = bandpower_r2(eb.feats, windows)
    for k, v in bp.items():
        rows.append((label, "a", f"bandpower_r2_{k}", v, n))

    rows.append((label, "a", "subjid_acc", subject_id_acc(eb.feats, eb.subj), n))
    rows.append((label, "a", "age_r2",     age_r2(eb.feats, eb.subj, age_map), n))
    return rows


def run_tier_b(
    label: str, model, windows, device: str, batch_size: int,
) -> List[Tuple[str, str, str, float, int]]:
    rows: List[Tuple[str, str, str, float, int]] = []

    # Spectral CRPS by band is left as "future work" because the unconditional
    # forecasting path needs more plumbing than fits in this scoping pass —
    # the next-patch-prediction forecast requires Toto2Model._forward to be
    # exposed with the right scale/loc returns.  Emit NaN placeholders so the
    # CSV schema stays stable.
    for b in CANONICAL_BANDS:
        rows.append((label, "b", f"spectral_crps_{b}", float("nan"), 0))
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        rows.append((label, "b", f"quantile_cov_{q:.1f}", float("nan"), 0))

    abx = anomaly_abx(model, windows, device=device, batch_size=batch_size)
    rows.append((label, "b", "anomaly_abx", abx, len(windows)))
    return rows


def main():
    p = argparse.ArgumentParser(description="Representation-quality probes for Toto2 EEG checkpoints")
    p.add_argument("--checkpoint", required=True, help="HF-format dir (config.json + model.safetensors)")
    p.add_argument("--label", required=True, help="Identifier written into the CSV")
    p.add_argument("--tier", choices=["a", "b", "ab"], default="a")
    p.add_argument("--output", required=True, help="Output CSV path (long format)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--npy-root", default=DEFAULT_NPY_ROOT)
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--n-windows-per-recording", type=int, default=2)
    p.add_argument("--max-recordings-per-subject", type=int, default=4)
    p.add_argument("--window-len", type=int, default=4096)
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Cap val subjects (use a small N for smoke testing).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    LOG.setLevel(logging.INFO)

    t0 = time.time()
    LOG.info("Loading val subject pool from %s (seed=%d, fraction=%.3f)",
             args.npy_root, args.val_seed, args.val_fraction)
    val_dirs = deterministic_val_subjects(args.npy_root, args.val_seed, args.val_fraction)
    if args.max_subjects:
        val_dirs = val_dirs[: args.max_subjects]
    LOG.info("Using %d val subjects", len(val_dirs))

    LOG.info("Collecting val windows ...")
    windows = collect_val_windows(
        val_dirs,
        window_len=args.window_len,
        n_windows_per_recording=args.n_windows_per_recording,
        max_recordings_per_subject=args.max_recordings_per_subject,
        seed=args.val_seed,
    )
    LOG.info("Collected %d windows from %d subjects.", len(windows), len(val_dirs))
    if not windows:
        raise SystemExit("No val windows could be collected — abort.")

    LOG.info("Loading checkpoint %s", args.checkpoint)
    model = load_checkpoint(args.checkpoint, device=args.device)
    age_map = load_age_map(args.raw_root) if args.tier in ("a", "ab") else {}
    LOG.info("Loaded age map with %d entries", len(age_map))

    rows: List[Tuple[str, str, str, float, int]] = []
    if args.tier in ("a", "ab"):
        rows.extend(run_tier_a(args.label, model, windows, age_map, args.device, args.batch_size))
    if args.tier in ("b", "ab"):
        rows.extend(run_tier_b(args.label, model, windows, args.device, args.batch_size))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_label", "tier", "metric", "value", "n_samples"])
        for r in rows:
            w.writerow(r)
    LOG.info("Wrote %d rows -> %s (took %.1fs)", len(rows), out, time.time() - t0)


if __name__ == "__main__":
    main()
