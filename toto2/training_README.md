# Toto 2.0 Training

This directory adds **from-scratch pre-training** support to Toto 2.0. The
shipped Toto 2.0 package was designed for inference only; this submodule
fills in the missing training loop, loss, optimizer, and data plumbing so
you can pre-train on a new domain such as EEG.

> Continue-pretraining from a published Datadog/Toto-2.0-* checkpoint is
> **not supported**. The published checkpoints converged under a different
> data distribution and recipe; in our HBN-EEG experiments we found that
> resuming with a cold AdamW state on EEG reliably destroyed the
> pretrained variance balance within a few hundred steps. Use
> `Toto2Model.from_pretrained` for zero-shot inference / evaluation only.

## What's new

```
toto2/
├── toto2/
│   └── training/                 ← new
│       ├── losses.py             # quantile/pinball loss
│       ├── scheduler.py          # WSD LR schedule
│       ├── lightning_module.py   # Toto2ForTraining
│       ├── datasets.py           # ArrayTimeSeriesDataset / HFTimeSeriesDataset
│       ├── collate.py            # variate-padding collation
│       └── datamodule.py         # TimeSeriesDataModule
├── scripts/                      ← new
│   ├── train_toto2.py            # CLI entry point
│   ├── configs/
│   │   └── pretrain_eeg_from_scratch_v3.yaml
│   └── examples/
│       └── eeg_builder.py        # reference EEG dataset builder
└── tests/
    └── test_training_smoke.py    # CPU smoke tests
```

## Why a separate training module?

Toto 2.0's `Toto2Model` outputs a fixed set of nine quantile knots
(`[0.1, …, 0.9]`) instead of Toto 1.0's Student-T mixture, so its training
recipe is materially different:

| Stage | Toto 1.0 | Toto 2.0 |
|---|---|---|
| Output | Student-T mixture density | 9 fixed quantile knots |
| Loss | NLL + Barron robust (composite) | Mean pinball loss across knots |
| Parameterization | Standard | u-μP (unit-scaled MUP) |
| Optimizer | AdamW (PyTorch) | u-μP-aware `dd_unit_scaling.AdamW` (or NorMuon / Dion2) |
| Pre-norm | RMSNorm + standard residual | RMSNorm + τ-rule residual scaling |

The key practical implications:

1. **Quantile head ⇒ pinball loss.** The mean weighted quantile loss is
   exactly the discrete CRPS approximation used in BOOM/GIFT-Eval, so
   training and evaluation share an objective family.
2. **u-μP ⇒ μP-aware optimizers.** Per-parameter learning-rate scaling is
   handled by `dd_unit_scaling.AdamW`, which reads the `mup_type` /
   `fan_in` metadata that `Toto2Model`'s `uu.Linear` modules attach.
3. **Asinh transform.** Toto 2.0 applies `asinh()` after the causal patch
   scaler. We compute the loss in *scaled + asinh* space (the same space
   the head predicts in) to avoid double-mapping during training.

## Installation

```bash
# Editable install of the toto-2 package (already in pyproject.toml):
cd toto2
pip install -e .

# dd-unit-scaling brings the u-μP modules + optimizers:
pip install "dd-unit-scaling @ git+https://github.com/DataDog/toto.git#subdirectory=dd_unit_scaling"

# Optional Muon-family optimizers:
pip install git+https://github.com/microsoft/dion.git

# Lightning + standard deps if not already installed:
pip install "lightning>=2.2" pyyaml
```

## From-scratch pre-training

```bash
python -m toto2.scripts.train_toto2 \
    --config toto2/scripts/configs/pretrain_eeg_from_scratch_v3.yaml \
    --dataset-builder toto2.scripts.examples.eeg_builder:build_datasets
```

The `model:` block of the YAML is forwarded to `Toto2ModelConfig`. Set
`residual_attn_ratio: null` and the script computes `sqrt(S / log S)`
automatically from `data.context_length / model.patch_size` (Toto-2's
default). Most EEG runs should override this explicitly to a u-µP-balanced
value such as `1.0` — see the `pretrain_eeg_from_scratch_v3.yaml` header
for the full v3 recipe rationale.

## Data format

Each batch element is a tuple `(target, target_mask, series_ids)` with the
shapes Toto 2.0 expects:

- `target`: `(n_var, context_length + patch_size)` — the input window
  *and* the one-patch-shifted target packed together. The Lightning module
  splits them internally.
- `target_mask`: same shape; `True` where the value is observed,
  `False` where it is missing/padded.
- `series_ids`: `(n_var,)` — group id per variate. Variates with the same
  id can attend to each other through the variate-axis attention; the
  sentinel `-1` marks padding variates that never contribute to loss or
  attention.

The collate function `collate_timeseries` pads `n_var` to the batch max,
fills padding slots with `series_id=-1`, and zeros their masks.

### Plug in your own data

Implement a callable

```python
def build_datasets(config: dict) -> tuple[Dataset, Optional[Dataset]]:
    ...
```

and pass `--dataset-builder my_pkg.module:build_datasets`. Reference
implementation in `scripts/examples/eeg_builder.py` shows how to:

- glob `.npz` recordings under `data.eeg.data_root`
- enforce a consistent montage (`expected_channels`)
- apply per-channel z-score / robust z-score normalization
- emit sliding windows of `context_length + patch_size`

For HuggingFace datasets, use `HFTimeSeriesDataset` directly; for any
other format (EDF, BDF, FIF, parquet, S3 streams, …) drop in your own
`Dataset` subclass that returns the three-key dict above.

## Training recipe details

### Supervision (next-patch quantile prediction)

For every input window of length `T = context_length`:

```
input  = series[ 0 : T ]                  # T = K * patch_size
target = series[ patch_size : T + patch_size ]  # shifted by 1 patch
```

The model produces quantile predictions of shape
`(Q, B, V, K, patch_size)` — at patch token `i`, the model predicts the
next patch (`i+1`).

We supervise the predictions against `asinh((target_patch - loc) / scale)`,
where `(loc, scale)` come from the model's own `PatchedCausalStdScaler` —
the same statistics the model used to scale its inputs. This is what
`Toto2Model.forecast()` un-applies at inference time
(`block_q.sinh() * scale + loc`).

### Loss

```
L(θ) = mean over (B, V, K, P) of  E_q [ pinball(asinh_target, ŷ_q, q) ]
```

with optional Huber-smoothing of width `huber_kappa` to stabilize
gradients near zero error (recommended for noisy domains like EEG).

### Optimizer

By default we use `dd_unit_scaling.AdamW`, which:

- Caches per-parameter `mup_type` and `fan_in` *before* FSDP wrapping
  (handled by `Toto2ForTraining.setup`).
- Applies the u-μP per-parameter LR scaling
  (`bias/norm/output → 1.0`, `weight → 1/√fan_in`, then a depth scale).
- Computes weight decay independently of the LR (`independent_weight_decay=True`).

Switch to `optimizer: normuon` or `dion2` in the YAML to use Muon-family
optimizers (requires `dion`). Both are also wrapped to apply the u-μP
depth scale and rely on spectral-norm adjustment for width transfer.

### Schedule

Three-phase Warmup-Stable-Decay:

```
        ▲
base_lr ┤ ─────╱──────────╮
        │   ╱             ╲
   min_lr ╱                ╲___________
        └─────────────────────────────► step
        warmup   stable    decay   plateau
```

The decay segment uses the `1 - √(progress)` shape from the original Toto
paper / MiniCPM (Hu et al., 2024); the plateau at `min_lr` makes
checkpoint-then-extend resumes safe.

### u-μP world-size caching

Before any `torch.compile` pass, the script calls

```python
uu.init_world_size_cache(world_size = trainer.world_size)
uu.set_grad_accumulation_steps(accumulate_grad_batches)
```

so that batch-dependent unit-scaling factors bake into the compiled graph
as plain ints rather than process-group queries. If you use
TP/PP/SP/EP — which do *not* count toward the data-parallel world size —
override via `training.world_size_override:` in the YAML (the dd-unit-scaling
README has the full table).

## Diagnostics & checkpointing

`Toto2ForTraining` logs:

- `train_loss` / `val_loss` (mean weighted pinball)
- `train_observed_frac` / `val_observed_frac` (fraction of supervised steps)
- `grad_norm` (global L2) before each optimizer step
- `lr-AdamW` (or `lr-NorMuon` etc.) via `LearningRateMonitor`

The script writes:

- TensorBoard / CSV logs under `lightning_logs/<name>/version_N/`
- Top-K and `last.ckpt` Lightning checkpoints under `checkpoint.dirpath`
- A final HuggingFace-format export under `output_dir/` (config.json +
  `model.safetensors`), so the trained model can be reloaded with
  `Toto2Model.from_pretrained(output_dir)` and used by `Toto2GluonTSModel`
  for inference / evaluation without code changes.

## Resuming

Lightning checkpoints can be resumed transparently:

```bash
python -m toto2.scripts.train_toto2 \
    --config toto2/scripts/configs/pretrain_eeg_from_scratch_v3.yaml \
    --dataset-builder toto2.scripts.examples.eeg_builder:build_datasets \
    --resume-from checkpoints/.../last.ckpt
```

This restores model weights, optimizer state, scheduler state, and the
RNG state.

## Smoke tests

CPU-only sanity checks:

```bash
pytest toto2/tests/test_training_smoke.py -v
```

These verify:

1. Pinball loss matches the analytic value at `q=0.5` (`0.5·|e|`).
2. Huber-smoothed pinball is exactly zero at `e=0`.
3. `ArrayTimeSeriesDataset` produces windows of the right shape.
4. `collate_timeseries` pads heterogeneous variate counts and zeros
   padding masks.
5. A two-layer Toto 2.0 with random init runs one training step on a
   tiny synthetic batch and produces a finite loss.

## Caveats / future work

- **`num_output_patches` is required to be 1** in the current training
  module. Multi-patch heads can be added by aligning the target with
  `[..., ::nop]` in `_step`; left as a follow-up because the public 2.0
  checkpoints all use `num_output_patches=1`.
- **Causal patch mask (`cpm_mask`)** is set to `target_mask` everywhere by
  default, matching standard LLM-style next-patch training. If you want
  to train under the same masking conditions as `forecast()` (target
  positions hidden from the scaler), pass an explicit `cpm_mask` to the
  module's `forward`.
- **EEG-specific preprocessing** (band-pass, ICA, artifact rejection)
  is your responsibility — only feed clean recordings to
  `ArrayTimeSeriesDataset`. The reference builder offers per-channel
  standardization but does not filter.
- **Flash-Attention kernels.** The Toto 2.0 model picks the SDPA backend
  automatically; `precision: bf16-mixed` is the recommended training
  precision and works on Ampere+ GPUs.
