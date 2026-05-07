# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Top-level training script for Toto 2.0.

Instantiates a fresh ``Toto2Model`` from the ``model:`` block in the YAML
config and trains it end-to-end. Continue-pretraining from a published
checkpoint is no longer supported (see lightning_module.py header); use
``Toto2Model.from_pretrained`` for zero-shot inference / evaluation only.

The script is intentionally minimal: dataset construction is delegated to a
user-provided ``--dataset-builder`` callable so this file does not need to
know about EEG-specific I/O. A reference builder is shipped in
``examples/eeg_builder.py``.

Usage
-----

```bash
python -m toto2.scripts.train_toto2 \
    --config toto2/scripts/configs/pretrain_eeg_from_scratch_v3.yaml \
    --dataset-builder my_project.eeg:build_datasets
```

The dataset builder must be importable as ``module:callable`` and have the
signature::

    def build_datasets(config: dict) -> tuple[Dataset, Optional[Dataset]]: ...

It receives the entire parsed config so it can read ``data:`` parameters.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Dataset

import dd_unit_scaling as uu

from toto2.training import (
    SlidingWindowConfig,
    TimeSeriesDataModule,
    Toto2ForTraining,
)


DatasetBuilder = Callable[[Dict[str, Any]], Tuple[Dataset, Optional[Dataset]]]


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping; got {type(cfg).__name__}.")
    return cfg


def load_dotenv(path: str | os.PathLike) -> int:
    """Load ``KEY=VALUE`` lines from a `.env`-style file into ``os.environ``.

    No external dependency. Lines starting with ``#`` and blank lines are
    skipped. Existing environment variables are not overwritten so a value
    explicitly exported in the shell wins over the file.
    """
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


def _flatten_for_wandb(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict into a single-level mapping for wandb.

    Lists / scalars pass through; nested dicts become ``parent.child`` keys.
    """
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_wandb(v, prefix=f"{kk}."))
        else:
            out[kk] = v
    return out


def import_builder(spec: str) -> DatasetBuilder:
    """Resolve ``module.path:callable`` into a callable."""
    if ":" not in spec:
        raise ValueError(
            f"Dataset builder spec must be of the form 'module:callable', got {spec!r}."
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    builder = getattr(module, attr)
    if not callable(builder):
        raise TypeError(f"{spec!r} is not callable.")
    return builder


# ----------------------------------------------------------------------
# Lightning module construction
# ----------------------------------------------------------------------


def _model_kwargs_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Toto2Model hyperparameters from the ``model`` block."""
    model_cfg = dict(cfg.get("model", {}))
    # ``residual_attn_ratio`` is allowed to be omitted; the Lightning module
    # will derive it from ``context_length`` and ``patch_size``.
    return model_cfg


def _trainer_kwargs_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Lightning Trainer kwargs from the ``trainer`` block."""
    tcfg = dict(cfg.get("trainer", {}))
    return tcfg


def build_lightning_module(cfg: Dict[str, Any]) -> Toto2ForTraining:
    train_cfg = cfg.get("training", {})
    context_length = int(cfg.get("data", {}).get("context_length", 4096))

    # exp27 supervision-augmentation suite: optional ``training.auxiliary``
    # block selecting which probes (jepa / aamp / pars / phase / mrstft /
    # denoise) are active and their hyperparameters.  See
    # ``toto2.training.lightning_module._DEFAULT_AUX`` for shape.
    auxiliary = train_cfg.get("auxiliary")

    # exp43 AMSE / exp45 Whittle: optional ``training.loss_type`` switch +
    # ``training.amse`` / ``training.whittle`` blocks.
    loss_type = train_cfg.get("loss_type", "pinball")
    amse_cfg = train_cfg.get("amse")
    whittle_cfg = train_cfg.get("whittle")

    common_kwargs = dict(
        context_length=context_length,
        base_lr=float(train_cfg.get("base_lr", 5e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        stable_steps=int(train_cfg.get("stable_steps", 50_000)),
        decay_steps=int(train_cfg.get("decay_steps", 5_000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),  # type: ignore[arg-type]
        optimizer_name=train_cfg.get("optimizer", "adamw"),
        huber_kappa=float(train_cfg.get("huber_kappa", 0.0)),
        log_grad_norm=bool(train_cfg.get("log_grad_norm", True)),
        auxiliary=auxiliary,
        loss_type=loss_type,
        amse=amse_cfg,
        whittle=whittle_cfg,
    )

    model_kwargs = _model_kwargs_from_config(cfg)
    if not model_kwargs:
        raise ValueError("`model:` block is required to instantiate Toto2ForTraining.")
    return Toto2ForTraining(config=model_kwargs, **common_kwargs)


# ----------------------------------------------------------------------
# Trainer construction
# ----------------------------------------------------------------------


def build_trainer(cfg: Dict[str, Any]) -> L.Trainer:
    tcfg = _trainer_kwargs_from_config(cfg)
    lcfg = cfg.get("logging", {})
    cckpt = cfg.get("checkpoint", {})

    callbacks: list[Any] = [
        TQDMProgressBar(refresh_rate=int(tcfg.get("refresh_rate", 1))),
    ]
    # LearningRateMonitor requires at least one logger; only attach it when
    # we'll actually have one (TensorBoard / CSV / WandB).
    has_logger = any(lcfg.get(k, default) for k, default in
                     [("tensorboard", True), ("csv", True), ("wandb", False)])
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if "dirpath" in cckpt:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(cckpt.get("dirpath")),
                filename=str(cckpt.get("filename", "{epoch}-{step}-{val_loss:.4f}")),
                monitor=cckpt.get("monitor", "val_loss"),
                mode=cckpt.get("mode", "min"),
                save_top_k=int(cckpt.get("save_top_k", 1)),
                every_n_train_steps=cckpt.get("every_n_train_steps"),
                every_n_epochs=cckpt.get("every_n_epochs"),
                save_last=bool(cckpt.get("save_last", True)),
            )
        )

    loggers: list[Any] = []
    if lcfg.get("tensorboard", True):
        loggers.append(
            TensorBoardLogger(
                save_dir=str(lcfg.get("save_dir", "lightning_logs")),
                name=str(lcfg.get("name", "toto2_training")),
            )
        )
    if lcfg.get("csv", True):
        loggers.append(
            CSVLogger(
                save_dir=str(lcfg.get("save_dir", "lightning_logs")),
                name=str(lcfg.get("name", "toto2_training")),
            )
        )
    if lcfg.get("wandb", False):
        # Import lazily so wandb is only required when actually enabled.
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "logging.wandb=true requires `wandb` and lightning's WandbLogger. "
                "Install with `pip install wandb`."
            ) from e

        wandb_kwargs = {
            "project": lcfg.get("wandb_project") or os.environ.get("WANDB_PROJECT", "toto2-eeg"),
            "entity": lcfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY"),
            "name": lcfg.get("wandb_run_name"),
            "tags": list(lcfg.get("wandb_tags", []) or []) or None,
            "save_dir": str(lcfg.get("save_dir", "lightning_logs")),
            "log_model": bool(lcfg.get("wandb_log_model", False)),
            "config": _flatten_for_wandb(cfg),
        }
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        loggers.append(WandbLogger(**wandb_kwargs))

    trainer_kwargs: Dict[str, Any] = dict(
        max_steps=int(tcfg.get("max_steps", 100_000)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=tcfg.get("gradient_clip_val"),
        precision=tcfg.get("precision", "bf16-mixed"),
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", "auto"),
        strategy=tcfg.get("strategy", "auto"),
        val_check_interval=tcfg.get("val_check_interval"),
        check_val_every_n_epoch=tcfg.get("check_val_every_n_epoch"),
        callbacks=callbacks,
        logger=loggers if loggers else False,
        deterministic=bool(tcfg.get("deterministic", False)),
    )
    # Drop None values so Lightning uses its own defaults.
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}
    return L.Trainer(**trainer_kwargs)


# ----------------------------------------------------------------------
# u-μP world-size caching
# ----------------------------------------------------------------------


def configure_unit_scaling_for_world(cfg: Dict[str, Any], trainer: L.Trainer) -> None:
    """Cache the global batch size assumptions before any ``torch.compile``.

    We approximate the data-parallel degree from Lightning's launched world
    size (``num_nodes * devices``). For TP/PP setups this should be overridden
    in the config via ``training.world_size_override``.
    """
    train_cfg = cfg.get("training", {})
    override = train_cfg.get("world_size_override")
    if override is not None:
        world_size = int(override)
    else:
        world_size = int(trainer.world_size)
    accum = int(cfg.get("trainer", {}).get("accumulate_grad_batches", 1))
    try:
        uu.init_world_size_cache(world_size=max(1, world_size))
        uu.set_grad_accumulation_steps(max(1, accum))
    except Exception:
        # If unit_scaling isn't fully initialized (e.g., model has no μP
        # params), this is a no-op rather than a hard failure.
        pass


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-train Toto 2.0 from scratch.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset-builder",
        required=True,
        help="Importable as 'module.path:callable'; receives the parsed config dict.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the config.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional Lightning checkpoint to resume from (.ckpt path).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional .env file to load secrets / settings from. Auto-detected "
             "as ./.env (cwd) and the repo root .env if not specified.",
    )
    args = parser.parse_args(argv)

    # Load secrets from .env (cwd, then repo root). Existing env vars win.
    candidate_env_files: list[Path] = []
    if args.env_file is not None:
        candidate_env_files.append(Path(args.env_file))
    candidate_env_files.append(Path.cwd() / ".env")
    # Best-effort: walk up from this script to find a repo-level .env
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate_env_files.append(parent / ".env")
        if (parent / ".git").exists():
            break
    for env_path in candidate_env_files:
        n_loaded = load_dotenv(env_path)
        if n_loaded > 0:
            print(f"[train_toto2] loaded {n_loaded} key(s) from {env_path}")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = int(args.seed)
    seed = int(cfg.get("training", {}).get("seed", 42))
    L.seed_everything(seed, workers=True)

    # Allow TF32-class matmuls everywhere torch sees an opportunity.
    # On H100 with bf16-mixed AMP this is essentially free and avoids the
    # "tensor cores are not being used to their full potential" warning.
    matmul_precision = cfg.get("training", {}).get("matmul_precision", "high")
    torch.set_float32_matmul_precision(str(matmul_precision))

    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    # Build datasets via user-supplied builder
    builder = import_builder(args.dataset_builder)
    train_ds, val_ds = builder(cfg)

    # ---- Lightning components ----
    lightning_module = build_lightning_module(cfg)
    data_cfg = cfg.get("data", {})
    data_module = TimeSeriesDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_batch_size=int(data_cfg.get("train_batch_size", 16)),
        val_batch_size=int(data_cfg.get("val_batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        drop_last=bool(data_cfg.get("drop_last", True)),
        multiprocessing_context=data_cfg.get("multiprocessing_context", "fork"),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )
    trainer = build_trainer(cfg)

    # u-μP world-size caching must happen after trainer init (ranks set up)
    # but before fit() so that any ``torch.compile`` inside the model picks
    # up the cached ints rather than tracing process-group calls.
    configure_unit_scaling_for_world(cfg, trainer)

    # Optional torch.compile pass — set ``training.compile: true`` to enable.
    if bool(cfg.get("training", {}).get("compile", False)):
        compile_mode = cfg["training"].get("compile_mode", "default")
        lightning_module.model = torch.compile(lightning_module.model, mode=compile_mode)  # type: ignore[assignment]

    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=args.resume_from,
    )

    # Persist the final HF-format model so it can be reloaded by Toto2Model.from_pretrained.
    output_dir = cfg.get("output_dir")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _save_toto2_for_inference(lightning_module, out)
        print(f"Saved trained Toto 2.0 model to {out}")

    return 0


def _save_toto2_for_inference(lightning_module: Toto2ForTraining, out: Path) -> None:
    """Dump ``config.json`` + ``model.safetensors`` matching ``Toto2Model.from_pretrained``.

    The shipped ``Toto2Model._from_pretrained`` reads a JSON config whose
    keys match the ``Toto2ModelConfig`` dataclass fields. We replicate that
    layout exactly so the trained model is loadable with the public API:

    .. code-block:: python

        Toto2Model.from_pretrained("/path/to/output_dir")
    """
    import dataclasses

    from safetensors.torch import save_file

    cfg = lightning_module.model.config
    cfg_dict = dataclasses.asdict(cfg)
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2, sort_keys=True))

    state_dict = {k: v.detach().cpu().contiguous() for k, v in lightning_module.model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"))


if __name__ == "__main__":
    raise SystemExit(main())
