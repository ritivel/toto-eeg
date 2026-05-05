# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Top-level training script for Toto 2.0.

Supports two complementary modes selected via ``mode:`` in the YAML config:

- ``from_scratch``: instantiate a fresh ``Toto2Model`` from the
  ``model:`` block in the config and train end-to-end.
- ``continue_pretrain``: load a published Toto 2.0 checkpoint
  (e.g. ``Datadog/Toto-2.0-22m``) and continue pre-training on a
  different distribution (e.g. EEG).

The script is intentionally minimal: dataset construction is delegated to a
user-provided ``--dataset-builder`` callable so this file does not need to
know about EEG-specific I/O. Two reference builders are shipped in
``examples/eeg_builder.py``.

Usage
-----

```bash
# From-scratch pretraining
python -m toto2.scripts.train_toto2 \
    --config toto2/scripts/configs/pretrain_eeg_from_scratch.yaml \
    --dataset-builder my_project.eeg:build_datasets

# Continued pretraining
python -m toto2.scripts.train_toto2 \
    --config toto2/scripts/configs/continue_pretrain_eeg.yaml \
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
    mode = cfg.get("mode", "from_scratch")
    train_cfg = cfg.get("training", {})
    context_length = int(cfg.get("data", {}).get("context_length", 4096))

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
    )

    if mode == "from_scratch":
        model_kwargs = _model_kwargs_from_config(cfg)
        if not model_kwargs:
            raise ValueError("`model:` block is required for mode=from_scratch.")
        module = Toto2ForTraining(config=model_kwargs, **common_kwargs)
    elif mode == "continue_pretrain":
        model_id = cfg.get("pretrained_model_id")
        if model_id is None:
            raise ValueError("`pretrained_model_id` is required for mode=continue_pretrain.")
        module = Toto2ForTraining.from_pretrained(
            model_id=str(model_id),
            map_location=str(cfg.get("map_location", "cpu")),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}; expected 'from_scratch' or 'continue_pretrain'.")

    return module


# ----------------------------------------------------------------------
# Trainer construction
# ----------------------------------------------------------------------


def build_trainer(cfg: Dict[str, Any]) -> L.Trainer:
    tcfg = _trainer_kwargs_from_config(cfg)
    lcfg = cfg.get("logging", {})
    cckpt = cfg.get("checkpoint", {})

    callbacks: list[Any] = [
        TQDMProgressBar(refresh_rate=int(tcfg.get("refresh_rate", 1))),
        LearningRateMonitor(logging_interval="step"),
    ]
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
    parser = argparse.ArgumentParser(description="Train or continue pre-training Toto 2.0.")
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
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = int(args.seed)
    seed = int(cfg.get("training", {}).get("seed", 42))
    L.seed_everything(seed, workers=True)

    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    # Build datasets via user-supplied builder
    builder = import_builder(args.dataset_builder)
    train_ds, val_ds = builder(cfg)

    # ---- Lightning components ----
    lightning_module = build_lightning_module(cfg)
    data_module = TimeSeriesDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_batch_size=int(cfg.get("data", {}).get("train_batch_size", 16)),
        val_batch_size=int(cfg.get("data", {}).get("val_batch_size", 16)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(cfg.get("data", {}).get("pin_memory", True)),
        persistent_workers=bool(cfg.get("data", {}).get("persistent_workers", False)),
        drop_last=bool(cfg.get("data", {}).get("drop_last", True)),
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
