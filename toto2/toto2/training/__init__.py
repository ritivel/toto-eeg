# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Toto 2.0 training pipeline.

Provides PyTorch Lightning modules, loss functions, schedulers, and data
modules required to pre-train Toto 2.0 from scratch on new domains
(e.g., EEG signals).
"""

from .amse_loss import AMSELoss, amse_loss_1d
from .whittle_loss import WhittleLoss, whittle_loss_1d
from .auxiliary_losses import (
    JEPAHead,
    MultiResolutionSTFTAuxLoss,
    PARSHead,
    amplitude_aware_cpm_mask,
    online_denoise_target,
    phase_aware_fourier_loss,
    sigreg_loss,
)
from .collate import collate_timeseries
from .datamodule import TimeSeriesDataModule
from .datasets import (
    ArrayTimeSeriesDataset,
    HFTimeSeriesDataset,
    SlidingWindowConfig,
)
from .lazy_npz import LazyNpzTimeSeriesDataset
from .losses import QuantileLoss, quantile_loss
from .lightning_module import Toto2ForTraining
from .scheduler import WarmupStableDecayLR

__all__ = [
    "Toto2ForTraining",
    "QuantileLoss",
    "quantile_loss",
    "WarmupStableDecayLR",
    "TimeSeriesDataModule",
    "ArrayTimeSeriesDataset",
    "HFTimeSeriesDataset",
    "LazyNpzTimeSeriesDataset",
    "SlidingWindowConfig",
    "collate_timeseries",
    # exp27 auxiliary supervision suite
    "JEPAHead",
    "MultiResolutionSTFTAuxLoss",
    "PARSHead",
    "amplitude_aware_cpm_mask",
    "online_denoise_target",
    "phase_aware_fourier_loss",
    "sigreg_loss",
    # exp43 AMSE primary loss (Subich et al., ICML 2025)
    "AMSELoss",
    "amse_loss_1d",
    # exp45 Whittle pseudo-likelihood (Whittle 1953)
    "WhittleLoss",
    "whittle_loss_1d",
]
