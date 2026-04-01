"""Core PFE components: encoders, losses, settling functions, and bridges."""

from .pfe_encoder import PFEEncoder, LatentPFEEncoder
from .pfe_loss import PFELoss, DSMOnlyLoss, NCEOnlyLoss, vMFInfoNCELoss
from .tsu_settle import tsu_settle, TSUSettler, SettlingMethod
from .memory import MemoryBridge, DLPackError, TensorCache
from .bridge import TSUBridge, ImplicitGradientBridge, create_bridge, GradientMethod
from .tsu_layer import (
    TSULayer,
    EnergyBridge,
    DTMLayer,
    LatentDTM,
    SettlingMode,
    create_tsu_layer,
    create_energy_bridge,
)

__all__ = [
    # Encoders
    "PFEEncoder",
    "LatentPFEEncoder",
    # Losses
    "PFELoss",
    "DSMOnlyLoss",
    "NCEOnlyLoss",
    "vMFInfoNCELoss",
    # Settling
    "tsu_settle",
    "TSUSettler",
    "SettlingMethod",
    # Memory
    "MemoryBridge",
    "DLPackError",
    "TensorCache",
    # Bridge
    "TSUBridge",
    "ImplicitGradientBridge",
    "create_bridge",
    "GradientMethod",
    # Layers
    "TSULayer",
    "EnergyBridge",
    "DTMLayer",
    "LatentDTM",
    "SettlingMode",
    "create_tsu_layer",
    "create_energy_bridge",
]