"""Repository-relative configuration and reproducibility."""

from pathlib import Path
import random

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path="configs/default_config.yaml"):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, value in config["paths"].items():
        path = Path(value)
        config["paths"][key] = str(path if path.is_absolute() else PROJECT_ROOT / path)
    return config


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested=None):
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the GPU driver.")
    return device
