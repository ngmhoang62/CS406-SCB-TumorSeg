"""Configuration loader and reproducibility utilities."""

from pathlib import Path
from typing import Any, Dict
import random
import numpy as np
import yaml


def load_config(config_path: str | Path = "configs/default_config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random and numpy."""
    random.seed(seed)
    np.random.seed(seed)
