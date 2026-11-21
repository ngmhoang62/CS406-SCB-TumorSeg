"""Compatibility entry point for full-volume preprocessing."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import preprocess
from src.utils.config import load_config, seed_everything


def execute_steps_1_to_3(config_path="configs/default_config.yaml"):
    config = load_config(config_path)
    seed_everything(config["project"]["seed"])
    preprocess(argparse.Namespace(rebuild_splits=False, overwrite=False), config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    args = parser.parse_args()
    execute_steps_1_to_3(args.config)
