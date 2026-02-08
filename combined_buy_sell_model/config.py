import os
from dataclasses import dataclass

import yaml


@dataclass
class PipelineConfig:
    ticker: str
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    buy_threshold: float
    sell_threshold: float


CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(config_path: str = CONFIG_FILE) -> PipelineConfig:
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    required_keys = [
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "buy_threshold",
        "sell_threshold",
    ]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise ValueError(f"Missing config keys in {config_path}: {', '.join(missing)}")

    return PipelineConfig(
        ticker=str(raw["ticker"]).upper(),
        train_start_date=str(raw["train_start_date"]),
        train_end_date=str(raw["train_end_date"]),
        test_start_date=str(raw["test_start_date"]),
        test_end_date=str(raw["test_end_date"]),
        buy_threshold=float(raw["buy_threshold"]),
        sell_threshold=float(raw["sell_threshold"]),
    )
