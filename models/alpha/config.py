from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_ARTIFACT_DIR = Path("models/alpha/artifacts")
TARGET_COLUMN = "target_return"

ENGINEERED_FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "momentum_20d",
    "vol_20d",
)

FUNDAMENTAL_FEATURE_COLUMNS = (
    "roe",
    "gross_margin",
    "debt_to_equity",
)

# All model inputs produced by build_dataset (target_column is the label, not a feature).
MODEL_FEATURE_COLUMNS = ENGINEERED_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS


@dataclass
class TrainingConfig:
    horizon_trading_days: int = 5
    bars_per_trading_day: int = 7
    target_column: str = TARGET_COLUMN
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    sample_daily: bool = True


@dataclass
class XGBoostConfig:
    feature_names: Sequence[str]
    n_estimators: int = 1_000
    max_depth: int = 5
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    min_child_weight: float = 5.0
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 50
    verbose: int | bool = True
