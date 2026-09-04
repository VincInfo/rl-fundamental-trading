from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_ARTIFACT_DIR = Path("models/alpha/artifacts")
DEFAULT_MARKET_ARTIFACT_DIR = Path("models/alpha/artifacts_market")
TARGET_COLUMN = "target_return"

# Shared VANILLA market baseline: derived from OHLCV (not pipeline indicators).
ENGINEERED_FEATURE_COLUMNS = (
    # Close
    "return_1d",
    "return_5d",
    "momentum_20d",
    "vol_20d",
    # Open / High / Low
    "oc_return",
    "hl_range",
    # Volume
    "volume_change_1d",
    "rel_volume_20d",
)

VANILLA_PRICE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)

# Cross-sectionally comparable fundamental ratios (used as levels).
FUNDAMENTAL_LEVEL_COLUMNS = (
    "roe",
    "gross_margin",
    "debt_to_equity",
)

# Accounting flows: too scale-dependent as raw levels → only deltas (+ ranks of deltas).
FUNDAMENTAL_FLOW_COLUMNS = (
    "revenue",
    "net_income",
    "operating_cashflow",
)

CONTEXT_FEATURE_COLUMNS = (
    "filing_lag_days",
)

# Emphasise the post-filing window where fundamental news is most actionable.
EVENT_FEATURE_COLUMNS = (
    "filing_recency",
    "post_filing_5d",
)

CHANGE_FEATURE_COLUMNS = tuple(
    f"delta_{column}"
    for column in FUNDAMENTAL_LEVEL_COLUMNS + FUNDAMENTAL_FLOW_COLUMNS
)

RANK_FEATURE_COLUMNS = tuple(
    f"rank_{column}" for column in FUNDAMENTAL_LEVEL_COLUMNS
) + tuple(
    f"rank_delta_{column}" for column in FUNDAMENTAL_FLOW_COLUMNS
)

# Columns copied from the pipeline wide frame (levels + flows + recency).
FUNDAMENTAL_FEATURE_COLUMNS = (
    FUNDAMENTAL_LEVEL_COLUMNS + FUNDAMENTAL_FLOW_COLUMNS + CONTEXT_FEATURE_COLUMNS
)

# All model inputs when fundamentals are enabled (target_column is the label, not a feature).
MODEL_FEATURE_COLUMNS = (
    ENGINEERED_FEATURE_COLUMNS
    + FUNDAMENTAL_LEVEL_COLUMNS
    + CONTEXT_FEATURE_COLUMNS
    + EVENT_FEATURE_COLUMNS
    + CHANGE_FEATURE_COLUMNS
    + RANK_FEATURE_COLUMNS
)

MARKET_FEATURE_COLUMNS = ENGINEERED_FEATURE_COLUMNS


def feature_columns_for(include_fundamentals: bool) -> tuple[str, ...]:
    """Return Alpha model inputs for the with/without-fundamentals ablation."""
    if include_fundamentals:
        return MODEL_FEATURE_COLUMNS
    return MARKET_FEATURE_COLUMNS


@dataclass
class TrainingConfig:
    # Longer horizon fits slow-moving fundamentals better than 5-day noise.
    horizon_trading_days: int = 20
    bars_per_trading_day: int = 7
    target_column: str = TARGET_COLUMN
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    sample_daily: bool = True
    # Primary ablation switch: same data, market features only vs + fundamentals.
    include_fundamentals: bool = True
    # Predict out/underperformance vs cross-sectional mean (market-neutral target).
    active_return: bool = True

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return feature_columns_for(self.include_fundamentals)

    @property
    def feature_set_name(self) -> str:
        return "with_fundamentals" if self.include_fundamentals else "market_only"


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


def default_xgboost_config(
    feature_names: Sequence[str],
    *,
    include_fundamentals: bool,
) -> XGBoostConfig:
    """Stronger regularisation when fundamentals add sparse/noisy columns."""
    if include_fundamentals:
        return XGBoostConfig(
            feature_names=feature_names,
            max_depth=4,
            min_child_weight=10.0,
            reg_alpha=0.5,
            reg_lambda=2.0,
            colsample_bytree=0.7,
            subsample=0.8,
        )
    return XGBoostConfig(feature_names=feature_names)
