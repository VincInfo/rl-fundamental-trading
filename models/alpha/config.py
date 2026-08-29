from dataclasses import dataclass
from typing import Sequence


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

