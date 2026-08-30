from dataclasses import dataclass, field
from pathlib import Path

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR


DEFAULT_ARTIFACT_DIR = Path("models/ppo/artifacts")


@dataclass
class EnvConfig:
    initial_cash: float = 1_000_000.0
    transaction_cost_bps: float = 10.0
    trade_penalty_bps: float = 2.0
    min_holding_days: int = 3
    w_max: float = 0.2
    rebalance_budget: float = 0.2
    vol_window: int = 20
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    timesteps: int = 20_000
    seed: int = 42
    alpha_model_dir: Path = DEFAULT_ALPHA_ARTIFACT_DIR
    alpha_scores_path: str | None = None
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    env: EnvConfig = field(default_factory=EnvConfig)
