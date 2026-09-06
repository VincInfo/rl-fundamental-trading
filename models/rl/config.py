from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR


DEFAULT_ARTIFACT_DIR = Path("models/rl/artifacts")

Algorithm = Literal["ppo", "sac"]


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
class PpoConfig:
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 4
    clip_range: float = 0.1
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


@dataclass
class SacConfig:
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    ent_coef: str | float = "auto"
    learning_rate: float = 3e-4
    gamma: float = 0.99


@dataclass
class TrainingConfig:
    timesteps: int = 200_000
    seed: int = 42
    algorithm: Algorithm = "ppo"
    alpha_model_dir: Path = DEFAULT_ALPHA_ARTIFACT_DIR
    alpha_scores_path: str | None = None
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    sac: SacConfig = field(default_factory=SacConfig)
