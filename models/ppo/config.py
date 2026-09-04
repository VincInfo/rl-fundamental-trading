from dataclasses import dataclass, field
from pathlib import Path

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR


DEFAULT_ARTIFACT_DIR = Path("models/ppo/artifacts")
DEFAULT_MARKET_ARTIFACT_DIR = Path("models/ppo/artifacts_market")


@dataclass
class EnvConfig:
    initial_cash: float = 1_000_000.0
    transaction_cost_bps: float = 10.0
    # Stronger overtrading brake so PPO follows alpha instead of churning.
    trade_penalty_bps: float = 10.0
    # Small reward bonus when actions agree with the alpha sign (scaled by q).
    alpha_alignment_bps: float = 5.0
    min_holding_days: int = 5
    w_max: float = 0.2
    rebalance_budget: float = 0.2
    vol_window: int = 20
    eps: float = 1e-8
    # Rule-based alpha baseline: buy top / sell bottom quantile each day.
    rule_buy_fraction: float = 0.3
    rule_sell_fraction: float = 0.3


@dataclass
class PpoConfig:
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 6
    clip_range: float = 0.1
    # Mild entropy bonus early; still far below a uniform random policy.
    ent_coef: float = 0.005
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


@dataclass
class TrainingConfig:
    timesteps: int = 200_000
    seed: int = 42
    alpha_model_dir: Path = DEFAULT_ALPHA_ARTIFACT_DIR
    alpha_scores_path: str | None = None
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    # Evaluate validation every N env steps and keep the best checkpoint.
    eval_freq_steps: int = 10_000
    early_stop_patience: int = 8
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
