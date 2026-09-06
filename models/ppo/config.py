from dataclasses import dataclass, field
from pathlib import Path

from models.alpha.config import DEFAULT_ARTIFACT_DIR as DEFAULT_ALPHA_ARTIFACT_DIR


DEFAULT_ARTIFACT_DIR = Path("models/ppo/artifacts")
DEFAULT_MARKET_ARTIFACT_DIR = Path("models/ppo/artifacts_market")
DEFAULT_HORIZON_ARTIFACT_DIR = Path("models/ppo/artifacts_20d")
DEFAULT_MARKET_HORIZON_ARTIFACT_DIR = Path("models/ppo/artifacts_market_20d")


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
    # Rule policy: cross-sectional z-score with dead zone (default) or fixed quantiles.
    rule_mode: str = "zscore"  # "zscore" | "quantile"
    rule_z_threshold: float = 0.5
    # Only trade if |alpha| covers this multiple of the transaction-cost rate.
    rule_cost_multiple: float = 1.0
    # Used only when rule_mode == "quantile".
    rule_buy_fraction: float = 0.3
    rule_sell_fraction: float = 0.3
    # Training episodes are random slices of this many steps (None = full panel).
    episode_window: int | None = 80
    randomize_start: bool = True
    # 1 = trade every day; 20 matches the alpha forecast horizon.
    rebalance_every: int = 1
    # If True, SELL can open a short (negative weight) instead of only exiting longs.
    allow_short: bool = False
    # Cap on sum(|weights|). Default: 1.0 long-only, 2.0 long/short (100/100).
    max_gross_exposure: float | None = None
    # "incremental" = PPO-style deltas; "snapshot" = set target book from actions.
    rebalance_mode: str = "incremental"


@dataclass
class PpoConfig:
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 6
    clip_range: float = 0.1
    # Mild entropy bonus; residual policy should mostly stay near "keep rule".
    ent_coef: float = 0.002
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
    # Warm-start by cloning the alpha rule (residual "keep") before PPO fine-tuning.
    use_residual_actions: bool = True
    imitation_episodes: int = 1
    imitation_epochs: int = 40
    imitation_batch_size: int = 256
    # Extra KEEP logit bias after init / BC so the residual prior stays near the rule.
    keep_bias: float = 1.5
    # Behavioral-cloning KEEP loss applied on each PPO rollout (0 disables).
    keep_coef: float = 0.08
    # Observation normalization only; keep PnL and costs in the same units.
    norm_reward: bool = False
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
