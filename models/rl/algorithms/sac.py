from __future__ import annotations

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc

from models.rl.config import TrainingConfig


def build_sac(env: VecNormalize, config: TrainingConfig) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=1,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        learning_starts=config.sac.learning_starts,
        batch_size=config.sac.batch_size,
        tau=config.sac.tau,
        gamma=config.sac.gamma,
        train_freq=config.sac.train_freq,
        gradient_steps=config.sac.gradient_steps,
        ent_coef=config.sac.ent_coef,
    )


def sac_hyperparams(config: TrainingConfig) -> dict:
    return {
        "tau": config.sac.tau,
        "batch_size": config.sac.batch_size,
        "buffer_size": config.sac.buffer_size,
    }
