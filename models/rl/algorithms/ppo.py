from __future__ import annotations

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc

from models.rl.config import TrainingConfig


def build_ppo(env: VecNormalize, config: TrainingConfig) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=1,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        max_grad_norm=config.ppo.max_grad_norm,
    )


def ppo_hyperparams(config: TrainingConfig) -> dict:
    return {
        "clip_range": config.ppo.clip_range,
        "n_epochs": config.ppo.n_epochs,
    }
