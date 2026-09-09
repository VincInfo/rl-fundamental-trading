from __future__ import annotations

import numpy as np

from eval.ppo import make_alpha_rule_policy
from models.rl.residual import RESIDUAL_KEEP, ResidualAlphaEnv
from models.rl.envs import MultiStockTradingEnv

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc


def collect_demonstrations(
    env: MultiStockTradingEnv | ResidualAlphaEnv,
    *,
    n_episodes: int = 3,
    residual: bool = True,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (observation, expert_action) pairs from the alpha rule policy."""
    if residual and not isinstance(env, ResidualAlphaEnv):
        raise TypeError("residual=True requires a ResidualAlphaEnv wrapper.")

    rule_policy = make_alpha_rule_policy(
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(max(int(n_episodes), 1)):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if residual:
                expert = np.full(env.action_space.nvec.shape[0], RESIDUAL_KEEP, dtype=np.int64)
                base_env = env.trading_env
            else:
                base_env = env if isinstance(env, MultiStockTradingEnv) else env.trading_env
                expert = np.asarray(rule_policy(base_env), dtype=np.int64)
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(expert)
            obs, _, terminated, truncated, _ = env.step(expert)

    return np.stack(observations, axis=0), np.stack(actions, axis=0)


def behavioral_clone_policy(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> dict[str, float]:
    """Supervised warm-start of the PPO policy toward expert actions."""
    if len(observations) == 0:
        return {"imitation_loss": float("nan"), "imitation_accuracy": float("nan")}

    device = model.device
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    obs = np.asarray(observations, dtype=np.float32)
    acts = np.asarray(actions, dtype=np.int64)
    n_samples = obs.shape[0]
    batch_size = max(1, min(int(batch_size), n_samples))

    last_loss = 0.0
    last_acc = 0.0
    policy.train()
    for _ in range(max(int(epochs), 1)):
        permutation = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            index = permutation[start : start + batch_size]
            obs_batch = obs[index]
            act_batch = torch.as_tensor(acts[index], device=device)

            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            distribution = policy.get_distribution(obs_tensor)
            log_prob = distribution.log_prob(act_batch)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = distribution.mode()
                last_acc = float((pred == act_batch).float().mean().item())
                last_loss = float(loss.item())

    # Keep SB3 optimizer state consistent with cloned weights for PPO updates.
    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=model.lr_schedule(1.0),
    )
    return {
        "imitation_loss": last_loss,
        "imitation_accuracy": last_acc,
        "imitation_samples": float(n_samples),
    }


def collect_vec_demonstrations(
    vec_env: VecEnv,
    *,
    n_episodes: int = 3,
    residual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect demos through a (possibly normalized) vectorized env."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    n_envs = vec_env.num_envs

    for _ in range(max(int(n_episodes), 1)):
        obs = vec_env.reset()
        dones = np.array([False] * n_envs)
        # DummyVecEnv: one env; finish when that episode ends.
        while not bool(dones[0]):
            if residual:
                expert = np.full(
                    (n_envs, vec_env.action_space.nvec.shape[0]),
                    RESIDUAL_KEEP,
                    dtype=np.int64,
                )
            else:
                raise ValueError("Non-residual vec demos are not used; wrap with ResidualAlphaEnv.")
            observations.append(np.asarray(obs[0], dtype=np.float32))
            actions.append(np.asarray(expert[0], dtype=np.int64))
            obs, _, dones, _ = vec_env.step(expert)

    return np.stack(observations, axis=0), np.stack(actions, axis=0)
