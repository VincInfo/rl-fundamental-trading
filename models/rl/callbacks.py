from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.ppo import rollout_diagnostics
from models.rl.residual import RESIDUAL_KEEP

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc


def init_keep_logit_bias(model: PPO, keep_bias: float = 1.5) -> None:
    """Shift residual KEEP logits so the policy starts near the alpha rule."""
    if keep_bias == 0.0:
        return
    action_net = model.policy.action_net
    bias = getattr(action_net, "bias", None)
    if bias is None:
        return
    n_stocks = int(model.action_space.nvec.shape[0])
    with torch.no_grad():
        bias.view(n_stocks, 3)[:, RESIDUAL_KEEP] += float(keep_bias)


def _sync_obs_rms(eval_env: VecNormalize, train_env: VecNormalize | None) -> None:
    if train_env is None:
        return
    eval_env.obs_rms = train_env.obs_rms


def _apply_keep_prior(model: PPO, keep_coef: float) -> None:
    if keep_coef <= 0.0:
        return
    buffer = getattr(model, "rollout_buffer", None)
    if buffer is None or buffer.observations is None or buffer.observations.size == 0:
        return
    observations = buffer.observations
    flat = np.asarray(observations, dtype=np.float32).reshape(-1, observations.shape[-1])
    n_stocks = int(model.action_space.nvec.shape[0])
    device = model.device
    policy = model.policy
    keep = torch.full((flat.shape[0], n_stocks), RESIDUAL_KEEP, dtype=torch.long, device=device)
    obs_tensor, _ = policy.obs_to_tensor(flat)
    distribution = policy.get_distribution(obs_tensor)
    loss = -distribution.log_prob(keep).mean() * float(keep_coef)
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()


class KeepRegularizedPPO(PPO):
    """PPO that adds a KEEP residual prior after each policy update."""

    def __init__(self, *args, keep_coef: float = 0.08, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_coef = float(keep_coef)

    def train(self) -> None:
        super().train()
        _apply_keep_prior(self, self.keep_coef)


class UnshapedEvalCallback(BaseCallback):
    """Checkpoint on unshaped validation Sharpe, not shaped env reward."""

    def __init__(
        self,
        eval_env: VecNormalize,
        *,
        eval_freq: int,
        patience: int,
        save_path: Path,
        residual: bool = True,
        min_evals: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.patience = max(int(patience), 1)
        self.min_evals = max(int(min_evals), 1)
        self.save_path = Path(save_path)
        self.residual = bool(residual)
        self.best_sharpe = -np.inf
        self.no_improve = 0
        self.n_evals = 0
        self.last_metrics: dict[str, float] = {}

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        train_norm = self.model.get_vec_normalize_env()
        _sync_obs_rms(self.eval_env, train_norm)
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        metrics = rollout_diagnostics(self.model, self.eval_env, residual=self.residual)
        self.last_metrics = metrics
        self.n_evals += 1
        sharpe = float(metrics.get("sharpe_ratio", metrics.get("mean_log_return", 0.0)))
        mean_log = float(metrics.get("mean_log_return", 0.0))
        if self.verbose:
            print(
                f"eval timesteps={self.num_timesteps}  "
                f"sharpe={sharpe:.3f}  mean_log_return={mean_log:.6f}  "
                f"keep={metrics.get('residual_share_keep', float('nan')):.2f}"
            )

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.no_improve = 0
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path / "best_model"))
            if train_norm is not None:
                train_norm.save(str(self.save_path / "best_vecnormalize.pkl"))
            if self.verbose:
                print(f"new best validation Sharpe {sharpe:.3f} -> {self.save_path / 'best_model'}")
        else:
            self.no_improve += 1

        if self.n_evals >= self.min_evals and self.no_improve >= self.patience:
            if self.verbose:
                print(
                    f"Stopping training: no Sharpe improvement in the last "
                    f"{self.no_improve} evaluations"
                )
            return False
        return True
