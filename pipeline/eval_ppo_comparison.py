"""Compare two already-trained PPO agents on the shared test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline import DataSplit
from models.alpha.features import load_splits
from models.alpha.scoring import load_trained_alpha_model, predict_alpha_wide
from models.rl.config import DEFAULT_ARTIFACT_DIR, DEFAULT_MARKET_ARTIFACT_DIR, EnvConfig
from models.rl.evaluation import compare_metric_sets, ppo_rollout_metrics, reference_portfolio_metrics
from models.rl.features import build_market_features
from models.rl.training import _load_vecnormalize, _make_vec_env
from stable_baselines3 import PPO


def _required_artifact(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PPO artifact: {path}. Train the requested agent first; "
            "expected ppo_agent.zip and vecnormalize.pkl."
        )
    return path


def _training_metadata(artifact_dir: Path, feature_set: str) -> dict:
    """Load optional training metadata without making evaluation depend on it."""
    candidates = [
        artifact_dir / "training_metrics.json",
        Path("eval") / f"ppo_training_metrics_{feature_set}.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return {"source": str(path), "metrics": json.load(handle)}
    return {"source": None, "metrics": None}


def evaluate_agent(artifact_dir: Path, alpha_dir: Path, test_frame, env_config: EnvConfig):
    """Load one PPO artifact and evaluate it deterministically on the test split."""
    model_path = _required_artifact(artifact_dir, "ppo_agent.zip")
    vec_path = _required_artifact(artifact_dir, "vecnormalize.pkl")
    alpha_model = load_trained_alpha_model(alpha_dir)
    alpha = predict_alpha_wide(alpha_model, test_frame)
    features = build_market_features(test_frame)
    # Training uses residual actions by default; evaluation must recreate the
    # same observation and action spaces before loading VecNormalize/PPO.
    env = _make_vec_env(
        features,
        alpha,
        env_config,
        training=False,
        residual=True,
        randomize_start=False,
    )
    env = _load_vecnormalize(env, vec_path)
    model = PPO.load(str(model_path), env=env)
    return ppo_rollout_metrics(model, env)


def run_comparison(
    market_artifact_dir: Path,
    full_artifact_dir: Path,
    market_alpha_dir: Path,
    full_alpha_dir: Path,
    env_config: EnvConfig | None = None,
) -> dict:
    _required_artifact(market_artifact_dir, "ppo_agent.zip")
    _required_artifact(market_artifact_dir, "vecnormalize.pkl")
    _required_artifact(full_artifact_dir, "ppo_agent.zip")
    _required_artifact(full_artifact_dir, "vecnormalize.pkl")
    test_frame = load_splits()[DataSplit.TEST]
    config = env_config or EnvConfig()
    market = evaluate_agent(market_artifact_dir, market_alpha_dir, test_frame, config)
    full = evaluate_agent(full_artifact_dir, full_alpha_dir, test_frame, config)
    result = compare_metric_sets(market, full)
    # PPO starts at environment index 1 and realizes returns from index 2.
    result["baselines"] = reference_portfolio_metrics(
        build_market_features(test_frame),
        start_index=2,
    )
    result["training"] = {
        "market_only": _training_metadata(market_artifact_dir, "market_only"),
        "full_alpha": _training_metadata(full_artifact_dir, "full_alpha"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-artifact-dir", type=Path, default=DEFAULT_MARKET_ARTIFACT_DIR)
    parser.add_argument("--full-artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--market-alpha-dir", type=Path, default=Path("models/alpha/artifacts_market"))
    parser.add_argument("--full-alpha-dir", type=Path, default=Path("models/alpha/artifacts"))
    parser.add_argument(
        "--match-alpha-horizon",
        action="store_true",
        help="Evaluate with the 20-day snapshot environment used by the 20d agents.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    env_config = (
        EnvConfig(rebalance_every=20, rebalance_mode="snapshot", min_holding_days=0)
        if args.match_alpha_horizon
        else EnvConfig()
    )
    result = run_comparison(
        args.market_artifact_dir, args.full_artifact_dir,
        args.market_alpha_dir, args.full_alpha_dir,
        env_config=env_config,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()