import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.comparison import evaluate_comparison, save_comparison, save_rule_baselines


def main() -> None:
    results = evaluate_comparison()
    save_comparison(results)
    save_rule_baselines(results)


if __name__ == "__main__":
    main()
