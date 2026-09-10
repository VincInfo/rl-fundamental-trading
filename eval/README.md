# Evaluation

Dieses Verzeichnis enthält den Evaluationscode für das hybride Trading-System.
Die Evaluation trennt:

1. **Technische Diagnostik:** Alpha-Metriken auf der Validation und PPO-
   Rollout-Diagnostik auf den Training- und Validation-Splits.
2. **Portfolio-Evaluation:** Netto-Portfolioentwicklung auf dem ungesehenen
   Test-Split einschließlich Transaktionskosten.

Das Notebook [`notebooks/evaluation_results.ipynb`](../notebooks/evaluation_results.ipynb)
lädt die erzeugten Ergebnisse, prüft die Ledger und erstellt Tabellen sowie
Abbildungen. Es trainiert keine Modelle und implementiert die Simulation nicht
erneut.

## Reproduzierbarer Ablauf

Die Befehle werden aus dem Repository-Root ausgeführt:

```bash
uv sync
uv run python pipeline/train_alpha_model.py
uv run python pipeline/train_ppo.py
uv run python -m eval.run_test_evaluation
```

Die Auswertung mit Fundamentaldaten verwendet standardmäßig
`DataVariant.WITH_FUNDAMENTALS` und schreibt in die Standardpfade
`models/alpha/artifacts`, `models/ppo/artifacts` und `eval/results`.
Für einen Vergleich ohne Fundamentaldaten und ohne von der Datenpipeline
bereitgestellte technische Indikatoren wird `DataVariant.VANILLA` verwendet.
Dieser Lauf muss strikt getrennte Modell- und Ergebnisordner verwenden, damit
keine Artefakte des Fundamentals-Laufs überschrieben werden:

```bash
uv run python pipeline/train_alpha_model.py \
  --data-variant VANILLA \
  --output models/alpha/artifacts_vanilla \
  --evaluation-output eval/results/vanilla

uv run python pipeline/train_ppo.py \
  --data-variant VANILLA \
  --alpha-model models/alpha/artifacts_vanilla \
  --output models/ppo/artifacts_vanilla \
  --evaluation-output eval/results/vanilla

uv run python -m eval.run_test_evaluation \
  --data-variant VANILLA \
  --alpha-model models/alpha/artifacts_vanilla \
  --ppo-artifact models/ppo/artifacts_vanilla \
  --output eval/results/vanilla
```

Die beiden Varianten werden anschließend über
`eval/results/test_performance_summary.csv` und
`eval/results/vanilla/test_performance_summary.csv` verglichen. Die
Metadaten speichern zusätzlich die verwendete Datenvariante. Die im Alpha-
Modell aus `Close` berechneten Merkmale wie Momentum und Volatilität bleiben
in beiden Varianten identisch; entfernt werden die Fundamentaldaten und die
von der Datenpipeline gelieferten Indikatorspalten.

Für die Aussage über den Einfluss der Fundamentaldaten werden ausschließlich
die beiden separat trainierten PPO-Läufe miteinander verglichen: der Lauf mit
`WITH_FUNDAMENTALS` und der Lauf mit `VANILLA`. Jede Variante verwendet dabei
ein auf derselben Variante trainiertes Alpha-Modell. Die Baselines werden
innerhalb jeder Variante mit den jeweils passenden Alpha-Scores berechnet.
Ein Vergleich des Vanilla-Agenten mit dem Fundamentals-Alpha-Modell oder ein
Vergleich über Kreuz wäre für diese Fragestellung nicht zulässig.

Die Trainingsschritte können entfallen, wenn die benötigten Modellartefakte
bereits vorhanden sind. Die finale Testauswertung darf erst ausgeführt werden,
wenn Modellvarianten und Hyperparameter anhand der Trainings- und Validation-
Daten festgelegt wurden.

Für die optionale Neutral-Alpha-Ablation wird ein separater PPO-Agent trainiert
und ausgewertet:

```bash
uv run python pipeline/train_ppo.py \
  --neutral-alpha \
  --output models/ppo/artifacts_neutral_alpha \
  --evaluation-output eval/results/neutral_alpha

uv run python -m eval.run_test_evaluation \
  --neutral-alpha \
  --ppo-artifact models/ppo/artifacts_neutral_alpha \
  --output eval/results/neutral_alpha
```

## Struktur der Ergebnisse

Generierte Evaluationsdateien werden unter `eval/results/` gespeichert:

```text
eval/results/
  alpha_training_metrics.json
  ppo_training_metrics.json
  ppo_train_portfolio.csv
  ppo_validation_portfolio.csv
  ppo_test_portfolio.csv
  equal_weight_test_portfolio.csv
  buy_and_hold_test_portfolio.csv
  alpha_ranking_test_portfolio.csv
  test_performance_summary.csv
  test_evaluation_metrics.json
  test_evaluation_metadata.json
  vanilla/                 # separate VANILLA-Auswertung
    alpha_training_metrics.json
    ppo_training_metrics.json
    ppo_test_portfolio.csv
    test_performance_summary.csv
  neutral_alpha/              # optionale Ablation
  notebook_outputs/           # abgeleitete Notebook-Exporte
```

Diese generierten Dateien werden von Git ignoriert. Der Ergebnisordner selbst
bleibt durch `eval/results/.gitkeep` im Repository erhalten.

## Daten und Ausführung

Die Evaluation verwendet die chronologischen Train-, Validation- und Test-
Splits der gemeinsamen Datenpipeline. Der Test-Runner lädt das trainierte
Alpha-Modell und die PPO-Artefakte, führt die Testauswertung einmal aus und
schreibt die Portfolio-Ledger sowie die Zusammenfassung.

Das Test-Panel enthält einen Initialisierungstag. Deshalb beginnt das Ledger
am zweiten Testdatum; die Renditemetriken verwenden die darauffolgenden
Return-Perioden. `test_evaluation_metadata.json` speichert den Roh-Testzeitraum,
den Ledgerzeitraum, die Anzahl der Datenpunkte, die Symbole, die Modellpfade,
den Seed, die Environment-Parameter und die Portfolioannahmen.

Der PPO-Test-Rollout ist deterministisch. Die PPO-Umgebung verwendet die
konfigurierten Transaktionskosten, Long-only-Bedingungen, das maximale
Einzelgewicht, die Mindesthaltedauer, das Rebalancing-Budget und das
volatilitätsbasierte Position Sizing.

Die Baselines verwenden dieselben Daten, dasselbe Startkapital, dieselben
Transaktionskosten und dasselbe maximale Einzelgewicht. Ihre Ausführung ist
bewusst nicht identisch mit PPO:

- **Equal Weight:** tägliche Zielgewichte über alle Aktien.
- **Buy and Hold:** gleichgewichtige Anfangsallokation ohne weiteres
  Rebalancing.
- **Alpha Ranking:** tägliche Zielgewichte für die Aktien mit den höchsten
  Scores. Der ausgewählte Anteil beträgt `0.2`; mindestens `ceil(1 / w_max)`
  Positionen werden ausgewählt, damit das Einzelgewichtslimit eingehalten
  werden kann.

Die Baseline-Ledger werden von `eval.portfolio` erzeugt, die PPO-Ledger von
`eval.ppo`. Alle Strategien werden mit derselben Portfolio-Metrik verglichen.

## Berichtete Metriken

Die Portfolio-Zusammenfassung enthält:

- kumulierte Rendite,
- annualisierte Rendite,
- annualisierte Volatilität,
- Sharpe Ratio mit einem risikofreien Zinssatz von null,
- maximalen Drawdown,
- mittleren Turnover,
- gesamte Transaktionskosten.

Die PPO-Diagnostik enthält zusätzlich den mittleren Reward, die mittlere und
maximale Policy-Entropie sowie die Anteile der Aktionen Buy, Hold und Sell.
Die Alpha-Validation-Metriken enthalten MSE, den globalen IC und die Anzahl der
Validation-Samples.

Die Portfolio-Metriken werden aus dem täglichen Portfolio-Value-Ledger und
nicht aus dem PPO-Reward berechnet. Der Reward ist ein internes Lernsignal und
kann von der realisierten Netto-Portfolio-Rendite abweichen.

## Alpha-Beitrag und Einschränkungen

Der Alpha-Beitrag wird auf zwei Ebenen betrachtet:

1. `alpha_ranking` misst den direkten Portfolio-Mehrwert des Alpha-Signals.
2. Der Neutral-Alpha-PPO-Lauf ersetzt aktienspezifische Alpha-Scores durch
   konstante positive Werte. Der Vergleich mit dem normalen PPO-Lauf misst den
   Einfluss der Alpha-Information innerhalb von PPO. Dies ist eine Ablation und
   keine neue Handelsstrategie; sie erfordert ein separates Training.

Die aktuelle Implementierung hat folgende Einschränkungen:

- standardmäßig wird nur ein PPO-Seed ausgewertet;
- zeitliche Verläufe von Trainings-Reward und Entropie werden nicht gespeichert;
- der Alpha-IC ist eine globale Validation-Korrelation und kein täglicher
  querschnittlicher IC;
- die Ausführungsregeln von Baselines und PPO sind nicht identisch;
- der Commit des Datenproviders und ein Daten-Hash werden nicht in den
  Evaluationsmetadaten gespeichert. Die Providerversion ist jedoch in
  `pyproject.toml` und `uv.lock` festgelegt

Ein einzelner Testlauf bewertet ein trainiertes Modell technisch und wirtschaftlich, ist aber kein Nachweis für Robustheit über mehrere Seeds oder für eine allgemeine Investment-Performance.