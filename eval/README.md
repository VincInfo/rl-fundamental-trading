# Evaluation

## Zentrale Forschungsfrage

Die zentrale Evaluation untersucht, welchen zusätzlichen Nutzen
Fundamentaldaten in einem hybriden System aus Alpha-Prognose und
Reinforcement Learning liefern. Dazu werden zwei separat trainierte Systeme
verglichen:

| Agent | Alpha-Eingabe |
| --- | --- |
| `market_only` | Alpha-Scores aus marktbezogenen Merkmalen |
| `full_alpha` | Alpha-Scores aus Markt- und zeitpunktbezogenen Fundamentaldaten |

Die Fundamentaldaten werden dem PPO-Agenten nicht direkt als eigene
Zustandsvariablen übergeben. Ihr Einfluss erfolgt über die Alpha-Scores des
vorgelagerten XGBoost-Modells. Der Vergleich misst daher den Nutzen eines
fundamental erweiterten Alpha-Signals innerhalb des hybriden Systems.

Das zentrale Ergebnis ist `eval/ppo_comparison.json`, erzeugt durch
`pipeline/eval_ppo_comparison.py`. Es enthält deterministische Metriken aus
den Test-Rollouts beider Agenten sowie die Differenzen
`full_alpha - market_only`. Zusätzlich enthält der Report unter `baselines`
die Referenzen `buy_and_hold` und `equal_weight` sowie unter `training`
verfügbare Trainingsmetriken und deren Quellen.

Beide PPO-Agenten verwenden denselben chronologischen Testsplitt, dieselbe
Environment-Konfiguration, dieselben Transaktionskosten und Positionslimits
sowie dasselbe deterministische Evaluationsprotokoll. Die PPO-Rollouts und
Referenzportfolios verwenden dasselbe Return-Fenster mit 91 Perioden.
Testergebnisse dürfen nicht zur Auswahl von Modellen oder Hyperparametern
verwendet werden.

## PPO-Evaluation

Trainieren der beiden Agenten mit demselben Seed und denselben Einstellungen.
Nur das Alpha-Modell und damit die Alpha-Eingabe unterscheiden sich:

```bash
uv run python pipeline/train_ppo.py \
	--no-fundamentals \
	--output models/rl/artifacts_market

uv run python pipeline/train_ppo.py \
	--output models/rl/artifacts
```

Nachdem für beide Agenten die Artefakte `ppo_agent.zip` und
`vecnormalize.pkl` vorhanden sind, führe Folgendes aus:

```bash
uv run python pipeline/eval_ppo_comparison.py \
	--output eval/ppo_comparison.json
```

Der PPO-Report enthält kumulative Rendite, annualisierte Volatilität,
Sharpe-Ratio, maximalen Drawdown, durchschnittlichen Turnover,
Transaktionskosten, durchschnittlichen Reward, die Anzahl der Schritte sowie
finale und Residual-Aktionsanteile. Das Notebook
`notebooks/ppo_evaluation_results.ipynb` visualisiert diesen Report. Es
trainiert keine Agenten und berechnet die Evaluation nicht neu.

Die PPO-Policy verwendet standardmäßig Residualaktionen pro Aktie:

```text
0 = Down, 1 = Keep, 2 = Up
```

`Keep` übernimmt die aktuelle Alpha-Regel. Erst die Kombination aus
Residualaktion und Alpha-Regel ergibt die finale Handelsaktion `Sell`, `Hold`
oder `Buy`. Deshalb bedeutet ein Residual-Anteil von 100 Prozent `Keep` nicht,
dass 100 Prozent finale `Hold`-Aktionen ausgeführt wurden.

Buy-and-Hold startet mit einer gleichgewichteten Position und rebalanced danach
nicht. Equal-Weight rebalanced täglich. Beide Referenzen verwenden dasselbe
Return-Fenster wie PPO, werden aber nicht mit den PPO-Environmentkosten
ausgewertet und starten nicht mit demselben Cash-Zustand. Sie dienen daher der
wirtschaftlichen Einordnung, nicht als vollständig kontrollierte
algorithmische Vergleichspartner.

## Ergänzende Alpha-Diagnostik

Die folgenden Dateien sind ergänzende Analysen und nicht das primäre
RL-Ergebnis:

- `eval/comparison.py`: direkter Vergleich von Alpha-Ranking und Regelportfolios
- `eval/alpha.py`: Metriken für Alpha-Prognosen und Bootstrap-Hilfsfunktion
- `eval/rule_baselines.py`: Kompatibilitätshilfsfunktionen für Regel-Baselines
- `eval/comparison.json`: gespeicherter Alpha-/Regelportfolio-Report
- `eval/comparison_ledgers.csv`: Alpha-/Regelportfolio-Ledger
- `notebooks/evaluation_results.ipynb`: frühere Ergebnisse zu Alpha und Regelportfolios
- `notebooks/fundamental_diagnostics.ipynb`: ergänzende Alpha-Diagnostik

Diese Diagnostik hilft dabei, Signalqualität, Kosten und die Auswirkungen
der Features zu erklären. Sie darf nicht als Ergebnis des PPO-Agenten
bezeichnet werden. `no_levels` ist eine optionale Alpha-Feature-Ablation und
für den primären Zwei-Agenten-PPO-Vergleich nicht erforderlich.

Der direkte Alpha-Regelvergleich und die historischen Regel-Baselines sind
ergänzende Analysen. Der primäre Report vergleicht ausschließlich die beiden
PPO-Systeme und die beiden Referenzportfolios.

## Daten und methodische Korrektheit

- Die Splits sind chronologisch und werden nicht zufällig gemischt
- Fundamentaldaten werden erst ab ihrer Verfügbarkeit verwendet
- Beide PPO-Agenten verwenden dieselben Testbeobachtungen und
	Umgebungsbedingungen
- PPO und Referenzen werden auf demselben Return-Fenster mit 91 Perioden
	berechnet
- Portfolio-Werte, Turnover und Transaktionskosten der PPO-Agenten stammen aus
	dem Environment-Rollout
- Die Referenzen verwenden andere Start- und Ausführungsbedingungen und keine
	PPO-Transaktionskostenberechnung
- Die berichteten PPO-Differenzen sind deskriptiv und belegen keine
	statistische Überlegenheit

Die Evaluation basiert auf einem deterministischen Test-Rollout pro Agent.
Mehrere Seeds, zusätzliche Testzeiträume und statistische Konfidenzintervalle
sind nicht Bestandteil des aktuellen Reports.

Ausführen der vollständigen Testsuite mit folgendem Befehl:

```bash
uv sync
uv run pytest -q
```
