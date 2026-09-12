# Evaluation

## Ziel

Die Evaluation prüft, ob Fundamentaldaten gegenüber einem identischen `market`-Modell zusätzlichen Nutzen liefern:

- **Alpha:** MSE, gepoolter IC und täglicher Rank-IC
- **Portfolio:** Netto-Rendite, Volatilität, Sharpe Ratio, Drawdown, Turnover und Transaktionskosten
- **Sensitivität:** tägliche Reallokation (`rebalance=1`) gegenüber 20-Tage-Reallokation (`rebalance=20`)

Das Alpha-Target ist der Forward-Return über 20 Handelstage. Das Portfolio wird täglich bewertet; Prognosehorizont und Reallokationsfrequenz sind getrennte Konzepte.

## Vergleich

| Variante | Merkmale |
| --- | --- |
| `market` | OHLCV-basierte Marktmerkmale |
| `full` | Marktmerkmale plus point-in-time Fundamentaldaten |

Beide Varianten benötigen dieselben chronologischen Splits, dasselbe Target und dieselben Kosten-/Positionsregeln. Validation dient der Auswahl; der Test-Split wird erst danach explizit mit `--splits test` ausgeführt. Equal-Weight und Buy-and-Hold sind Referenzen. PPO gehört nicht zum primären Fundamentals-Vergleich.

## Korrektheit

- Kein zufälliges Shuffling und keine Verwendung zukünftiger Fundamentaldaten
- Identische Handelstage, Aktien, Kosten und Limits für alle Vergleiche
- Tägliche Portfolio-Werte werden netto nach Transaktionskosten verbucht
- Ledgerzeilen enthalten Split, Alpha-Variante und Rebalancing-Frequenz
- Testmetriken werden nicht zur Modell- oder Hyperparameterwahl verwendet

## Ausführung

Installiern / synchronisieren der Projektumgebung & ausführen
aller automatisierten Tests: 

```bash
uv sync
uv run pytest -q
```

Trainieren von zwei getrennte XGBoost-Alpha-Modelle mit identischem 20-Tage-Target:
`full` verwendet Markt- und Fundamentaldaten, `market` ausschließlich
Marktmerkmale. Artefakte werden in den angegebenen Verzeichnissen gespeichert.

```bash
uv run python pipeline/train_alpha_model.py --feature-set full --horizon-days 20 --output models/alpha/artifacts
uv run python pipeline/train_alpha_model.py --feature-set market --horizon-days 20 --output models/alpha/artifacts_market
```

Erster Lauf evaluiert auf dem Validation-Split und vergleicht tägliche mit
20-tägiger Reallokation. Dient der Beurteilung und Auswahl von Einstellungen.
Zweiter Lauf verwendet den danach unberührten Test-Split & ist finale
Out-of-Sample-Auswertung. Beide Läufe schreiben Alpha-Metriken, Portfolio-
Metriken und tägliche Ledgerdaten.

```bash
uv run python pipeline/eval_comparison.py --splits validation --rebalance 1,20
uv run python pipeline/eval_comparison.py --splits test --rebalance 1,20
```

Die Ergebnisse stehen in `eval/comparison.json` und `eval/comparison_ledgers.csv`. Das Notebook `notebooks/evaluation_results.ipynb` lädt nur diese Dateien und erstellt Tabellen, Equity-Kurven und Drawdowns. Das ergänzende Notebook `notebooks/fundamental_diagnostics.ipynb` untersucht mögliche Erklärungskanäle wie Signalqualität, Persistenz und Kosten; es ersetzt keine Point-in-Time-Datenprüfung oder statistische Inferenz.

## Einordnung

Ein Fundamentaldaten-Mehrwert ist erst plausibel, wenn bessere Out-of-Sample-Rangordnung und robuste risikoadjustierte Netto-Performance gemeinsam auftreten. Der primäre gespeicherte Testreport muss deshalb immer beide Varianten (`market` und `full`) enthalten. Block-Bootstrap, Diebold-Mariano und echte Konfidenzintervalle sind derzeit nicht implementiert.
