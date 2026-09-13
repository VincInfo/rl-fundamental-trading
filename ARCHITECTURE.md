# Hybrid RL Architecture for Quantitative Trading

## 1. Overview

Das System trennt **Prognose**, **Handelsentscheidung** und **Positionsgröße**:

- Das **Alpha-Modell** (XGBoost) prognostiziert die Attraktivität einer Aktie.
- Das **Risk Adjustment** setzt das Signal ins Verhältnis zur rollierenden Volatilität.
- Die **Alpha-Regel** liefert eine Baseline-Handelsrichtung.
- Das **RL-Environment** kombiniert diese Regel mit Residualaktionen des PPO-Agenten.
- Das **Position Sizing** übersetzt die finale Aktion und die Signalstärke in eine Trade-Größe.
- **Portfolio Constraints** begrenzen die resultierenden Positionen und Trades.

```text
Market + Fundamental Data → Feature Engineering → Alpha Model
→ Alpha Scores → Alpha Rule (z-score) → PPO Residual Policy (Down/Keep/Up)
→ finale Aktion (Sell/Hold/Buy)
→ Position Sizing → Portfolio Constraints → Rebalancing
→ New Portfolio → Log-Return nach Kosten − Trade Penalty + Alignment → Reward
```

Training und Evaluation nutzen standardmäßig die Residual-Umgebung. Das direkte
Environment (absolute Sell/Hold/Buy-Aktionen) bleibt als Ablation verfügbar.

## 2. Data & Features

Das Universum umfasst `N = 20` Aktien:

```text
AAPL, MSFT, NVDA, IBM, CSCO, TXN, AMAT, CRM, MU, LRCX,
TSLA, LOW, TGT, PEP, COST, MDLZ, JNJ, LLY, ISRG, ADP
```

Marktdaten stammen aus OHLCV (Yahoo Finance). Fundamentaldaten kommen aus den
SEC-EDGAR Company Facts (10-Q / 10-K) und werden Point-in-Time auf den
Marktindex projiziert: Zum Zeitpunkt `t` dürfen nur bereits veröffentlichte
Filings eingehen. Features und Labels werden auf der ersten Stundenbar jedes
Kalendertags berechnet (`sample_daily=True`), sodass Alpha-Modell und PPO
denselben Rebalancing-Zeitpunkt verwenden. Die Splits sind chronologisch
(Training / Validation / Test) und werden nicht gemischt.

### Market Features

Aus Open, High, Low, Close und Volume:

- `return_1d`, `return_5d`, `momentum_20d`, `vol_20d`
- `oc_return`, `hl_range`
- `volume_change_1d`, `rel_volume_20d`

### Fundamental Features

Nur im Feature-Set `full` bzw. in der Ablation `no_levels`:

- Levels: `roe`, `gross_margin`, `debt_to_equity`
- Flows (nur als Deltas, nicht als Rohwerte): `revenue`, `net_income`,
  `operating_cashflow`
- Context / Events: `filing_lag_days`, `filing_recency`, `post_filing_5d`
- Changes: `delta_*` der Levels und Flows
- Cross-Section-Ranks: `rank_roe`, `rank_gross_margin`, `rank_debt_to_equity`
  sowie `rank_delta_*` der Flows

P/E, EV/EBITDA oder ROIC werden nicht verwendet. `no_levels` behält Deltas,
Filing-Features und Flow-Ranks, lässt aber sticky Accounting-Levels und deren
Ranks weg.

## 3. Alpha Model

XGBoost ist das implementierte Alpha-Modell. LightGBM ist nicht vorhanden.
Es gibt drei Feature-Sets:

1. **`market`**: nur Market Features
2. **`full`**: Market- plus Fundamental-Features (Default)
3. **`no_levels`**: Fundamentals ohne sticky Levels; ergänzende Ablation

### Input & Output

Das Alpha-Modell wird für jede Aktie `i` und jeden Entscheidungszeitpunkt `t`
angewendet. Sein Input ist ein Feature-Vektor:

```math
\mathbf{x}_{i,t}
=
[
\text{Market Features}_{i,t},
\text{Fundamental Features}_{i,t}
]
```

Im Feature-Set `market` entfallen die Fundamental Features. Alle Inputs müssen
**Point-in-Time korrekt** sein.

Der Prognosehorizont ist `H = 20` Handelstage. Das Trainingsziel ist die
**aktive** Forward-Rendite relativ zum gleichgewichteten Cross-Section-Mittel
desselben Tages, nicht die absolute Rendite und nicht ein externer Index:

```math
y_{i,t}
=
r_{i,t\rightarrow t+H}
-
\frac{1}{N}\sum_{j=1}^{N} r_{j,t\rightarrow t+H}
```

mit

```math
r_{i,t\rightarrow t+H}
=
\frac{P_{i,t+H}}{P_{i,t}}-1
```

XGBoost lernt die Abbildung

```math
f_\theta(\mathbf{x}_{i,t})
\rightarrow
\hat y_{i,t}
```

Der Modelloutput ist der Alpha-Score:

```math
\alpha_{i,t}
=
\hat y_{i,t}
```

Damit gilt:

- $\alpha_{i,t}>0$: erwartete Outperformance
- $\alpha_{i,t}<0$: erwartete Underperformance
- $|\alpha_{i,t}|$: Stärke der Prognose

Für einen Rebalancing-Zeitpunkt werden die Prognosen aller `N` Aktien zu einem
Alpha-Vektor zusammengefasst:

```math
\boldsymbol{\alpha}_t
=
[\alpha_{1,t},\alpha_{2,t},\ldots,\alpha_{N,t}]
```

Dieser Vektor geht in Risk Adjustment, Alpha-Regel und RL-State ein.

## 4. Risk Estimation & Risk-Adjusted Alpha

Die Risikoschätzung ist die rollierende Volatilität der täglichen Returns über
`vol_window = 20` Handelstage:

```math
\sigma_{i,t}
=
\mathrm{std}(r_{i,t-19},\ldots,r_{i,t})
```

Alpha und Risiko werden zu einem signierten Signal kombiniert. Gegen Division
durch null wird $\varepsilon = 10^{-8}$ zur Volatilität addiert:

```math
z_{i,t}
=
\frac{\alpha_{i,t}}{\sigma_{i,t}+\varepsilon}
```

Für das Position Sizing wird die richtungsunabhängige, auf Summe 1
normalisierte Opportunity verwendet:

```math
q_{i,t}
=
\frac{|\alpha_{i,t}|/(\sigma_{i,t}+\varepsilon)}
{\sum_{j=1}^{N}|\alpha_{j,t}|/(\sigma_{j,t}+\varepsilon)}
=
\frac{|z_{i,t}|}{\sum_{j=1}^{N}|z_{j,t}|}
```

Damit gilt $\sum_{i=1}^{N}q_{i,t}=1$, sofern die Summe der Beträge positiv
ist; sonst ist $q_{i,t}=0$. Ein starkes Alpha bei geringer Volatilität erhält
einen höheren Anteil als ein gleich starkes Alpha bei hohem Risiko.

## 5. RL Environment

Training und Evaluation verwenden PPO mit einem Multi-Discrete Action Space.
Der Agent sieht Alpha-Scores, Risiko-, Markt- und Portfolioinformationen sowie
im Residual-Setup die aktuelle Regelaktion. Fundamentaldaten gehen nicht direkt
in den PPO-State ein, sondern nur über die Alpha-Scores.

Ein optionaler SAC-Pfad mit kontinuierlichem Action Space `[-1, +1]^N`
existiert, ist aber nicht der evaluierte Default.

### Alpha Rule

Die Default-Regel ist ein Cross-Section-Z-Score von $\boldsymbol{\alpha}_t$ mit
Dead Zone `rule_z_threshold = 0.5` und Cost Floor
$|\alpha_{i,t}| \ge c_{TC}$:

- Buy, wenn $z^\alpha_{i,t} \ge 0.5$ und das Alpha den Cost Floor erreicht
- Sell, wenn $z^\alpha_{i,t} \le -0.5$ und das Alpha den Cost Floor erreicht
- sonst Hold

Eine Quantile-Regel (Top/Bottom 30 %) ist implementiert, aber nicht Default.

### State Space

Der Basis-State enthält pro Aktie Alpha, $q$, $z$, das Aktiengewicht, den
Return und die Haltedauer in Tagen. Zusätzlich wird der Cash-Anteil
angehängt; im Residual-Setup folgt die Regelaktion als $\{-1,0,+1\}$:

```math
s_t=
[
\boldsymbol{\alpha}_t,
\mathbf{q}_t,
\mathbf{z}_t,
\mathbf{w}_{t-1},
cash_t,
\mathbf{r}_t,
\mathbf{h}_t,
\mathbf{rule}_t
]
```

Dabei bezeichnet:

- $\boldsymbol{\alpha}_t$: Alpha-Scores für alle `N` Aktien
- $\mathbf{q}_t$: normalisierte risikoadjustierte Opportunity
- $\mathbf{z}_t$: $\alpha_{i,t}/(\sigma_{i,t}+\varepsilon)$
- $\mathbf{w}_{t-1}$: aktuelle Aktiengewichte (Länge $N$, ohne Cash)
- $cash_t$: aktueller Cash-Anteil (Skalar)
- $\mathbf{r}_t$: aktueller 1-Tages-Return je Aktie
- $\mathbf{h}_t$: Haltedauer in Kalendertagen seit Positionsöffnung
- $\mathbf{rule}_t$: aktuelle Alpha-Regelaktion, nur im Residual-Setup

Der Basis-State hat die Dimension $6N+1$. Mit Regelaktion sind es $7N+1$.
Beobachtungen werden mit `VecNormalize` normalisiert; der Reward bleibt
unnormalisiert.

### Action Space

Das Gymnasium-Environment nutzt `MultiDiscrete([3] * N)`. Intern werden die
Codes auf Handelsrichtungen abgebildet:

```math
\{0,1,2\}\mapsto\{-1,0,+1\},
\qquad
0=\mathrm{SELL},\;1=\mathrm{HOLD},\;2=\mathrm{BUY}
```

Im Default-Residual-Setup gibt PPO pro Aktie eine Residualaktion aus:

```text
0 = Down, 1 = Keep, 2 = Up
```

`Keep` übernimmt die Alpha-Regel. `Down` bzw. `Up` verschieben die Regelaktion
um einen Schritt in Richtung Sell bzw. Buy und clippen auf
$\{\mathrm{Sell},\mathrm{Hold},\mathrm{Buy}\}$. Der Agent entscheidet über die
Richtung, nicht über die Positionsgröße.

### Implementation Approach

XGBoost und PPO werden getrennt trainiert. Zuerst lernt XGBoost auf
historischen Features und aktiven 20-Tage-Renditen. Seine Point-in-Time-
Prognosen werden danach zusammen mit Risiko- und Portfolioinformationen als
Input des RL-Agenten verwendet.

Die Residual-Policy wird vor dem PPO-Fine-Tuning per Behavioral Cloning auf
`Keep` warmgestartet. Zusätzlich bleiben ein KEEP-Logit-Bias (`keep_bias = 1.5`)
und ein KEEP-Verlust auf jedem Rollout (`keep_coef = 0.08`) aktiv, damit die
Policy nahe an der Alpha-Regel bleibt.

Eine gemeinsame Klassifikation aller $3^N$ Aktionskombinationen wird
vermieden. PPO verwendet eine separate diskrete Verteilung pro Aktie.

Trainingsepisoden sind zufällige Fenster von 80 Schritten. Die Evaluation
läuft deterministisch über den vollen Split.

## 6. Risk-Adjusted Position Sizing

Default ist der inkrementelle Modus `rebalance_mode = "incremental"` mit
täglicher Entscheidung (`rebalance_every = 1`). Die finale Richtung
$a_{i,t}\in\{-1,0,+1\}$ wird mit Budget $B_t = 0.2$ und Opportunity $q_{i,t}$
in eine Gewichtänderung übersetzt:

```math
\Delta w_{i,t}
=
a_{i,t}\cdot B_t\cdot q_{i,t}
```

Ein Buy bei hohem $q_{i,t}$ kauft mehr als bei einem schwachen Signal. Hold
lässt die Position unverändert.

Optional existiert der Snapshot-Modus (`rebalance_mode = "snapshot"`), der
das Zielbuch komplett aus den Buy-/Sell-Namen neu setzt und $q$ nur innerhalb
der jeweiligen Seite normalisiert. Er wird für den 20-Tage-Horizont-Abgleich
(`--match-alpha-horizon`) verwendet, nicht für den täglichen Default.

## 7. Portfolio Constraints

Nach dem Position Sizing gilt im inkrementellen Modus:

```math
w_{i,t}^{target}
=
\mathrm{clip}(w_{i,t-1}+\Delta w_{i,t},\; 0,\; w_{max})
```

mit $w_{max} = 0.2$. Anschließend wird die Summe der Aktiengewichte auf
höchstens 1 skaliert. Nicht investiertes Kapital bleibt Cash. Das Environment
startet mit 100 % Cash (`initial_cash = 1\,000\,000`).

Long-only ist Default (`allow_short = False`). Im inkrementellen Modus blockiert
`min_holding_days = 5` Verkäufe, bevor eine neu eröffnete Long-Position diese
Mindesthaltedauer erreicht hat. Zwischen Rebalancing-Tagen (`rebalance_every > 1`)
werden Aktionen ignoriert und die Gewichte gehalten.

## 8. Rebalancing, Costs & Reward

Der Turnover ist die L1-Veränderung der Aktiengewichte:

```math
Turnover_t
=
\sum_{i=1}^{N}
|w_{i,t}-w_{i,t-1}|
```

Transaktionskosten reduzieren den Portfoliowert proportional zum Turnover.
Der Default-Kostensatz ist $c_{TC} = 10\,\mathrm{bp}$:

```math
C_t = c_{TC}\cdot Turnover_t,
\qquad
V_t^{+} = V_{t}^{-}\,(1-C_t)
```

Die nächste Portfoliorendite entsteht durch das Halten der neuen Positionen
über den Folgetag. Der **Reward** ist nicht die einfache Nettorendite, sondern:

```math
R_{t+1}
=
\log\!\left(\frac{V_{t+1}}{V_t^{-}}\right)
-
\lambda_{\mathrm{trade}}\cdot Turnover_t
+
\lambda_{\mathrm{align}}\sum_{i=1}^{N} a_{i,t}\,\mathrm{sign}(\alpha_{i,t})\,q_{i,t}
```

$V_{t+1}$ enthält bereits die Transaktionskosten. Zusätzlich gilt
$\lambda_{\mathrm{trade}} = 10\,\mathrm{bp}$ als Overtrading-Strafe auf den
Reward (nicht noch einmal auf den Portfoliowert) und
$\lambda_{\mathrm{align}} = 5\,\mathrm{bp}$ als Bonus, wenn die finale Richtung
mit dem Alpha-Vorzeichen übereinstimmt.

Sharpe Ratio, Maximum Drawdown, Volatilität, Turnover, Transaktionskosten
sowie kumulierte und annualisierte Rendite dienen der Evaluation, nicht als
zusätzliche Reward-Terme.

## 9. Baselines & Evaluation

Die primäre Evaluation vergleicht zwei separat trainierte Residual-PPO-
Systeme:

1. **`market_only`**: Alpha-Modell mit Market Features
2. **`full_alpha`**: Alpha-Modell mit Markt- und Fundamentaldaten

Beide verwenden denselben chronologischen Testsplit, dieselbe
Environment-Konfiguration (täglicher inkrementeller Default) und einen
deterministischen Rollout. Die PPO-Rollouts und die Referenzportfolios werden
auf demselben Return-Fenster mit 91 Perioden ausgewertet. Als wirtschaftliche
Referenzen werden **Buy-and-Hold** und **Equal Weight** berichtet. Sie
verwenden dasselbe Return-Fenster, aber nicht die PPO-Cash-Ausgangslage und
keine PPO-Environment-Transaktionskosten.

Alpha-Ranking, direkte Alpha-Regelportfolios, `no_levels` und optionale
20-Tage-Snapshot-Läufe sind ergänzende Diagnostik und nicht der primäre
PPO-Vergleich.

Die zentrale Forschungsfrage lautet:

> Welchen zusätzlichen Nutzen liefern Fundamentaldaten gegenüber reinen
> Marktdaten für datengetriebene Handelsentscheidungen in einem hybriden
> System aus Alpha-Prognose und Reinforcement Learning?

## 10. Component Responsibilities

```text
Data / Features      Informationen über Unternehmen und Markt
        ↓
Alpha Model          Wie attraktiv ist die Aktie? (aktive 20-Tage-Rendite)
        ↓
Risk Adjustment      Wie stark ist das Signal relativ zur Volatilität?
        ↓
Alpha Rule           Baseline Buy / Hold / Sell (z-score)
        ↓
RL Agent             Residual Down, Keep oder Up
        ↓
Final Action         Sell, Hold oder Buy
        ↓
Position Sizing      Δw = a · B_t · q
        ↓
Portfolio Layer      Clip, Gross-Cap, Min-Hold, Cash
        ↓
Rebalancing          Log-Return nach Kosten − Penalty + Alignment
```

- **Alpha Model:** prognostiziert die erwartete Out-/Underperformance.
- **Risk Adjustment:** bildet $z$ und $q$ aus Alpha und Volatilität.
- **Alpha Rule:** setzt das Alpha in eine Baseline-Richtung um.
- **RL Agent:** darf die Regel um höchstens einen Schritt verschieben.
- **Position Sizing:** übersetzt Richtung und $q$ in Gewichtänderungen.
- **Portfolio Layer:** erzwingt Long-only, $w_{max}$, Exposure- und Haltelimits.

## 11. Implemented Defaults

Die folgenden Werte sind der aktuelle Default, nicht offene Designfragen:

| Größe | Default |
| --- | --- |
| Universum | 20 Aktien (siehe Abschnitt 2) |
| Prognosehorizont `H` | 20 Handelstage |
| Alpha-Ziel | aktive Rendite vs. Cross-Section-Mittel |
| Alpha-Modell | XGBoost |
| Volatilitätsfenster | 20 Tage |
| Rebalancing | täglich, `incremental` |
| Residual-PPO | an, inkl. Keep-Prior und Imitation |
| `w_max` | 0.2 |
| Rebalancing-Budget `B_t` | 0.2 |
| Transaktionskosten | 10 bp auf den Portfoliowert |
| Reward | Log-Return nach Kosten, plus 10 bp Penalty und 5 bp Alignment |
| Cash | Start 100 % Cash |
| Buch | Long-only |

Experimentell vorhanden, aber nicht der primäre Vergleich:

- Snapshot-Rebalancing alle 20 Tage (`--match-alpha-horizon`)
- Feature-Ablation `no_levels`
- Quantile-Alpha-Regel
- Short Selling (`--allow-short`)
- SAC mit kontinuierlichem Action Space
- direktes (nicht-residuales) PPO
