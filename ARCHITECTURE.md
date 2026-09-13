# Hybrid RL Architecture for Quantitative Trading

## 1. Overview

Das System trennt **Prognose**, **Handelsentscheidung** und **Positionsgröße**:

- Das **Alpha-Modell** prognostiziert die Attraktivität einer Aktie.
- Das **Risk Adjustment** setzt das Signal ins Verhältnis zum Risiko.
- Eine **Alpha-Regel** erzeugt daraus eine Buy-/Hold-/Sell-Vorlage.
- Der **RL-Agent** entscheidet je Aktie zwischen **Buy, Hold und Sell**, standardmäßig als Residual um diese Regel.
- Das **Position Sizing** übersetzt Aktion und Signalstärke in eine Trade-Größe.
- **Portfolio Constraints** begrenzen die resultierenden Positionen und Trades.

```text
Market + Fundamental Data → Feature Engineering → Alpha Model
→ Alpha Scores → Risk Adjustment → Alpha Rule + RL Residual (Buy/Hold/Sell)
→ Position Sizing → Portfolio Constraints → Rebalancing
→ New Portfolio → Return − Transaction Costs → Reward
```

## 2. Data & Features

Verwendet werden Point-in-Time-Marktdaten (OHLCV) sowie SEC-Fundamentaldaten zu Profitabilität, Verschuldung, Umsätzen, Gewinn, Cashflow und Filing-Zeitpunkten.

Alle Daten und daraus berechneten Features müssen Point-in-Time korrekt sein: Zum Zeitpunkt `t` dürfen nur tatsächlich veröffentlichte und verfügbare Informationen eingehen. Das gilt besonders für Fundamentaldaten und verhindert Look-Ahead Bias.

Das Feature Engineering sampled den ersten Stundenbar jedes Kalendertags als Rebalancing-Zeitpunkt. Marktseitig gehen daraus `return_1d`, `return_5d`, `momentum_20d`, `vol_20d`, Open-Close-Rendite, High-Low-Range und Volumenmaße ein. Fundamentalseitig werden Ratio-Levels (`roe`, `gross_margin`, `debt_to_equity`), Accounting-Flows als Deltas (`revenue`, `net_income`, `operating_cashflow`), Filing-Kontext (`filing_lag_days`, `filing_recency`, `post_filing_5d`) sowie querschnittliche Ränge verwendet. Trainiert werden drei Feature-Sets: nur Markt, Fundamentals ohne sticky Levels, und das volle Set.

## 3. Alpha Model

Als Alpha-Modell ist XGBoost als überwachter Regressor implementiert. Das Modell prognostiziert für jede Aktie eine marktneutrale 20-Tage-Rendite, die anschließend als Alpha-Score dient. LightGBM bleibt ein mögliches Vergleichsmodell.

### Input & Output

Das Alpha-Modell wird für jede Aktie `i` und jeden Entscheidungszeitpunkt `t` angewendet. Sein Input ist ein Feature-Vektor:

```math
\mathbf{x}_{i,t}
=
[
\text{Market Features}_{i,t},
\text{Fundamental Features}_{i,t},
\text{Context Features}_t
]
```

Dabei steht $\mathbf{x}_{i,t}$ für alle zum Zeitpunkt `t` verfügbaren Informationen zur Aktie `i`. Die konkreten Markt- und Fundamentalfeatures sind in Abschnitt 2 aufgeführt. Alle Features müssen **Point-in-Time korrekt** sein: Es dürfen nur Informationen verwendet werden, die zum Zeitpunkt `t` tatsächlich bekannt waren.

Das Standard-Trainingsziel ist die marktneutrale Rendite über `H = 20` Handelstage: die absolute Forward-Rendite abzüglich des gleichgewichteten Querschnittsmittels im Universum.

```math
y_{i,t}
=
r_{i,t\rightarrow t+H}
-
\bar r_{t\rightarrow t+H},
\qquad
r_{i,t\rightarrow t+H}
=
\frac{P_{i,t+H}}{P_{i,t}}-1
```

Dabei bezeichnet:

- $P_{i,t}$ den Preis der Aktie `i` zum Entscheidungszeitpunkt `t`
- $P_{i,t+H}$ den Preis nach `H` Handelstagen
- $\bar r_{t\rightarrow t+H}$ das gleichgewichtete Mittel der Forward-Renditen aller Aktien am Tag `t`
- $y_{i,t}$ die erwartete Out- oder Underperformance, die im Training als Zielwert dient

Optional kann statt der aktiven Rendite die unbereinigte Forward-Rendite $r_{i,t\rightarrow t+H}$ trainiert werden.

XGBoost lernt aus den historischen Trainingsdaten die Abbildung:

```math
f_\theta(\mathbf{x}_{i,t})
\rightarrow
\hat r_{i,t+H}
```

Dabei ist $f_\theta$ das Alpha-Modell mit den gelernten Modellparametern $\theta$. Es verarbeitet den Feature-Vektor $x_{i,t}$ und prognostiziert mit $\hat r_{i,t+H}$ die marktneutrale Rendite der Aktie über die nächsten `H` Handelstage.

Der Modelloutput wird als Alpha-Score verwendet:

```math
\alpha_{i,t}
=
\hat r_{i,t+H}
```

Damit gilt:

- $\alpha_{i,t}>0$: erwartete Outperformance gegenüber dem Querschnitt
- $\alpha_{i,t}<0$: erwartete Underperformance
- $|\alpha_{i,t}|$: Stärke der Prognose

Für einen Rebalancing-Zeitpunkt werden die Prognosen aller `N` betrachteten Aktien zu einem Alpha-Vektor zusammengefasst:

```math
\boldsymbol{\alpha}_t
=
[\alpha_{1,t},\alpha_{2,t},\ldots,\alpha_{N,t}]
```

Dabei bezeichnet `N` die Anzahl der Aktien im aktuellen Universum. Der Alpha-Vektor enthält somit für jede Aktie genau einen prognostizierten Alpha-Score. Er wird anschließend für das Risk Adjustment und als Bestandteil des States des RL-Agenten verwendet.

Ein positives bzw. negatives Alpha signalisiert erwartete Out- bzw. Underperformance; der Betrag beschreibt die Signalstärke.

## 4. Risk Estimation & Risk-Adjusted Alpha

Als Risikoschätzung dient die historische rollierende Volatilität $\sigma_{i,t}$ über die letzten 20 Handelstage (`vol_window`). Sie wird für jede Aktie `i` zum Zeitpunkt `t` aus den 1-Tages-Returns berechnet.

Alpha und Risiko werden zu einem risikoadjustierten Signal kombiniert:

```math
z_{i,t}
=
\frac{|\alpha_{i,t}|}{\sigma_{i,t}+\varepsilon}
```

Dabei bezeichnet:

- $\alpha_{i,t}$ den Alpha-Score der Aktie `i` zum Zeitpunkt `t`
- $|\alpha_{i,t}|$ die Stärke des Signals unabhängig von seiner Richtung
- $\sigma_{i,t}$ die geschätzte Volatilität bzw. das Risiko der Aktie
- $\varepsilon$ einen kleinen Sicherheitswert gegen Division durch null
- $z_{i,t}$ die risikoadjustierte Signalstärke

Ein starkes Alpha bei geringer Volatilität erzeugt somit einen höheren Wert als ein gleich starkes Alpha bei hohem Risiko.

Anschließend wird $z_{i,t}$ über alle `N` Aktien des zum Zeitpunkt `t` betrachteten Universums normalisiert:

```math
q_{i,t}
=
\frac{z_{i,t}}{\sum_{j=1}^{N}z_{j,t}}
```

Der Index `j` iteriert dabei über alle Aktien des aktuellen Universums. Der Nenner ist die Summe ihrer risikoadjustierten Signalstärken. Dadurch gilt:

```math
\sum_{i=1}^{N}q_{i,t}=1
```

$q_{i,t}$ beschreibt den relativen Anteil der risikoadjustierten Opportunity einer Aktie und dient später dem Position Sizing. Zusätzlich geht die vorzeichenbehaftete Größe $z_{i,t}^{\mathrm{signed}}=\alpha_{i,t}/(\sigma_{i,t}+\varepsilon)$ in den RL-State ein.

## 5. RL Environment

Standard ist PPO mit einem Multi-Discrete Action Space. Der Agent sieht die von XGBoost erzeugten Alpha-Scores zusammen mit risikoadjustierten, Portfolio- und Haltedauer-Informationen. SAC mit kontinuierlichem Action Space ist als Alternative implementiert.

Das RL Environment verbindet das Alpha-Modell mit der Portfolio-Simulation. Zu jedem Rebalancing-Zeitpunkt `t` erhält der Agent einen State, wählt Aktionen und bekommt nach deren Ausführung einen Reward. Standardmäßig handelt PPO nicht direkt Buy/Hold/Sell, sondern ein Residual um eine Alpha-Regelpolitik.

### Alpha-Regel

Die Regelpolitik erzeugt aus dem Alpha-Vektor eine Buy-/Hold-/Sell-Vorlage. Default ist ein querschnittlicher z-Score mit toter Zone (`rule_z_threshold = 0.5`): Aktien mit $z\geq 0{,}5$ werden gekauft, mit $z\leq -0{,}5$ verkauft, dazwischen gehalten. Trades, deren $|\alpha|$ die Transaktionskosten nicht deckt, werden auf Hold gesetzt. Optional gibt es eine Quantil-Regel (festes Top-/Bottom-Quantil).

### State Space

Der implementierte State ist:

```math
s_t=
[
\boldsymbol{\alpha}_t,
\mathbf{q}_t,
\mathbf{z}_t^{\mathrm{signed}},
\mathbf{w}_t,
c_t,
\mathbf{r}_t,
\mathbf{h}_t
]
```

Dabei bezeichnet:

- $\boldsymbol{\alpha}_t$: von XGBoost erzeugte Alpha-Scores für alle `N` Aktien
- $\mathbf{q}_t$: relative risikoadjustierte Opportunity aus Abschnitt 4
- $\mathbf{z}_t^{\mathrm{signed}}$: vorzeichenbehaftetes Signal $\alpha_{i,t}/(\sigma_{i,t}+\varepsilon)$
- $\mathbf{w}_t$: aktuelle Aktiengewichte
- $c_t$: aktuelles Cash-Gewicht
- $\mathbf{r}_t$: 1-Tages-Returns
- $\mathbf{h}_t$: Haltedauer in Tagen je Position

Im Residual-Modus wird $s_t$ um die aktuelle Regelaktion als $\{-1,0,+1\}$ je Aktie ergänzt. Die Beobachtung wird vor dem Training mit `VecNormalize` skaliert; der Reward bleibt in den ursprünglichen Einheiten.

### Action Space

Für PPO ist der Output je Aktie `i` eine diskrete Aktion. Intern kodiert die Umgebung `{0,1,2}` als Sell/Hold/Buy; fachlich gilt:

```math
a_{i,t}\in\{-1,0,+1\},
\qquad
-1=SELL,\;0=HOLD,\;+1=BUY
```

Im Residual-Modus (PPO-Default) gibt die Policy statt der Handelsaktion eine Abweichung von der Regel aus:

```math
\delta_{i,t}\in\{\mathrm{DOWN},\;\mathrm{KEEP},\;\mathrm{UP}\}
```

KEEP folgt der Regel, DOWN/UP verschieben sie um eine Stufe in Richtung Sell bzw. Buy und clippen auf `{SELL, HOLD, BUY}`.

SAC verwendet statt diskreter Codes einen kontinuierlichen Vektor in $[-1,+1]^N$. Der Wert je Aktie ist direkt Richtung und Stärke des Trades; Residual-Aktionen entfallen dort.

Über alle `N` Aktien entsteht der Aktionsvektor $\mathbf{a}_t$. Der Agent entscheidet über die Handelsrichtung und bei SAC zusätzlich über die relative Stärke $|a_{i,t}|$. Die absolute Positionsgröße folgt anschließend aus Budget $B_t$ und $q_{i,t}$.

### Implementation Approach

XGBoost und der RL-Agent werden getrennt trainiert. Zuerst wird XGBoost auf historischen Features und zukünftigen aktiven Renditen trainiert. Seine Point-in-Time-Prognosen werden anschließend zusammen mit Risiko- und Portfolioinformationen als Input des RL-Agenten verwendet.

PPO erzeugt je Aktie drei Wahrscheinlichkeiten. Vor dem Fine-Tuning wird die Residual-Policy per Behavioral Cloning auf KEEP warmgestartet und mit einem KEEP-Bias sowie einer KEEP-Regularisierung nahe der Regel gehalten. Nach Auswahl und Ausführung der Aktionen berechnet das Environment die Positionsgrößen, wendet die Portfolio Constraints an und bestimmt die neuen Portfolio-Gewichte, Transaktionskosten, Portfoliorendite und den Reward.

Eine gemeinsame Klassifikation aller Aktionskombinationen wird vermieden, da bei `N` Aktien bereits $3^N$ Kombinationen entstehen. Stattdessen verwendet PPO eine separate diskrete Aktionsverteilung pro Aktie.

## 6. Risk-Adjusted Position Sizing

Im Default-Modus `incremental` übersetzt die Position-Sizing-Regel die RL-Aktion in eine Veränderung des Portfolio-Gewichts:

```math
\Delta w_{i,t}
=
a_{i,t}\cdot B_t\cdot q_{i,t}
```

Dabei bezeichnet:

- $\Delta w_{i,t}$ die Veränderung des Portfolio-Gewichts der Aktie `i` zum Zeitpunkt `t`
- $a_{i,t}\in[-1,+1]$ die Handelsrichtung (bei PPO diskret $\{-1,0,+1\}$, bei SAC kontinuierlich)
- $B_t=0{,}2$ das Rebalancing-Budget
- $q_{i,t}$ die in **Abschnitt 4 „Risk Estimation & Risk-Adjusted Alpha“** berechnete relative risikoadjustierte Signalstärke

Ein Buy bei einem hohen $q_{i,t}$ führt zu einem größeren Kauf als bei einem schwachen Signal. Ein Sell erzeugt entsprechend eine negative Positionsänderung. Für Hold gilt $\Delta w_{i,t}=0$; die bestehende Position bleibt unverändert.

Im Modus `snapshot` setzt die Umgebung statt inkrementeller Deltas ein vollständiges Zielbuch: Buy-Namen erhalten ein zu $q$ proportionales Long-Gewicht, Sell-Namen nur bei aktivierten Shorts ein negatives Gewicht. Regelbaselines und optionale Alpha-Horizont-Rebalances (`rebalance_every = 20`) nutzen diesen Modus. SAC unterstützt nur `incremental`.

## 7. Portfolio Constraints

Nach dem Position Sizing werden die Zielgewichte begrenzt. Im Modus `incremental` gilt:

```math
w_{i,t}^{target}
=
w_{i,t-1}+\Delta w_{i,t}
```

Im Modus `snapshot` wird $w_{i,t}^{target}$ direkt aus den Aktionen und $q_{i,t}$ gesetzt. Vor der Ausführung werden die Zielgewichte auf einen zulässigen Portfolioraum begrenzt.

Default ist **long-only** mit $w_{\max}=0{,}2$:

```math
0\leq w_{i,t}\leq w_{\max},
\qquad
\sum_i |w_{i,t}|\leq 1
```

Damit kann keine Aktie negativ gewichtet werden, das Gewicht einer einzelnen Aktie ist begrenzt und das Brutto-Exposure bleibt bei 100 %. Nicht investiertes Kapital wird als Cash gehalten. Optional sind Shorts erlaubt (`allow_short`); dann gilt $w_{i,t}\in[-w_{\max},w_{\max}]$ und das Brutto-Exposure ist auf 2 begrenzt. Im inkrementellen Modus blockiert eine Mindesthaltedauer von 5 Tagen vorzeitige Exits.

Die Constraints bilden somit die Sicherheitsschicht zwischen den vom Modell vorgeschlagenen Positionsänderungen und den tatsächlich ausgeführten Trades.

## 8. Rebalancing, Costs & Reward

Nach Anwendung der Portfolio Constraints werden die zulässigen Zielgewichte durch Käufe und Verkäufe umgesetzt. Der dabei entstehende Turnover misst die gesamte tatsächlich ausgeführte Portfolioveränderung:

```math
Turnover_t
=
\sum_{i=1}^{N}
|w_{i,t}-w_{i,t-1}|
```

Dabei sind $w_{i,t-1}$ und $w_{i,t}$ die Portfolio-Gewichte vor und nach dem Rebalancing. Über `i` wird dabei über alle `N` Aktien des Portfolios iteriert. Die Transaktionskosten werden proportional zum Turnover vom Portfoliowert abgezogen (`transaction_cost_bps = 10`):

```math
C_t=c_{TC}\cdot Turnover_t,
\qquad
V_t' = V_t\cdot(1-C_t)
```

Nach dem Rebalancing wachsen die Positionen mit den nächsten 1-Tages-Returns. Der Reward ist die Log-Rendite dieses Werts plus zwei zusätzliche Lernterme:

```math
R_{t+1}
=
\log\frac{V_{t+1}}{V_t}
-
\lambda_{\mathrm{trade}}\,Turnover_t
+
\lambda_{\mathrm{align}}\sum_{i=1}^{N}a_{i,t}\,\mathrm{sign}(\alpha_{i,t})\,q_{i,t}
```

$C_t$ steckt bereits in $V_{t+1}$. $\lambda_{\mathrm{trade}}$ (10 bp) bremst Overtrading zusätzlich im Lernsignal, ohne den Portfoliowert nochmals zu belasten. $\lambda_{\mathrm{align}}$ (5 bp) belohnt Aktionen, die mit dem Alpha-Vorzeichen übereinstimmen, gewichtet mit $q_{i,t}$.

Sharpe Ratio, Maximum Drawdown, Volatilität, Turnover, Transaktionskosten sowie Gesamt- und annualisierte Rendite dienen der Evaluation und sind vom Reward getrennt.

## 9. Baselines & Evaluation

Alle Varianten werden auf identischen Daten und Testzeiträumen verglichen:

1. **Buy & Hold**
2. **Equal Weight**
3. **Alpha-Regel** (z-Score, risikoadjustiertes Snapshot-Sizing)
4. **Alpha-Regel + Residual-RL** (PPO-Default; SAC als kontinuierliche Alternative)

Die zentrale Forschungsfrage lautet:

> Liefert der RL-Agent zusätzlichen Nutzen gegenüber einer direkten regelbasierten Verwendung derselben Alpha-Signale?

## 10. Component Responsibilities

```text
Data / Features      Informationen über Unternehmen und Markt
        ↓
Alpha Model          Wie attraktiv ist die Aktie relativ zum Querschnitt?
        ↓
Risk Adjustment      Wie stark ist das Signal relativ zum Risiko?
        ↓
Alpha Rule           Regel-Vorlage Buy / Hold / Sell
        ↓
RL Agent             Residual um die Regel, oder direkte Aktion (SAC)
        ↓
Position Sizing      Wie groß soll der Trade sein?
        ↓
Portfolio Layer      Ist der Trade innerhalb der Regeln möglich?
        ↓
Rebalancing          Log-Return − Trade-Penalty + Alignment → Reward
```

Die Kernidee ist die klare Trennung der Verantwortlichkeiten:

- **Alpha Model:** prognostiziert die zukünftige Out-/Underperformance eines Assets.
- **Risk Adjustment:** berücksichtigt das aktuelle Risiko.
- **Alpha Rule:** übersetzt das Signal in eine regelbasierte Handelsvorlage.
- **RL Agent:** korrigiert diese Vorlage kontextabhängig (PPO-Residual) oder gibt kontinuierliche Trade-Richtungen aus (SAC).
- **Position Sizing:** berechnet daraus konkrete Positionsänderungen.
- **Portfolio Layer:** setzt Constraints durch und bestimmt die tatsächlich ausführbaren Trades.

## 11. Open Design Decisions

Die folgenden Größen sind in der Implementierung festgelegt, bleiben aber experimentell vergleichbar:

- Prognosehorizont `H = 20`, tägliches Rebalancing als Default, optional `rebalance_every = 20`
- XGBoost als Alpha-Modell mit den Feature-Sets Markt / ohne Levels / voll
- Volatilität über 20 Tage, Risk-Adjusted Alpha mit $\varepsilon$
- $w_{\max}=0{,}2$, $B_t=0{,}2$, Transaktionskosten 10 bp
- PPO-Residual mit MultiDiscrete-Aktionen als Default; SAC mit $[-1,+1]^N$ als Alternative
- Long-only Default, Shorts optional; nicht investiertes Kapital als Cash

Weiterhin offen bzw. als Vergleich vorgesehen:

- LightGBM oder andere Alpha-Modelle
- DQN-Varianten
- welche Feature-Set- und Rebalancing-Kombination den RL-Mehrwert am klarsten zeigt
- Stärke der Reward-Terme $\lambda_{\mathrm{trade}}$ und $\lambda_{\mathrm{align}}$