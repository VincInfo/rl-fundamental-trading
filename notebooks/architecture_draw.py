"""Shared drawing code for the architecture report figures."""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1b2838"
MUTED = "#4b5b70"

PALETTE = {
    "forecast": ("#e7f1fa", "#2f5d8a"),
    "risk": ("#f6f0e2", "#8a6428"),
    "decision": ("#efeaf8", "#5a4784"),
    "portfolio": ("#e8eef3", "#3c5268"),
    "reward": ("#f3f4f6", "#1b2838"),
    "chip_market": ("#ffffff", "#2f5d8a"),
    "chip_fund": ("#ffffff", "#2f6b4f"),
    "chip_model": ("#ffffff", "#9c3d2e"),
    "chip_rule": ("#ffffff", "#5a4784"),
    "chip_rl": ("#fff7f4", "#9c3d2e"),
}

# Vertical budget so title, subtitle, and chip text never share the same band.
HEADER = 1.18
BOTTOM = 0.22
GAP = 0.34


def rounded(ax, x, y, w, h, fc, ec, lw=1.25, radius=0.10, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, x, y, w, h, title, body, fc, ec):
    rounded(ax, x, y, w, h, fc, ec, lw=1.25, radius=0.09, z=3)
    ax.text(
        x + w / 2,
        y + h - 0.32,
        title,
        ha="center",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=ec,
        zorder=4,
        clip_on=False,
    )
    body_top = y + h - 0.62
    body_bot = y + 0.22
    ax.text(
        x + w / 2,
        (body_top + body_bot) / 2,
        body,
        ha="center",
        va="center",
        fontsize=10.4,
        color=INK,
        linespacing=1.40,
        zorder=4,
        clip_on=False,
    )
    return x, y, w, h


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.5, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            linestyle=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
            shrinkA=3,
            shrinkB=3,
        )
    )


def layer(ax, x, y, w, h, number, title, subtitle, fc, ec):
    rounded(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.12, z=1)
    ax.plot(
        [x + 0.11, x + 0.11],
        [y + 0.18, y + h - 0.18],
        color=ec,
        lw=4.6,
        solid_capstyle="round",
        zorder=2,
    )
    ax.text(
        x + 0.36,
        y + h - 0.30,
        f"{number}   {title}",
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color=ec,
        zorder=4,
        clip_on=False,
    )
    ax.text(
        x + 0.36,
        y + h - 0.68,
        subtitle,
        ha="left",
        va="top",
        fontsize=11.0,
        color=MUTED,
        zorder=4,
        linespacing=1.32,
        clip_on=False,
    )
    return x, y, w, h


def _setup_axes(ax, top):
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0.0, top)
    ax.axis("off")


def _stack():
    """Bottom-up layer positions. Returns dict name -> (y, h, chip_y, chip_h)."""
    l5_h = 1.55
    l4_ch, l4_h = 1.28, HEADER + 1.28 + BOTTOM
    l3_ch, l3_h = 1.62, HEADER + 1.62 + BOTTOM
    l2_ch, l2_h = 1.32, HEADER + 1.32 + BOTTOM
    l1_ch, l1_h = 1.42, HEADER + 1.42 + BOTTOM

    y = 0.20
    l5 = (y, l5_h)
    y = y + l5_h + GAP
    l4 = (y, l4_h, y + BOTTOM, l4_ch)
    y = y + l4_h + GAP
    l3 = (y, l3_h, y + BOTTOM, l3_ch)
    y = y + l3_h + GAP
    l2 = (y, l2_h, y + BOTTOM, l2_ch)
    y = y + l2_h + GAP
    l1 = (y, l1_h, y + BOTTOM, l1_ch)
    top = y + l1_h + 0.18
    return l1, l2, l3, l4, l5, top


def _forecast_chips(ax, cy, ch, bodies):
    chip(ax, 0.52, cy, 2.52, ch, "Market data", bodies[0], *PALETTE["chip_market"])
    chip(ax, 3.32, cy, 2.62, ch, "Fundamentals", bodies[1], *PALETTE["chip_fund"])
    chip(ax, 6.22, cy, 2.52, ch, bodies[2][0], bodies[2][1], *PALETTE["chip_market"])
    chip(ax, 9.02, cy, 2.72, ch, "XGBoost alpha", bodies[3], *PALETTE["chip_model"])
    mid = cy + ch / 2
    arrow(ax, 3.04, mid, 3.32, mid, color=PALETTE["forecast"][1])
    arrow(ax, 5.94, mid, 6.22, mid, color=PALETTE["forecast"][1])
    arrow(ax, 8.74, mid, 9.02, mid, color=PALETTE["forecast"][1])


def _risk_chips(ax, cy, ch, titles_bodies):
    chip(ax, 0.52, cy, 3.48, ch, titles_bodies[0][0], titles_bodies[0][1], "#fffdf8", PALETTE["risk"][1])
    chip(ax, 4.28, cy, 3.62, ch, titles_bodies[1][0], titles_bodies[1][1], "#fffdf8", PALETTE["risk"][1])
    chip(ax, 8.18, cy, 3.58, ch, titles_bodies[2][0], titles_bodies[2][1], "#fffdf8", PALETTE["risk"][1])
    mid = cy + ch / 2
    arrow(ax, 4.00, mid, 4.28, mid, color=PALETTE["risk"][1])
    arrow(ax, 7.90, mid, 8.18, mid, color=PALETTE["risk"][1])


def _decision_chips(ax, cy, ch, specs):
    chip(ax, 0.52, cy, 4.42, ch, specs[0][0], specs[0][1], *PALETTE["chip_rule"])
    chip(ax, 5.22, cy, 3.28, ch, specs[1][0], specs[1][1], *PALETTE["chip_rl"])
    chip(ax, 8.78, cy, 2.98, ch, specs[2][0], specs[2][1], "#ffffff", PALETTE["chip_model"][1])
    mid = cy + ch / 2
    arrow(ax, 4.94, mid, 5.22, mid, color=PALETTE["decision"][1])
    arrow(ax, 8.50, mid, 8.78, mid, color=PALETTE["decision"][1])


def _portfolio_chips(ax, cy, ch, specs):
    chip(ax, 0.52, cy, 3.48, ch, specs[0][0], specs[0][1], "#ffffff", PALETTE["portfolio"][1])
    chip(ax, 4.28, cy, 3.62, ch, specs[1][0], specs[1][1], "#ffffff", PALETTE["portfolio"][1])
    chip(ax, 8.18, cy, 3.58, ch, specs[2][0], specs[2][1], "#ffffff", PALETTE["portfolio"][1])
    mid = cy + ch / 2
    arrow(ax, 4.00, mid, 4.28, mid, color=PALETTE["portfolio"][1])
    arrow(ax, 7.90, mid, 8.18, mid, color=PALETTE["portfolio"][1])


def _connect_layers(ax, y_bottom_of_upper, y_top_of_lower):
    arrow(ax, 6.15, y_bottom_of_upper, 6.15, y_top_of_lower + 0.02, color=INK)


def draw_architecture(ax):
    l1, l2, l3, l4, l5, top = _stack()
    _setup_axes(ax, top)

    layer(
        ax, 0.22, l1[0], 11.86, l1[1],
        "1", "Forecast",
        "Point-in-time market and SEC features  →  XGBoost predicts 20-day active return",
        *PALETTE["forecast"],
    )
    _forecast_chips(
        ax, l1[2], l1[3],
        [
            "OHLCV  ·  returns\nmomentum  ·  volume",
            "ROE, margins, D/E\nflows, filings, ranks",
            (r"Features  $x_{i,t}$", "daily first bar\nno look-ahead"),
            r"$\alpha_{i,t}=\hat r_{t\rightarrow t+H}$" + "\n" + r"$H=20$, active $y$",
        ],
    )

    layer(
        ax, 0.22, l2[0], 11.86, l2[1],
        "2", "Risk adjustment",
        r"Rolling 20-day volatility scales signal strength; $q$ later sizes the trade",
        *PALETTE["risk"],
    )
    _risk_chips(
        ax, l2[2], l2[3],
        [
            (r"Volatility  $\sigma_{i,t}$", "std of 1-day returns\n20-day window"),
            (r"$z_{i,t}=|\alpha_{i,t}|/(\sigma_{i,t}+\varepsilon)$", "risk-adjusted\nopportunity"),
            (r"$q_{i,t}=z_{i,t}/\sum_j z_{j,t}$", r"cross-section weights" + "\n" + r"$\sum q=1$"),
        ],
    )
    _connect_layers(ax, l1[0], l2[0] + l2[1])

    layer(
        ax, 0.22, l3[0], 11.86, l3[1],
        "3", "Decision  —  residual PPO (default)",
        "Rule proposes Buy/Hold/Sell; PPO outputs a residual around that rule, not a raw action",
        *PALETTE["decision"],
    )
    _decision_chips(
        ax, l3[2], l3[3],
        [
            (
                "Alpha rule",
                "z-score rule, dead zone 0.5\n"
                + r"buy if $z\geq 0.5$, sell if $z\leq -0.5$"
                + "\nhold the middle; skip tiny alpha",
            ),
            (
                r"PPO residual  $\delta$",
                r"$\delta\in\{\mathrm{DOWN},\mathrm{KEEP},\mathrm{UP}\}$"
                + "\nKEEP follows the rule\nagent sees the rule action",
            ),
            (
                r"Action  $a_{i,t}$",
                r"compose rule $+\,\delta$" + "\n" + r"$a\in\{-1,0,+1\}$" + "\nSell / Hold / Buy",
            ),
        ],
    )
    _connect_layers(ax, l2[0], l3[0] + l3[1])

    layer(
        ax, 0.22, l4[0], 11.86, l4[1],
        "4", "Position sizing, constraints, rebalancing",
        r"Incremental default:  $\Delta w_{i,t}=a_{i,t}\,B_t\,q_{i,t}$   with  $B_t=0.2$,  $w_{\max}=0.2$,  long-only",
        *PALETTE["portfolio"],
    )
    _portfolio_chips(
        ax, l4[2], l4[3],
        [
            ("Sizing", r"$\Delta w = a\cdot B_t\cdot q$"),
            ("Constraints", r"$0\leq w\leq w_{\max}$,  $\sum |w|\leq 1$" + "\n5-day min-hold"),
            ("Rebalancing", r"turnover $\rightarrow$ costs 10 bp" + "\n" + r"$\rightarrow$ next-day $r$"),
        ],
    )
    _connect_layers(ax, l3[0], l4[0] + l4[1])

    layer(
        ax, 0.22, l5[0], 11.86, l5[1],
        "5", "Reward",
        r"$R_{t+1}=\log(V_{t+1}/V_t)-\lambda_{\mathrm{trade}}\,\mathrm{Turnover}_t+\lambda_{\mathrm{align}}\sum_i a_{i,t}\,\mathrm{sign}(\alpha_{i,t})\,q_{i,t}$"
        + "\n"
        + r"Costs already in $V$;  $\lambda_{\mathrm{trade}}=10\,\mathrm{bp}$,  $\lambda_{\mathrm{align}}=5\,\mathrm{bp}$.",
        *PALETTE["reward"],
    )
    _connect_layers(ax, l4[0], l5[0] + l5[1])


def _overview_card(ax, x, y, w, h, number, title, blurb, out, fc, ec):
    """One architecture stage: name on the left, output quantity on the right."""
    rounded(ax, x, y, w, h, fc, ec, lw=1.15, radius=0.08, z=1)
    ax.plot(
        [x + 0.10, x + 0.10],
        [y + 0.14, y + h - 0.14],
        color=ec,
        lw=3.0,
        solid_capstyle="round",
        zorder=2,
    )
    ax.text(
        x + 0.32,
        y + h * 0.66,
        f"{number}   {title}",
        ha="left",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=ec,
        zorder=4,
        clip_on=False,
    )
    ax.text(
        x + 0.32,
        y + h * 0.30,
        blurb,
        ha="left",
        va="center",
        fontsize=11.0,
        color=MUTED,
        zorder=4,
        clip_on=False,
    )
    pw, ph = 2.05, 0.56
    px = x + w - pw - 0.18
    py = y + (h - ph) / 2
    rounded(ax, px, py, pw, ph, "#ffffff", ec, lw=1.05, radius=0.07, z=3)
    ax.text(
        px + pw / 2,
        py + ph / 2,
        out,
        ha="center",
        va="center",
        fontsize=13.5,
        color=ec,
        zorder=4,
        clip_on=False,
    )
    return x, y, w, h


def draw_architecture_overview(ax):
    """Methods overview for the report: five components and the quantities they produce.

    Intentionally no formulas — those follow in the subsequent paragraphs.
    """
    steps = [
        (
            "1",
            "Feature-Konstruktion",
            "Marktdaten und Fundamentaldaten",
            r"$x_{i,t}$",
            "#eef3f8",
            "#3a5670",
        ),
        (
            "2",
            "Alpha-Modell (XGBoost)",
            r"Prognose relativer Attraktivität",
            r"$\alpha_{i,t}$",
            "#f6eeea",
            "#8a4034",
        ),
        (
            "3",
            "Risikoadjustierung",
            "Signal relativ zur Volatilität",
            r"$q_{i,t}$",
            "#f4efe4",
            "#7a5a28",
        ),
        (
            "4",
            "RL-Agent (PPO)",
            "Sequenzielle Handelsentscheidung",
            r"$a_{i,t}$",
            "#efeaf6",
            "#5a4784",
        ),
        (
            "5",
            "Portfolio-Konstruktion",
            "Positionsgröße und Constraints",
            r"$w_{t}$",
            "#e8eef3",
            "#3c5268",
        ),
    ]
    h, gap, reward_h = 1.02, 0.26, 0.78
    y = 0.16
    reward_y = y
    y = y + reward_h + gap
    boxes = []
    for _ in reversed(steps):
        boxes.append((y, h))
        y = y + h + gap
    boxes.reverse()
    top = boxes[0][0] + h + 0.10
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0.0, top)
    ax.axis("off")

    x, w = 0.28, 10.64
    drawn = []
    for (by, bh), step in zip(boxes, steps):
        drawn.append(_overview_card(ax, x, by, w, bh, *step))
    for upper, lower in zip(drawn, drawn[1:]):
        arrow(ax, 5.6, upper[1], 5.6, lower[1] + lower[3] + 0.02, color=INK)

    last = drawn[-1]
    arrow(ax, 5.6, last[1], 5.6, reward_y + reward_h + 0.02, color=INK)

    rounded(ax, x, reward_y, w, reward_h, "#f4f5f6", "#5c6773", lw=1.05, radius=0.08, z=1)
    ax.text(
        x + 0.32,
        reward_y + reward_h * 0.66,
        "Reward",
        ha="left",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color="#5c6773",
        zorder=4,
        clip_on=False,
    )
    ax.text(
        x + 0.32,
        reward_y + reward_h * 0.30,
        "Portfoliorendite minus Transaktionskosten",
        ha="left",
        va="center",
        fontsize=10.6,
        color=MUTED,
        zorder=4,
        clip_on=False,
    )
    pw, ph = 2.05, 0.50
    px = x + w - pw - 0.18
    py = reward_y + (reward_h - ph) / 2
    rounded(ax, px, py, pw, ph, "#ffffff", "#5c6773", lw=1.05, radius=0.07, z=3)
    ax.text(
        px + pw / 2,
        py + ph / 2,
        r"$R_{t+1}$",
        ha="center",
        va="center",
        fontsize=13.5,
        color="#5c6773",
        zorder=4,
        clip_on=False,
    )
