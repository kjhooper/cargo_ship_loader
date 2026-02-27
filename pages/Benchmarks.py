"""Benchmark Results Viewer

Pre-computed comparison of all solvers, ships, and scenarios.

Run benchmarks first:
    conda run -n personal python benchmark.py

Then restart the Streamlit app to see results here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_PATH = Path(__file__).parent.parent / "benchmark_results.json"

SOLVER_ORDER = [
    "greedy", "beam_search", "simulated_annealing",
    "bayesian_opt", "neural_ranker", "rl_bayesian", "rl_bayesian_sa",
]
SOLVER_DISPLAY = {
    "greedy":              "Greedy",
    "beam_search":         "Beam Search",
    "simulated_annealing": "Sim. Annealing",
    "bayesian_opt":        "Bayesian Opt",
    "neural_ranker":       "Neural Ranker",
    "rl_bayesian":         "RL Bayesian",
    "rl_bayesian_sa":      "RL Bayes + SA",
}
SHIP_ORDER     = ["coastal", "handymax", "panamax"]
SCENARIO_ORDER = ["balanced", "weight_limited", "space_limited", "mixed"]
ML_SOLVERS     = ["neural_ranker", "rl_bayesian"]

SOLVER_COLORS = {
    "greedy":              "#60a5fa",
    "beam_search":         "#34d399",
    "simulated_annealing": "#f59e0b",
    "bayesian_opt":        "#f87171",
    "neural_ranker":       "#a78bfa",
    "rl_bayesian":         "#fb923c",
    "rl_bayesian_sa":      "#e879f9",
}
SHIP_COLORS = {"coastal": "#60a5fa", "handymax": "#34d399", "panamax": "#f59e0b"}

_DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_json() -> Optional[dict]:
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as fh:
        return json.load(fh)


def _to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "error" not in df.columns:
        df["error"] = None
    return df


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Return only non-error rows."""
    return df[df["error"].isna()].copy()


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _base_heatmap(
    z: np.ndarray,
    x: list,
    y: list,
    text: list,
    title: str,
    colorscale: str,
    zmin: float,
    zmax: float,
    zmid: Optional[float] = None,
    colorbar_title: str = "",
    height: int = 320,
) -> go.Figure:
    kwargs = dict(
        z=z, x=x, y=y,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=text,
        texttemplate="%{text}",
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>",
        colorbar=dict(thickness=14, len=0.85, title=dict(text=colorbar_title, side="right")),
    )
    if zmid is not None:
        kwargs["zmid"] = zmid
    fig = go.Figure(go.Heatmap(**kwargs))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=dict(l=100, r=20, t=55, b=55),
        **_DARK,
    )
    return fig


def _diagonal_boxes(fig: go.Figure, keys: list, color: str = "#facc15") -> None:
    """Overlay golden rectangles on the diagonal of a square heatmap."""
    for i in range(len(keys)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color=color, width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )


# ── Overview plots ─────────────────────────────────────────────────────────────

def plot_score_heatmap(df: pd.DataFrame, scenario: Optional[str]) -> go.Figure:
    sub = _ok(df)
    if scenario and scenario != "All":
        sub = sub[sub["scenario"] == scenario]
    grouped = sub.groupby(["solver_name", "ship_key"])["final_score"].mean()

    solvers = [s for s in SOLVER_ORDER if s in grouped.index.get_level_values(0).unique()]
    ships   = [s for s in SHIP_ORDER   if s in grouped.index.get_level_values(1).unique()]

    z, text = [], []
    for solver in solvers:
        row_z, row_t = [], []
        for ship in ships:
            val = grouped.get((solver, ship), np.nan)
            row_z.append(val)
            row_t.append(f"{val:.3f}" if not np.isnan(val) else "N/A")
        z.append(row_z)
        text.append(row_t)

    y_labels = [SOLVER_DISPLAY.get(s, s) for s in solvers]
    title = "Mean Final Score — " + (scenario if scenario and scenario != "All" else "All Scenarios")
    return _base_heatmap(
        np.array(z, dtype=float), ships, y_labels, text,
        title, "RdYlGn", 0.88, 1.0, colorbar_title="Score", height=330,
    )


def plot_runtime_scatter(df: pd.DataFrame) -> go.Figure:
    sub = _ok(df)
    agg = (
        sub.groupby(["solver_name", "ship_key"])
        .agg(mean_score=("final_score", "mean"), mean_runtime=("runtime_s", "mean"))
        .reset_index()
    )

    fig = go.Figure()
    for solver in SOLVER_ORDER:
        rows = agg[agg["solver_name"] == solver]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["mean_runtime"],
            y=rows["mean_score"],
            mode="markers+text",
            name=SOLVER_DISPLAY.get(solver, solver),
            text=rows["ship_key"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=13, color=SOLVER_COLORS.get(solver, "#aaa")),
            hovertemplate=(
                f"<b>{SOLVER_DISPLAY.get(solver, solver)}</b><br>"
                "Ship: %{text}<br>"
                "Runtime: %{x:.2f} s<br>"
                "Score: %{y:.4f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7,
                  annotation_text="0.92", annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_xaxes(type="log", title="Mean Runtime (s) — log scale")
    fig.update_yaxes(title="Mean Final Score", range=[0.84, 1.01])
    fig.update_layout(
        title=dict(text="Runtime vs Quality Tradeoff", font=dict(size=13)),
        height=330,
        margin=dict(l=60, r=20, t=55, b=55),
        legend=dict(orientation="v", x=1.02, font=dict(size=10)),
        **_DARK,
    )
    return fig


def plot_placement_heatmap(df: pd.DataFrame) -> go.Figure:
    sub = _ok(df)
    grouped = sub.groupby(["solver_name", "scenario"])["pct_placed"].mean()

    solvers   = [s for s in SOLVER_ORDER   if s in grouped.index.get_level_values(0).unique()]
    scenarios = [s for s in SCENARIO_ORDER if s in grouped.index.get_level_values(1).unique()]

    z, text = [], []
    for solver in solvers:
        row_z, row_t = [], []
        for sc in scenarios:
            val = grouped.get((solver, sc), np.nan)
            row_z.append(val)
            row_t.append(f"{val:.0f}%" if not np.isnan(val) else "N/A")
        z.append(row_z)
        text.append(row_t)

    y_labels = [SOLVER_DISPLAY.get(s, s) for s in solvers]
    return _base_heatmap(
        np.array(z, dtype=float), scenarios, y_labels, text,
        "Placement Rate (%) by Solver & Scenario",
        "RdYlGn", 0.0, 100.0, colorbar_title="%", height=310,
    )


def plot_cog_bars(df: pd.DataFrame) -> go.Figure:
    sub = _ok(df)
    if "cog_height_norm" not in sub.columns:
        return go.Figure()
    sub = sub[sub["cog_height_norm"] > 0]
    agg = sub.groupby(["solver_name", "ship_key"])["cog_height_norm"].mean().reset_index()

    fig = go.Figure()
    for ship in SHIP_ORDER:
        rows = agg[agg["ship_key"] == ship]
        if rows.empty:
            continue
        # preserve solver order
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in rows["solver_name"].values]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=ship.capitalize(),
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["cog_height_norm"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.3f}" for v in rows["cog_height_norm"]],
            textposition="outside",
            hovertemplate=f"Ship: {ship}<br>Solver: %{{x}}<br>CoG: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Centre-of-Gravity Height  (lower = more stable)", font=dict(size=13)),
        yaxis=dict(title="Normalised CoG height", range=[0, None]),
        height=320,
        margin=dict(l=60, r=20, t=55, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


# ── Scenario Deep-Dive plots ───────────────────────────────────────────────────

def plot_scenario_score(df: pd.DataFrame, scenario: str) -> go.Figure:
    sub = _ok(df[df["scenario"] == scenario])
    agg = sub.groupby(["solver_name", "ship_key"])["final_score"].mean().reset_index()

    fig = go.Figure()
    for ship in SHIP_ORDER:
        rows = agg[agg["ship_key"] == ship]
        if rows.empty:
            continue
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in rows["solver_name"].values]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=ship, x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["final_score"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.3f}" for v in rows["final_score"]],
            textposition="outside",
            hovertemplate=f"Ship: {ship}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7)
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Final Score — {scenario}", font=dict(size=13)),
        yaxis=dict(title="Final Score", range=[0.5, 1.06]),
        height=340,
        margin=dict(l=60, r=20, t=55, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_scenario_runtime(df: pd.DataFrame, scenario: str) -> go.Figure:
    sub = _ok(df[df["scenario"] == scenario])
    agg = sub.groupby(["solver_name", "ship_key"])["runtime_s"].mean().reset_index()

    fig = go.Figure()
    for ship in SHIP_ORDER:
        rows = agg[agg["ship_key"] == ship]
        if rows.empty:
            continue
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in rows["solver_name"].values]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=ship, x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["runtime_s"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.2f}s" for v in rows["runtime_s"]],
            textposition="outside",
            hovertemplate=f"Ship: {ship}<br>%{{x}}: %{{y:.2f}} s<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Mean Runtime — {scenario}", font=dict(size=13)),
        yaxis=dict(title="Runtime (s)", type="log"),
        height=340,
        margin=dict(l=60, r=20, t=55, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_balance_breakdown(df: pd.DataFrame, scenario: str) -> go.Figure:
    """PS / FA / Diag ratios side by side for a given scenario."""
    sub = _ok(df[df["scenario"] == scenario])
    metrics = {
        "PS ratio":   "ps_ratio",
        "FA ratio":   "fa_ratio",
        "Diag ratio": "diag_ratio",
    }
    colors = ["#60a5fa", "#34d399", "#a78bfa"]

    agg = sub.groupby("solver_name")[list(metrics.values())].mean().reset_index()
    agg = agg.set_index("solver_name").reindex(
        [s for s in SOLVER_ORDER if s in agg["solver_name"].values]
    ).reset_index()

    fig = go.Figure()
    for (label, col), color in zip(metrics.items(), colors):
        fig.add_trace(go.Bar(
            name=label,
            x=[SOLVER_DISPLAY.get(s, s) for s in agg["solver_name"]],
            y=agg[col],
            marker_color=color,
            text=[f"{v:.3f}" for v in agg[col]],
            textposition="outside",
        ))

    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7,
                  annotation_text="0.92", annotation_font_color="#f59e0b")
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Balance Ratios — {scenario}  (mean across ships & seeds)", font=dict(size=13)),
        yaxis=dict(title="Ratio", range=[0.7, 1.06]),
        height=340,
        margin=dict(l=60, r=20, t=55, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


# ── Transfer Analysis plots ────────────────────────────────────────────────────

def plot_transfer_pair(
    df_tr: pd.DataFrame, solver_name: str
) -> tuple[Optional[go.Figure], Optional[go.Figure], list[tuple]]:
    """Return (score_heatmap, degradation_heatmap, list_of_significant_drops)."""
    sub = _ok(df_tr[df_tr["solver_name"] == solver_name])
    if sub.empty:
        return None, None, []

    grouped = sub.groupby(["ship_key", "model_key"])["final_score"].mean()

    # ── Score heatmap ──────────────────────────────────────────────────────────
    z, text = [], []
    for ship in SHIP_ORDER:
        row_z, row_t = [], []
        for model in SHIP_ORDER:
            val = grouped.get((ship, model), np.nan)
            row_z.append(val)
            star = " ★" if ship == model else ""
            row_t.append(f"{val:.3f}{star}" if not np.isnan(val) else "N/A")
        z.append(row_z)
        text.append(row_t)

    z_arr = np.array(z, dtype=float)
    fig_score = _base_heatmap(
        z_arr, SHIP_ORDER, SHIP_ORDER, text,
        f"{SOLVER_DISPLAY.get(solver_name, solver_name)} — Final Score (ship × model)",
        "RdYlGn", 0.88, 1.0, colorbar_title="Score", height=320,
    )
    fig_score.update_xaxes(title_text="Model trained on →")
    fig_score.update_yaxes(title_text="Ship tested on →")
    _diagonal_boxes(fig_score, SHIP_ORDER)

    # ── Degradation heatmap ────────────────────────────────────────────────────
    dz, dtext = [], []
    significant: list[tuple] = []
    for i, ship in enumerate(SHIP_ORDER):
        matched = grouped.get((ship, ship), np.nan)
        row_dz, row_dt = [], []
        for j, model in enumerate(SHIP_ORDER):
            val = grouped.get((ship, model), np.nan)
            if np.isnan(val) or np.isnan(matched):
                row_dz.append(np.nan)
                row_dt.append("N/A")
            else:
                delta = val - matched
                row_dz.append(delta)
                row_dt.append(f"{delta:+.3f}")
                if ship != model and delta < -0.03:
                    significant.append((ship, model, float(val), float(delta)))
        dz.append(row_dz)
        dtext.append(row_dt)

    dz_arr = np.array(dz, dtype=float)
    dmin = float(np.nanmin(dz_arr)) if not np.all(np.isnan(dz_arr)) else -0.05
    dmax = float(np.nanmax(dz_arr)) if not np.all(np.isnan(dz_arr)) else 0.05
    # Symmetric scale centred on zero; widen slightly so diagonal (0) is white
    bound = max(abs(dmin), abs(dmax), 0.02)

    fig_deg = _base_heatmap(
        dz_arr, SHIP_ORDER, SHIP_ORDER, dtext,
        f"{SOLVER_DISPLAY.get(solver_name, solver_name)} — Score Δ vs. Matched (diagonal)",
        "RdBu", zmin=-bound, zmax=bound, zmid=0.0,
        colorbar_title="Δ Score", height=320,
    )
    fig_deg.update_xaxes(title_text="Model trained on →")
    fig_deg.update_yaxes(title_text="Ship tested on →")
    _diagonal_boxes(fig_deg, SHIP_ORDER)

    return fig_score, fig_deg, sorted(significant, key=lambda x: x[3])


# ── Transfer Analysis — aggregated regime charts ───────────────────────────────

MODEL_VARIANT_COLORS = {
    "coastal":  "#60a5fa",
    "handymax": "#34d399",
    "panamax":  "#f59e0b",
    "avg":      "#94a3b8",
}


def _make_transfer_bars(
    df_tr: pd.DataFrame,
    x_key: str,
    bar_key: str,
    avg_label: str,
    title: str,
    x_title: str,
) -> Optional[go.Figure]:
    """Shared logic for ship-perspective and model-fragility charts."""
    from plotly.subplots import make_subplots

    solvers_present = [s for s in ML_SOLVERS if s in df_tr["solver_name"].values]
    if not solvers_present:
        return None

    fig = make_subplots(
        rows=1, cols=len(solvers_present),
        subplot_titles=[SOLVER_DISPLAY.get(s, s) for s in solvers_present],
        shared_yaxes=True,
    )
    first = True

    for col_i, solver in enumerate(solvers_present, start=1):
        sub     = _ok(df_tr[df_tr["solver_name"] == solver])
        grouped = sub.groupby(["ship_key", "model_key"])["final_score"].mean()
        x_vals  = [s for s in SHIP_ORDER if s in sub[x_key].unique()]

        bar_variants = SHIP_ORDER + ["avg"]
        for bar_val in bar_variants:
            ys, texts, lines = [], [], []
            for xv in x_vals:
                if bar_val == "avg":
                    kv = [
                        grouped.get((xv, bv) if x_key == "ship_key" else (bv, xv), np.nan)
                        for bv in SHIP_ORDER
                    ]
                    val = np.nanmean(kv)
                    lines.append(dict(color="#94a3b8", width=1))
                else:
                    pair = (xv, bar_val) if x_key == "ship_key" else (bar_val, xv)
                    val  = grouped.get(pair, np.nan)
                    in_regime = (xv == bar_val)
                    lines.append(dict(
                        color="#facc15" if in_regime else "rgba(0,0,0,0)",
                        width=2.5,
                    ))
                ys.append(val)
                texts.append(f"{val:.3f}" if not np.isnan(val) else "N/A")

            if bar_val == "avg":
                lbl = avg_label
            else:
                side = "Model" if bar_key == "model_key" else "Ship"
                lbl = f"{side}: {bar_val.capitalize()}"

            fig.add_trace(go.Bar(
                name=lbl,
                x=[v.capitalize() for v in x_vals],
                y=ys,
                text=texts,
                textposition="outside",
                marker=dict(color=MODEL_VARIANT_COLORS[bar_val], line=lines),
                showlegend=first,
            ), row=1, col=col_i)
        first = False
        fig.update_xaxes(title_text=x_title, row=1, col=col_i)

    fig.add_hline(
        y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7,
        annotation_text="0.92", annotation_font_color="#f59e0b",
    )
    fig.update_yaxes(range=[0.82, 1.07], title_text="Final Score", col=1)
    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=13)),
        height=400,
        margin=dict(l=60, r=20, t=80, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, font=dict(size=10)),
        **_DARK,
    )
    return fig


def plot_ship_perspective(df_tr: pd.DataFrame) -> Optional[go.Figure]:
    """Which model works best on each ship? X = ship tested on."""
    return _make_transfer_bars(
        df_tr,
        x_key="ship_key",
        bar_key="model_key",
        avg_label="Ship avg (any model)",
        title="Ship Perspective — Which model works best on each ship?",
        x_title="Ship tested on →",
    )


def plot_model_fragility(df_tr: pd.DataFrame) -> Optional[go.Figure]:
    """How robust is each trained model across ship sizes? X = model trained on."""
    return _make_transfer_bars(
        df_tr,
        x_key="model_key",
        bar_key="ship_key",
        avg_label="Model avg (any ship)",
        title="Model Fragility — How robust is each trained model across ship sizes?",
        x_title="Model trained on →",
    )


def _regime_summary_table(df_tr: pd.DataFrame) -> pd.DataFrame:
    """Flat table: solver × ship tested × model trained, with Δ vs in-regime score."""
    rows = []
    for solver in ML_SOLVERS:
        sub = _ok(df_tr[df_tr["solver_name"] == solver])
        if sub.empty:
            continue
        grouped = sub.groupby(["ship_key", "model_key"])["final_score"].mean()
        for ship in SHIP_ORDER:
            if ship not in sub["ship_key"].unique():
                continue
            in_val = grouped.get((ship, ship), np.nan)
            for model in SHIP_ORDER:
                val = grouped.get((ship, model), np.nan)
                delta = (val - in_val) if not (np.isnan(val) or np.isnan(in_val)) else np.nan
                label = model + (" ★" if model == ship else "")
                rows.append({
                    "Solver":           SOLVER_DISPLAY.get(solver, solver),
                    "Ship tested":      ship,
                    "Model trained on": label,
                    "Score":            round(val,   4) if not np.isnan(val)   else None,
                    "Δ in-regime":      round(delta, 4) if not np.isnan(delta) else None,
                })
            # General-avg synthetic row
            all_vals  = [grouped.get((ship, m), np.nan) for m in SHIP_ORDER]
            avg_val   = np.nanmean(all_vals)
            avg_delta = (avg_val - in_val) if not np.isnan(in_val) else np.nan
            rows.append({
                "Solver":           SOLVER_DISPLAY.get(solver, solver),
                "Ship tested":      ship,
                "Model trained on": "avg (general)",
                "Score":            round(avg_val,   4),
                "Δ in-regime":      round(avg_delta, 4) if not np.isnan(avg_delta) else None,
            })
    return pd.DataFrame(rows)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Benchmarks — Cargo Ship Loader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Benchmark Results")
st.caption(
    "Pre-computed comparison of all solvers across ship sizes, loading scenarios, "
    "and cross-ship transfer tests."
)

# ── Load data ──────────────────────────────────────────────────────────────────

data = _load_json()

if data is None:
    st.warning(
        f"**No benchmark results found** at `{RESULTS_PATH.name}`.\n\n"
        "Generate them with:\n"
        "```\n"
        "conda run -n personal python benchmark.py\n"
        "```\n"
        "Quick smoke test (coastal + balanced only, ~1 min):\n"
        "```\n"
        "conda run -n personal python benchmark.py "
        "--ships coastal --scenarios balanced --no-transfer\n"
        "```"
    )
    st.stop()

# ── Metadata strip ────────────────────────────────────────────────────────────

meta = data.get("metadata", {})
mc1, mc2, mc3, mc4, mc5, mc6 = st.columns([2, 2, 1, 1, 2, 1])
mc1.metric("Generated",  meta.get("generated_at", "?")[:19].replace("T", " "))
mc2.metric("Ships",      "  ·  ".join(meta.get("ships_tested", [])))
mc3.metric("Scenarios",  str(len(meta.get("scenarios_tested", []))))
mc4.metric("Solvers",    str(len(meta.get("solvers_tested", []))))
mc5.metric("Seeds",      "  ·  ".join(map(str, meta.get("seeds", []))))
if mc6.button("↺ Reload"):
    _load_json.clear()
    st.rerun()

st.divider()

# ── DataFrames ────────────────────────────────────────────────────────────────

df_std = _to_df(data.get("standard", []))
df_tr  = _to_df(data.get("transfer", []))
has_transfer = not df_tr.empty

if df_std.empty:
    st.error("Standard benchmark data is empty.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Scenario Deep-Dive",
    "Transfer Analysis",
    "Raw Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Performance by Solver & Ship")

    sc_options = ["All"] + [s for s in SCENARIO_ORDER if s in df_std["scenario"].unique()]
    sc_sel_ov  = st.selectbox("Filter by scenario", sc_options, key="ov_sc")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_score_heatmap(df_std, sc_sel_ov), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_runtime_scatter(df_std), use_container_width=True)

    st.divider()
    st.subheader("Placement Rate by Scenario")
    st.caption(
        "Shows what fraction of containers were actually placed. "
        "Values < 100 % mean the weight or space cap was binding."
    )
    st.plotly_chart(plot_placement_heatmap(df_std), use_container_width=True)

    if "cog_height_norm" in df_std.columns and _ok(df_std)["cog_height_norm"].gt(0).any():
        st.divider()
        st.subheader("Centre-of-Gravity Height")
        st.caption(
            "Lower CoG = better metacentric stability. "
            "Computed as normalised Gz = Σ(w·tier) / (total_weight × (height−1))."
        )
        st.plotly_chart(plot_cog_bars(df_std), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Scenario Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    sc_options_2 = [s for s in SCENARIO_ORDER if s in df_std["scenario"].unique()]
    if not sc_options_2:
        st.info("No scenario data available.")
    else:
        sc_sel_2 = st.selectbox("Scenario", sc_options_2, key="sc2_sel")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_scenario_score(df_std, sc_sel_2), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_scenario_runtime(df_std, sc_sel_2), use_container_width=True)

        st.plotly_chart(plot_balance_breakdown(df_std, sc_sel_2), use_container_width=True)

        # Placement callout for capacity-constrained scenarios
        if sc_sel_2 in ("weight_limited", "space_limited"):
            st.divider()
            st.subheader("Placement Rate Detail")
            st.caption(
                "These scenarios are capacity-constrained. "
                "The table shows the mean % of containers placed per (solver, ship) combination."
            )
            sub_placed = _ok(df_std[df_std["scenario"] == sc_sel_2])
            pivot_placed = (
                sub_placed
                .groupby(["solver_name", "ship_key"])["pct_placed"]
                .mean()
                .unstack(level="ship_key")
                .rename(index=SOLVER_DISPLAY)
                .reindex(columns=[s for s in SHIP_ORDER if s in sub_placed["ship_key"].unique()])
            )
            st.dataframe(pivot_placed.round(1).style.background_gradient(
                cmap="RdYlGn", axis=None, vmin=0, vmax=100
            ), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Transfer Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if not has_transfer:
        st.info(
            "No transfer results found.\n\n"
            "Re-run the benchmark without `--no-transfer`:\n"
            "```\n"
            "conda run -n personal python benchmark.py\n"
            "```"
        )
    else:
        st.subheader("Cross-Ship Transfer Analysis")
        with st.expander("ℹ What is transfer testing?", expanded=True):
            st.markdown(
                "ML models (**Neural Ranker**, **RL Bayesian**) are trained on a specific ship size. "
                "Transfer tests apply these pre-trained models *unchanged* to ships of **different sizes** — "
                "measuring how well the model generalises.\n\n"
                "| Symbol | Meaning |\n"
                "|--------|--------|\n"
                "| ★ yellow border | In-speciality (model trained on the same ship) |\n"
                "| Δ Score < −0.03 | Out-of-speciality — significant degradation |\n\n"
                "**Why degradation occurs:**\n"
                "- *Neural Ranker*: position features are normalised by training-ship dimensions. "
                "  Applying to a different ship shifts the input distribution.\n"
                "- *RL Bayesian*: manifest feature `n_20ft / ship.length` uses the live ship's length, "
                "  so a panamax model applied to a coastal ship sees out-of-distribution ratios."
            )

        st.subheader("At a Glance — Specific vs General-Model Performance")
        st.caption(
            "Each cluster of bars shows, for a given test ship, how each specific trained model "
            "performs (gold border = in-regime) alongside the general-average score you'd expect "
            "from an arbitrarily chosen model. Larger gaps between the in-regime bar and the others "
            "indicate stronger specialisation."
        )
        fig_ship = plot_ship_perspective(df_tr)
        if fig_ship is not None:
            st.plotly_chart(fig_ship, use_container_width=True)

        st.subheader("Model Fragility — Robustness Across Ship Sizes")
        st.caption(
            "For each trained model, shows how it performs when deployed to every ship size. "
            "A fragile model has a high in-regime bar (gold border) but much lower bars elsewhere. "
            "A robust model keeps all bars close together near the in-regime score."
        )
        fig_frag = plot_model_fragility(df_tr)
        if fig_frag is not None:
            st.plotly_chart(fig_frag, use_container_width=True)

        st.subheader("Full Transfer Results Table")
        regime_df = _regime_summary_table(df_tr)
        if not regime_df.empty:
            st.dataframe(
                regime_df.style.background_gradient(
                    subset=["Δ in-regime"], cmap="RdYlGn_r", vmin=-0.10, vmax=0.01
                ),
                use_container_width=True,
            )
        else:
            st.caption("No transfer data available for regime table.")

        st.divider()

        for solver in ML_SOLVERS:
            if df_tr.empty or solver not in df_tr["solver_name"].values:
                st.caption(f"No transfer data for **{SOLVER_DISPLAY.get(solver, solver)}**.")
                continue

            st.subheader(f"🤖 {SOLVER_DISPLAY.get(solver, solver)} — Score & Degradation Maps")

            fig_sc, fig_deg, significant = plot_transfer_pair(df_tr, solver)
            if fig_sc is None:
                st.caption("No data available.")
                continue

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_sc, use_container_width=True)
                st.caption(
                    "Rows = ship the model is **tested on**. "
                    "Cols = ship the model was **trained on**. "
                    "★ = in-speciality (diagonal)."
                )
            with col_b:
                st.plotly_chart(fig_deg, use_container_width=True)
                st.caption(
                    "Δ Score vs. the matched diagonal. "
                    "Red = worse than in-speciality. Blue = better."
                )

            if significant:
                severe   = [(s, m, sc, d) for s, m, sc, d in significant if d < -0.05]
                moderate = [(s, m, sc, d) for s, m, sc, d in significant if -0.05 <= d < -0.03]
                if severe:
                    items = "\n".join(
                        f"- **{ship}** ship ← **{model}** model: "
                        f"score = {sc:.3f}  (Δ {d:+.3f})"
                        for ship, model, sc, d in severe
                    )
                    st.error(f"**Severe degradation** (Δ < −0.05):\n\n{items}")
                if moderate:
                    items = "\n".join(
                        f"- **{ship}** ship ← **{model}** model: "
                        f"score = {sc:.3f}  (Δ {d:+.3f})"
                        for ship, model, sc, d in moderate
                    )
                    st.warning(f"**Out-of-speciality cases** (−0.05 ≤ Δ < −0.03):\n\n{items}")
            else:
                st.success("No significant cross-ship degradation detected (all Δ ≥ −0.03).")

            st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Raw Results")

    # Combine standard + transfer
    dfs_all = []
    if not df_std.empty:
        tmp = df_std.copy(); tmp["type"] = "standard"; dfs_all.append(tmp)
    if not df_tr.empty:
        tmp = df_tr.copy();  tmp["type"] = "transfer";  dfs_all.append(tmp)
    all_df = pd.concat(dfs_all, ignore_index=True)

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        ship_f = st.multiselect("Ship", SHIP_ORDER, default=SHIP_ORDER, key="raw_ship")
    with fc2:
        sc_f   = st.multiselect(
            "Scenario", SCENARIO_ORDER,
            default=[s for s in SCENARIO_ORDER if s in all_df["scenario"].unique()],
            key="raw_sc",
        )
    with fc3:
        sv_f   = st.multiselect("Solver", SOLVER_ORDER, default=SOLVER_ORDER, key="raw_sv")
    with fc4:
        show_err = st.checkbox("Include errors / skips", value=False, key="raw_err")
        type_opts = all_df["type"].unique().tolist() if "type" in all_df.columns else ["standard"]
        type_f = st.multiselect("Type", type_opts, default=type_opts, key="raw_type")

    mask = (
        all_df["ship_key"].isin(ship_f) &
        all_df["scenario"].isin(sc_f) &
        all_df["solver_name"].isin(sv_f)
    )
    if "type" in all_df.columns:
        mask &= all_df["type"].isin(type_f)
    if not show_err:
        mask &= all_df["error"].isna()

    display_df = all_df[mask].copy()

    # Column order
    cols_ordered = [c for c in [
        "type", "ship_key", "scenario", "solver_name", "model_key", "seed",
        "final_score", "ps_ratio", "fa_ratio", "diag_ratio",
        "pct_placed", "placed", "total",
        "weight_loaded", "cog_height_norm", "runtime_s",
        "is_transfer", "error",
    ] if c in display_df.columns]
    display_df = display_df[cols_ordered].reset_index(drop=True)

    st.caption(f"Showing **{len(display_df)}** rows")
    st.dataframe(display_df.round(4), use_container_width=True, height=520)

    csv = display_df.to_csv(index=False)
    st.download_button(
        "⬇ Download as CSV",
        csv,
        file_name="benchmark_results.csv",
        mime="text/csv",
    )
