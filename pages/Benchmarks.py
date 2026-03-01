"""Benchmark Results Viewer

Four analysis levels:
  Overview    — all ships side-by-side, score + runtime at a glance
  Case Level  — drill into one scenario type (with explicit case definitions)
  Ship Level  — drill into one ship type across all cases
  Flexibility — algorithm consistency and robustness across all conditions

Run benchmarks first:
    conda run -n personal python benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path to benchmark results (always local, next to this file's parent) ───────
_HERE = Path(__file__).resolve().parent          # .../pages/
_ROOT = _HERE.parent                             # project root
RESULTS_PATH = _ROOT / "benchmark_results.json"

# ── Constants ──────────────────────────────────────────────────────────────────

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
SHIP_ORDER = ["coastal", "handymax", "panamax"]
SHIP_DISPLAY = {
    "coastal":  "Coastal  (12×9×5)",
    "handymax": "Handymax (24×11×7)",
    "panamax":  "Panamax  (36×13×9)",
}
SHIP_PROFILE = {
    "coastal":  dict(length=12, beam=9,  height=5, keel=5, max_weight="500 t"),
    "handymax": dict(length=24, beam=11, height=7, keel=6, max_weight="1,500 t"),
    "panamax":  dict(length=36, beam=13, height=9, keel=7, max_weight="3,000 t"),
}
SCENARIO_ORDER = ["balanced", "weight_limited", "space_limited", "mixed"]
ML_SOLVERS = ["neural_ranker", "rl_bayesian"]

CASE_DEFS = {
    "balanced": {
        "icon": "⚖️",
        "title": "Balanced",
        "summary": "Moderate 20 ft + 40 ft mix, uniform weights across the full range.",
        "tests": "Baseline quality: stability, trim, and list when no hard constraint binds.",
        "constraint": "None — well within weight and space limits.",
        "weights": "2,000 – 28,000 kg",
        "mix": "20 ft + 40 ft",
        "dist": "Uniform",
    },
    "weight_limited": {
        "icon": "🏋️",
        "title": "Weight-limited",
        "summary": "Many heavy containers — total manifest weight greatly exceeds the ship's weight cap.",
        "tests": "Weight management: which containers to reject and how to preserve balance while doing so.",
        "constraint": "Weight cap — solver must leave containers ashore.",
        "weights": "18,000 – 30,000 kg",
        "mix": "20 ft + 40 ft (large counts)",
        "dist": "Uniform",
    },
    "space_limited": {
        "icon": "📦",
        "title": "Space-limited",
        "summary": "Many very light 20 ft containers — total count exceeds available hold slots.",
        "tests": "Slot efficiency: how completely the solver fills the physical hold.",
        "constraint": "Hold capacity — more containers than slots.",
        "weights": "100 – 500 kg",
        "mix": "20 ft only",
        "dist": "Uniform",
    },
    "mixed": {
        "icon": "🎲",
        "title": "Mixed",
        "summary": "Bimodal weight distribution with large counts of both container sizes.",
        "tests": "Adaptability: handling unpredictable weight clusters, mirrors real-world manifests.",
        "constraint": "Varies by ship — may hit weight or space.",
        "weights": "500 – 30,000 kg (bimodal)",
        "mix": "20 ft + 40 ft",
        "dist": "Bimodal",
    },
}

SOLVER_COLORS = {
    "greedy":              "#60a5fa",
    "beam_search":         "#34d399",
    "simulated_annealing": "#f59e0b",
    "bayesian_opt":        "#f87171",
    "neural_ranker":       "#a78bfa",
    "rl_bayesian":         "#fb923c",
    "rl_bayesian_sa":      "#e879f9",
}
SHIP_COLORS = {
    "coastal":  "#60a5fa",
    "handymax": "#34d399",
    "panamax":  "#f59e0b",
}
SCENARIO_COLORS = {
    "balanced":       "#60a5fa",
    "weight_limited": "#f87171",
    "space_limited":  "#34d399",
    "mixed":          "#f59e0b",
}

_DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Benchmarks — Cargo Ship Loader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_results(path: str) -> dict | None:
    """Load benchmark_results.json from disk. Returns None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "error" not in df.columns:
        df["error"] = None
    return df


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with no error."""
    if df.empty:
        return df
    return df[df["error"].isna()].copy()


def _filter(df: pd.DataFrame, ships: list, solvers: list) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["ship_key"].isin(ships) & df["solver_name"].isin(solvers)
    return _ok(df[mask])


# ── Shared plot helpers ────────────────────────────────────────────────────────

def _heatmap(z, x, y, text, title, colorscale, zmin, zmax,
             zmid=None, colorbar_title="", height=320) -> go.Figure:
    kw = dict(
        z=z, x=x, y=y,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        text=text, texttemplate="%{text}",
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(thickness=14, len=0.85,
                      title=dict(text=colorbar_title, side="right")),
    )
    if zmid is not None:
        kw["zmid"] = zmid
    fig = go.Figure(go.Heatmap(**kw))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=dict(l=120, r=20, t=50, b=60),
        **_DARK,
    )
    return fig


# ── Overview plots ─────────────────────────────────────────────────────────────

def plot_score_heatmap(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    g = sub.groupby(["solver_name", "ship_key"])["final_score"].mean()
    sv = [s for s in SOLVER_ORDER if s in solvers and s in g.index.get_level_values(0)]
    sh = [s for s in SHIP_ORDER   if s in ships   and s in g.index.get_level_values(1)]
    z, text = [], []
    for solver in sv:
        rz, rt = [], []
        for ship in sh:
            v = g.get((solver, ship), np.nan)
            rz.append(v)
            rt.append(f"{v:.3f}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)
    return _heatmap(
        np.array(z, dtype=float),
        [SHIP_DISPLAY.get(s, s) for s in sh],
        [SOLVER_DISPLAY.get(s, s) for s in sv],
        text,
        "Mean Final Score  (all cases, all seeds)",
        "RdYlGn", 0.88, 1.0,
        colorbar_title="Score",
        height=max(280, 60 + 42 * len(sv)),
    )


def plot_runtime_bar(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "ship_key"])["runtime_s"].mean().reset_index()
    fig = go.Figure()
    for ship in [s for s in SHIP_ORDER if s in ships]:
        rows = agg[agg["ship_key"] == ship].copy()
        if rows.empty:
            continue
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in solvers and s in rows.index]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=SHIP_DISPLAY.get(ship, ship),
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["runtime_s"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.2f}s" for v in rows["runtime_s"]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Mean Runtime by Solver & Ship", font=dict(size=13)),
        yaxis=dict(title="Runtime (s)", type="log"),
        height=340, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_score_vs_runtime(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = (
        sub.groupby(["solver_name", "ship_key"])
        .agg(score=("final_score", "mean"), runtime=("runtime_s", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    for solver in [s for s in SOLVER_ORDER if s in solvers]:
        rows = agg[agg["solver_name"] == solver]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["runtime"], y=rows["score"],
            mode="markers+text",
            name=SOLVER_DISPLAY.get(solver, solver),
            text=[SHIP_DISPLAY.get(s, s)[:7] for s in rows["ship_key"]],
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=12, color=SOLVER_COLORS.get(solver, "#aaa")),
            hovertemplate=(
                f"<b>{SOLVER_DISPLAY.get(solver, solver)}</b><br>"
                "Ship: %{text}<br>Runtime: %{x:.2f} s<br>Score: %{y:.4f}<extra></extra>"
            ),
        ))
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7,
                  annotation_text="0.92", annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_xaxes(type="log", title="Mean Runtime (s) — log scale")
    fig.update_yaxes(title="Mean Final Score", range=[0.84, 1.01])
    fig.update_layout(
        title=dict(text="Quality vs Runtime Tradeoff", font=dict(size=13)),
        height=340, margin=dict(l=60, r=20, t=50, b=55),
        legend=dict(orientation="v", x=1.02, font=dict(size=10)),
        **_DARK,
    )
    return fig


# ── Case Level plots ───────────────────────────────────────────────────────────

def plot_case_scores(df: pd.DataFrame, scenario: str, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df[df["scenario"] == scenario], ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "ship_key"])["final_score"].mean().reset_index()
    fig = go.Figure()
    for ship in [s for s in SHIP_ORDER if s in ships]:
        rows = agg[agg["ship_key"] == ship].copy()
        if rows.empty:
            continue
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in solvers and s in rows.index]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=SHIP_DISPLAY.get(ship, ship),
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["final_score"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.3f}" for v in rows["final_score"]],
            textposition="outside",
        ))
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7)
    fig.update_layout(
        barmode="group",
        title=dict(text="Final Score by Solver", font=dict(size=13)),
        yaxis=dict(title="Final Score", range=[0.5, 1.08]),
        height=340, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_case_runtime(df: pd.DataFrame, scenario: str, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df[df["scenario"] == scenario], ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "ship_key"])["runtime_s"].mean().reset_index()
    fig = go.Figure()
    for ship in [s for s in SHIP_ORDER if s in ships]:
        rows = agg[agg["ship_key"] == ship].copy()
        if rows.empty:
            continue
        rows = rows.set_index("solver_name").reindex(
            [s for s in SOLVER_ORDER if s in solvers and s in rows.index]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=SHIP_DISPLAY.get(ship, ship),
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["runtime_s"],
            marker_color=SHIP_COLORS.get(ship, "#aaa"),
            text=[f"{v:.2f}s" for v in rows["runtime_s"]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Runtime by Solver", font=dict(size=13)),
        yaxis=dict(title="Runtime (s)", type="log"),
        height=340, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_case_balance(df: pd.DataFrame, scenario: str, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df[df["scenario"] == scenario], ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby("solver_name")[["ps_ratio", "fa_ratio", "diag_ratio"]].mean().reset_index()
    sv = [s for s in SOLVER_ORDER if s in solvers and s in agg["solver_name"].values]
    agg = agg.set_index("solver_name").reindex(sv).reset_index()
    metrics = {"PS ratio": "ps_ratio", "FA ratio": "fa_ratio", "Diag ratio": "diag_ratio"}
    colors = ["#60a5fa", "#34d399", "#a78bfa"]
    fig = go.Figure()
    for (label, col), color in zip(metrics.items(), colors):
        fig.add_trace(go.Bar(
            name=label,
            x=[SOLVER_DISPLAY.get(s, s) for s in agg["solver_name"]],
            y=agg[col],
            marker_color=color,
            text=[f"{v:.3f}" if not pd.isna(v) else "—" for v in agg[col]],
            textposition="outside",
        ))
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7,
                  annotation_text="0.92", annotation_font_color="#f59e0b")
    fig.update_layout(
        barmode="group",
        title=dict(text="Balance Ratios (mean across selected ships & seeds)", font=dict(size=13)),
        yaxis=dict(title="Ratio", range=[0.7, 1.08]),
        height=340, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


# ── Ship Level plots ───────────────────────────────────────────────────────────

def plot_ship_scores_by_case(df: pd.DataFrame, ship: str, solvers: list) -> go.Figure:
    sub = _filter(df[df["ship_key"] == ship], [ship], solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "scenario"])["final_score"].mean().reset_index()
    fig = go.Figure()
    for sc in [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]:
        rows = agg[agg["scenario"] == sc].copy()
        sv = [s for s in SOLVER_ORDER if s in solvers and s in rows["solver_name"].values]
        rows = rows.set_index("solver_name").reindex(sv).reset_index()
        d = CASE_DEFS.get(sc, {})
        fig.add_trace(go.Bar(
            name=f"{d.get('icon', '')} {d.get('title', sc)}",
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["final_score"],
            marker_color=SCENARIO_COLORS.get(sc, "#aaa"),
            text=[f"{v:.3f}" if not pd.isna(v) else "—" for v in rows["final_score"]],
            textposition="outside",
        ))
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7)
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Final Score by Case — {SHIP_DISPLAY.get(ship, ship)}", font=dict(size=13)),
        yaxis=dict(title="Final Score", range=[0.5, 1.08]),
        height=360, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_ship_runtime_by_case(df: pd.DataFrame, ship: str, solvers: list) -> go.Figure:
    sub = _filter(df[df["ship_key"] == ship], [ship], solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "scenario"])["runtime_s"].mean().reset_index()
    fig = go.Figure()
    for sc in [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]:
        rows = agg[agg["scenario"] == sc].copy()
        sv = [s for s in SOLVER_ORDER if s in solvers and s in rows["solver_name"].values]
        rows = rows.set_index("solver_name").reindex(sv).reset_index()
        d = CASE_DEFS.get(sc, {})
        fig.add_trace(go.Bar(
            name=f"{d.get('icon', '')} {d.get('title', sc)}",
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["runtime_s"],
            marker_color=SCENARIO_COLORS.get(sc, "#aaa"),
            text=[f"{v:.2f}s" if not pd.isna(v) else "—" for v in rows["runtime_s"]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Runtime by Case — {SHIP_DISPLAY.get(ship, ship)}", font=dict(size=13)),
        yaxis=dict(title="Runtime (s)", type="log"),
        height=360, margin=dict(l=60, r=20, t=50, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def plot_ship_seed_variance(df: pd.DataFrame, ship: str, solvers: list) -> go.Figure:
    sub = _filter(df[df["ship_key"] == ship], [ship], solvers)
    if sub.empty:
        return go.Figure()
    agg = sub.groupby("solver_name")["final_score"].agg(
        mean="mean", lo="min", hi="max"
    ).reset_index()
    sv = [s for s in SOLVER_ORDER if s in solvers and s in agg["solver_name"].values]
    agg = agg.set_index("solver_name").reindex(sv).reset_index()
    x_labels = [SOLVER_DISPLAY.get(s, s) for s in agg["solver_name"]]
    colors = [SOLVER_COLORS.get(s, "#aaa") for s in agg["solver_name"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels, y=agg["mean"],
        marker_color=colors,
        text=[f"{v:.4f}" if not pd.isna(v) else "—" for v in agg["mean"]],
        textposition="outside",
        error_y=dict(
            type="data", symmetric=False,
            array=list(agg["hi"] - agg["mean"]),
            arrayminus=list(agg["mean"] - agg["lo"]),
            color="#94a3b8", thickness=2, width=6,
        ),
        name="Mean ± seed range",
    ))
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.7)
    fig.update_layout(
        title=dict(
            text=f"Score Consistency (error bars = min/max across seeds) — {SHIP_DISPLAY.get(ship, ship)}",
            font=dict(size=13),
        ),
        yaxis=dict(title="Final Score", range=[0.5, 1.08]),
        height=340, margin=dict(l=60, r=20, t=55, b=65),
        showlegend=False, **_DARK,
    )
    return fig


def plot_ship_cog(df: pd.DataFrame, ship: str, solvers: list) -> go.Figure:
    sub = _filter(df[df["ship_key"] == ship], [ship], solvers)
    if sub.empty or "cog_height_norm" not in sub.columns:
        return go.Figure()
    agg = (sub[sub["cog_height_norm"] > 0]
           .groupby("solver_name")["cog_height_norm"].mean().reset_index())
    sv = [s for s in SOLVER_ORDER if s in solvers and s in agg["solver_name"].values]
    agg = agg.set_index("solver_name").reindex(sv).reset_index()
    fig = go.Figure(go.Bar(
        x=[SOLVER_DISPLAY.get(s, s) for s in agg["solver_name"]],
        y=agg["cog_height_norm"],
        marker_color=[SOLVER_COLORS.get(s, "#aaa") for s in agg["solver_name"]],
        text=[f"{v:.4f}" if not pd.isna(v) else "—" for v in agg["cog_height_norm"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(
            text=f"Centre-of-Gravity Height (lower = more stable) — {SHIP_DISPLAY.get(ship, ship)}",
            font=dict(size=13),
        ),
        yaxis=dict(title="Normalised CoG height"),
        height=310, margin=dict(l=60, r=20, t=50, b=65),
        showlegend=False, **_DARK,
    )
    return fig


# ── Flexibility plots ──────────────────────────────────────────────────────────

def _flex_stats(df: pd.DataFrame, ships: list, solvers: list) -> pd.DataFrame:
    sub = _filter(df, ships, solvers)
    rows = []
    for solver in [s for s in SOLVER_ORDER if s in solvers]:
        vals = sub[sub["solver_name"] == solver]["final_score"].dropna()
        rt   = sub[sub["solver_name"] == solver]["runtime_s"].dropna()
        if vals.empty:
            continue
        rows.append({
            "solver":     solver,
            "display":    SOLVER_DISPLAY.get(solver, solver),
            "mean":       float(vals.mean()),
            "std":        float(vals.std()),
            "min":        float(vals.min()),
            "max":        float(vals.max()),
            "range":      float(vals.max() - vals.min()),
            "flex_score": float(vals.mean() - 2 * vals.std()),
            "mean_rt":    float(rt.mean()) if not rt.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_flexibility_table(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return go.Figure()
    cols   = ["display", "mean", "std", "min", "range", "flex_score", "mean_rt"]
    labels = ["Solver", "Mean score", "Std dev", "Worst case", "Range", "Flex score†", "Mean RT (s)"]

    def fmt(col, val):
        if pd.isna(val):
            return "—"
        if col in ("mean", "std", "min", "range", "flex_score"):
            return f"{val:.4f}"
        if col == "mean_rt":
            return f"{val:.2f} s"
        return str(val)

    cell_vals = [[fmt(col, v) for v in stats[col]] for col in cols]

    fv = stats["flex_score"].tolist()
    fmin, fmax = min(fv), max(fv)
    frange = fmax - fmin or 1.0
    fill_colors = []
    for col in cols:
        if col == "flex_score":
            colors = []
            for v in fv:
                t = (v - fmin) / frange
                r = int(239 - t * (239 - 52))
                g = int(68  + t * (211 - 68))
                b = int(68  + t * (99  - 68))
                colors.append(f"rgba({r},{g},{b},0.35)")
            fill_colors.append(colors)
        else:
            fill_colors.append(["rgba(30,41,59,0.8)"] * len(stats))

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{l}</b>" for l in labels],
            fill_color="#334155",
            font=dict(color="#e2e8f0", size=12),
            align="center", height=32,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=fill_colors,
            font=dict(color="#e2e8f0", size=11),
            align="center", height=28,
        ),
    ))
    fig.update_layout(
        title=dict(text="Flexibility Summary  († = mean − 2×std, higher is better)", font=dict(size=13)),
        height=80 + 30 * len(stats),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0f172a", template="plotly_dark",
    )
    return fig


def plot_combo_heatmap(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    sh = [s for s in SHIP_ORDER     if s in ships]
    sc = [s for s in SCENARIO_ORDER if s in sub["scenario"].unique()]
    sv = [s for s in SOLVER_ORDER   if s in solvers and s in sub["solver_name"].unique()]
    x_labels = [
        f"{ship[:3].title()} / {CASE_DEFS.get(scenario, {}).get('icon', '')} "
        f"{CASE_DEFS.get(scenario, {}).get('title', scenario)[:6]}"
        for ship in sh for scenario in sc
    ]
    grouped = sub.groupby(["solver_name", "ship_key", "scenario"])["final_score"].mean()
    z, text = [], []
    for solver in sv:
        rz, rt = [], []
        for ship in sh:
            for scenario in sc:
                v = grouped.get((solver, ship, scenario), np.nan)
                rz.append(v)
                rt.append(f"{v:.3f}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)
    return _heatmap(
        np.array(z, dtype=float), x_labels,
        [SOLVER_DISPLAY.get(s, s) for s in sv],
        text,
        "Final Score — Every (Ship × Case) Combination",
        "RdYlGn", 0.85, 1.0,
        colorbar_title="Score",
        height=max(300, 60 + 42 * len(sv)),
    )


def plot_radar(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    sc_present = [s for s in SCENARIO_ORDER if s in sub["scenario"].unique()]
    if len(sc_present) < 3:
        return go.Figure()
    labels = [f"{CASE_DEFS[s]['icon']} {CASE_DEFS[s]['title']}" for s in sc_present]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    for solver in [s for s in SOLVER_ORDER if s in solvers]:
        agg = sub[sub["solver_name"] == solver].groupby("scenario")["final_score"].mean()
        vals = [float(agg.get(sc, np.nan)) for sc in sc_present]
        if all(np.isnan(v) for v in vals):
            continue
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            name=SOLVER_DISPLAY.get(solver, solver),
            line=dict(color=SOLVER_COLORS.get(solver, "#aaa"), width=2),
            fill="toself",
            fillcolor=SOLVER_COLORS.get(solver, "#aaa"),
            opacity=0.10,
        ))
    fig.update_layout(
        title=dict(text="Solver Profiles Across Case Types", font=dict(size=13)),
        polar=dict(
            radialaxis=dict(range=[0.80, 1.0], showticklabels=True,
                            tickfont=dict(size=9), gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
            bgcolor="#1e293b",
        ),
        height=420, margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="v", x=1.05, font=dict(size=10)),
        paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
        template="plotly_dark",
    )
    return fig


def plot_runtime_vs_score_scatter(df: pd.DataFrame, ships: list, solvers: list) -> go.Figure:
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    agg = (
        sub.groupby(["solver_name", "ship_key", "scenario"])
        .agg(score=("final_score", "mean"), rt=("runtime_s", "mean"))
        .reset_index()
    )
    shape_map = {"coastal": "circle", "handymax": "square", "panamax": "diamond"}
    fig = go.Figure()
    for solver in [s for s in SOLVER_ORDER if s in solvers]:
        rows = agg[agg["solver_name"] == solver]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["rt"], y=rows["score"],
            mode="markers",
            name=SOLVER_DISPLAY.get(solver, solver),
            marker=dict(
                size=10,
                color=[SCENARIO_COLORS.get(sc, "#aaa") for sc in rows["scenario"]],
                symbol=[shape_map.get(sh, "circle") for sh in rows["ship_key"]],
                line=dict(color=SOLVER_COLORS.get(solver, "#aaa"), width=2),
            ),
            hovertemplate=(
                f"<b>{SOLVER_DISPLAY.get(solver, solver)}</b><br>"
                "Ship: %{customdata[0]}<br>Case: %{customdata[1]}<br>"
                "Runtime: %{x:.2f} s<br>Score: %{y:.4f}<extra></extra>"
            ),
            customdata=list(zip(rows["ship_key"], rows["scenario"])),
        ))
    fig.update_xaxes(type="log", title="Mean Runtime (s) — log scale")
    fig.update_yaxes(title="Mean Final Score", range=[0.84, 1.01])
    fig.update_layout(
        title=dict(
            text="Score vs Runtime — Colour=Case, Shape=Ship, Border=Solver",
            font=dict(size=13),
        ),
        height=400, margin=dict(l=60, r=20, t=50, b=55),
        legend=dict(orientation="v", x=1.02, font=dict(size=10)),
        **_DARK,
    )
    return fig


# ── Transfer Analysis (ML solvers) ────────────────────────────────────────────

def plot_transfer_pair(df_tr: pd.DataFrame, solver_name: str) -> tuple:
    """Return (score_heatmap, delta_heatmap, significant_list) for one ML solver."""
    sub = _ok(df_tr[df_tr["solver_name"] == solver_name])
    if sub.empty:
        return go.Figure(), go.Figure(), []
    grouped = sub.groupby(["ship_key", "model_key"])["final_score"].mean()
    ships_present = [s for s in SHIP_ORDER if s in sub["ship_key"].unique()]
    models_present = [s for s in SHIP_ORDER if s in sub["model_key"].unique()]

    z, text = [], []
    for ship in ships_present:
        rz, rt = [], []
        for model in models_present:
            v = grouped.get((ship, model), np.nan)
            rz.append(v)
            star = " ★" if ship == model else ""
            rt.append(f"{v:.3f}{star}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)

    fig_score = _heatmap(
        np.array(z, dtype=float),
        [SHIP_DISPLAY.get(s, s) for s in models_present],
        [SHIP_DISPLAY.get(s, s) for s in ships_present],
        text,
        f"{SOLVER_DISPLAY.get(solver_name, solver_name)} — Score (ship tested × model trained)",
        "RdYlGn", 0.88, 1.0, colorbar_title="Score", height=310,
    )
    fig_score.update_xaxes(title_text="Model trained on →")
    fig_score.update_yaxes(title_text="Ship tested on →")

    dz, dtext, significant = [], [], []
    for ship in ships_present:
        matched = grouped.get((ship, ship), np.nan)
        rdz, rdt = [], []
        for model in models_present:
            v = grouped.get((ship, model), np.nan)
            if np.isnan(v) or np.isnan(matched):
                rdz.append(np.nan); rdt.append("—")
            else:
                delta = v - matched
                rdz.append(delta)
                rdt.append(f"{delta:+.3f}")
                if ship != model and delta < -0.03:
                    significant.append((ship, model, float(v), float(delta)))
        dz.append(rdz); dtext.append(rdt)

    dz_arr = np.array(dz, dtype=float)
    bound = max(abs(float(np.nanmin(dz_arr))), abs(float(np.nanmax(dz_arr))), 0.02)
    fig_deg = _heatmap(
        dz_arr,
        [SHIP_DISPLAY.get(s, s) for s in models_present],
        [SHIP_DISPLAY.get(s, s) for s in ships_present],
        dtext,
        f"{SOLVER_DISPLAY.get(solver_name, solver_name)} — Score Δ vs. in-speciality diagonal",
        "RdBu", zmin=-bound, zmax=bound, zmid=0.0,
        colorbar_title="Δ Score", height=310,
    )
    fig_deg.update_xaxes(title_text="Model trained on →")
    fig_deg.update_yaxes(title_text="Ship tested on →")
    return fig_score, fig_deg, sorted(significant, key=lambda x: x[3])


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════

# ── Load data ──────────────────────────────────────────────────────────────────
data = load_results(str(RESULTS_PATH))

if data is None:
    st.title("📊 Benchmark Results")
    st.error(
        f"**benchmark_results.json not found.**\n\n"
        f"Looking for: `{RESULTS_PATH}`\n\n"
        "Generate results with:\n"
        "```\nconda run -n personal python benchmark.py\n```\n\n"
        "Quick test (~1 min):\n"
        "```\nconda run -n personal python benchmark.py "
        "--ships coastal --scenarios balanced --no-transfer\n```"
    )
    st.stop()

df_std = _df(data.get("standard", []))
df_tr  = _df(data.get("transfer", []))
has_transfer = not df_tr.empty

if df_std.empty:
    st.error("benchmark_results.json was loaded but contains no standard results.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Benchmarks")
    meta = data.get("metadata", {})
    st.caption(
        f"**Source:** `{RESULTS_PATH.name}`  \n"
        f"**Run:** {meta.get('generated_at', '?')[:19].replace('T', ' ')}  \n"
        f"**Records:** {len(df_std)} standard"
        + (f" · {len(df_tr)} transfer" if has_transfer else "")
    )

    st.divider()
    st.subheader("Filters")

    ships_available   = [s for s in SHIP_ORDER   if s in df_std["ship_key"].unique()]
    solvers_available = [s for s in SOLVER_ORDER if s in df_std["solver_name"].unique()]

    sel_ships = st.multiselect(
        "Ships", ships_available,
        default=ships_available,
        format_func=lambda s: SHIP_DISPLAY.get(s, s),
        key="g_ships",
    )
    sel_solvers = st.multiselect(
        "Solvers", solvers_available,
        default=solvers_available,
        format_func=lambda s: SOLVER_DISPLAY.get(s, s),
        key="g_solvers",
    )

    if not sel_ships or not sel_solvers:
        st.warning("Select at least one ship and one solver.")
        st.stop()

    st.divider()
    st.caption(
        f"**{len(meta.get('seeds', []))} seeds** · "
        f"**{len(meta.get('scenarios_tested', []))} cases** · "
        f"**{len(meta.get('solvers_tested', []))} solvers**"
    )
    if st.button("↺ Reload data"):
        load_results.clear()
        st.rerun()

    st.divider()
    st.subheader("Case definitions")
    for sc, d in CASE_DEFS.items():
        if sc in df_std["scenario"].unique():
            with st.expander(f"{d['icon']} {d['title']}"):
                st.caption(d["summary"])
                st.markdown(
                    f"**Tests:** {d['tests']}  \n"
                    f"**Constraint:** {d['constraint']}  \n"
                    f"**Weights:** {d['weights']}  \n"
                    f"**Mix:** {d['mix']}  \n"
                    f"**Distribution:** {d['dist']}"
                )

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_ov, tab_case, tab_ship, tab_flex, tab_raw = st.tabs([
    "🗺 Overview",
    "📋 Case Level",
    "🚢 Ship Level",
    "🔀 Flexibility",
    "🗄 Raw Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_ov:
    st.subheader("Overview — All Ships & Cases")
    st.caption(
        "High-level snapshot across every selected ship, case, and seed. "
        "Use the sidebar to filter ships and solvers. Drill into specific cases "
        "or ships using the **Case Level** and **Ship Level** tabs."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_score_heatmap(df_std, sel_ships, sel_solvers),
            use_container_width=True,
        )
        st.caption(
            "Each cell = mean final score averaged over **all cases and all seeds** "
            "for that (solver, ship) pair."
        )
    with col_b:
        st.plotly_chart(
            plot_runtime_bar(df_std, sel_ships, sel_solvers),
            use_container_width=True,
        )
        st.caption(
            "Mean runtime per (solver, ship). Log scale — notice how runtimes "
            "scale with ship size for iterative solvers."
        )

    st.divider()
    st.plotly_chart(
        plot_score_vs_runtime(df_std, sel_ships, sel_solvers),
        use_container_width=True,
    )
    st.caption(
        "Each point = one (solver, ship) mean. "
        "Ideal solvers sit **top-left** (fast and accurate)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Case Level
# ══════════════════════════════════════════════════════════════════════════════

with tab_case:
    cases_present = [s for s in SCENARIO_ORDER if s in df_std["scenario"].unique()]
    if not cases_present:
        st.info("No case data available.")
    else:
        sel_case = st.radio(
            "Select case",
            cases_present,
            format_func=lambda s: f"{CASE_DEFS[s]['icon']}  {CASE_DEFS[s]['title']}",
            horizontal=True,
            key="case_sel",
        )
        d = CASE_DEFS.get(sel_case, {})
        st.info(
            f"**{d.get('icon', '')} {d.get('title', sel_case)} — What this case tests:**  \n"
            f"{d.get('tests', '')}  \n\n"
            f"| Binding constraint | Weight range | Container mix | Distribution |  \n"
            f"|---|---|---|---|  \n"
            f"| {d.get('constraint', '—')} | {d.get('weights', '—')} "
            f"| {d.get('mix', '—')} | {d.get('dist', '—')} |"
        )
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_case_scores(df_std, sel_case, sel_ships, sel_solvers),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_case_runtime(df_std, sel_case, sel_ships, sel_solvers),
                use_container_width=True,
            )
        st.plotly_chart(
            plot_case_balance(df_std, sel_case, sel_ships, sel_solvers),
            use_container_width=True,
        )
        if sel_case in ("weight_limited", "space_limited"):
            st.divider()
            st.subheader("Placement Rate Detail")
            st.caption(
                "This is a capacity-constrained case. "
                "Values < 100 % mean containers were left ashore."
            )
            sub_p = _filter(df_std[df_std["scenario"] == sel_case], sel_ships, sel_solvers)
            if not sub_p.empty:
                pivot = (
                    sub_p.groupby(["solver_name", "ship_key"])["pct_placed"]
                    .mean()
                    .unstack("ship_key")
                    .rename(index=SOLVER_DISPLAY)
                    .reindex(columns=[s for s in sel_ships if s in sub_p["ship_key"].unique()])
                    .rename(columns=SHIP_DISPLAY)
                )
                if not pivot.empty:
                    st.dataframe(
                        pivot.round(1).style.background_gradient(
                            cmap="RdYlGn", axis=None, vmin=0, vmax=100
                        ),
                        use_container_width=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# Ship Level
# ══════════════════════════════════════════════════════════════════════════════

with tab_ship:
    ships_in_filter = [s for s in SHIP_ORDER if s in sel_ships and s in df_std["ship_key"].unique()]
    if not ships_in_filter:
        st.info("No ships available for the current filter.")
    else:
        sel_ship = st.radio(
            "Select ship",
            ships_in_filter,
            format_func=lambda s: SHIP_DISPLAY.get(s, s),
            horizontal=True,
            key="ship_sel",
        )
        prof = SHIP_PROFILE.get(sel_ship, {})
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Length",     f"{prof.get('length',     '?')} bays")
        pc2.metric("Beam",       f"{prof.get('beam',       '?')} cols")
        pc3.metric("Height",     f"{prof.get('height',     '?')} tiers")
        pc4.metric("Keel width", f"{prof.get('keel',       '?')} cols")
        pc5.metric("Max weight",  prof.get("max_weight",   "?"))
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_ship_scores_by_case(df_std, sel_ship, sel_solvers),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_ship_runtime_by_case(df_std, sel_ship, sel_solvers),
                use_container_width=True,
            )
        st.plotly_chart(
            plot_ship_seed_variance(df_std, sel_ship, sel_solvers),
            use_container_width=True,
        )
        st.caption(
            "Error bars show min and max score across seeds. "
            "Tall bars = algorithm quality depends heavily on the random seed."
        )
        if "cog_height_norm" in df_std.columns:
            st.divider()
            st.plotly_chart(
                plot_ship_cog(df_std, sel_ship, sel_solvers),
                use_container_width=True,
            )
            st.caption("Lower CoG = better metacentric stability.")

# ══════════════════════════════════════════════════════════════════════════════
# Flexibility
# ══════════════════════════════════════════════════════════════════════════════

with tab_flex:
    st.subheader("Algorithm Flexibility & Robustness")
    with st.expander("ℹ️ What is flexibility?", expanded=True):
        st.markdown(
            "**Flexibility** measures how consistently an algorithm performs "
            "across **all** ship sizes and loading cases — not just its best-case scenario.\n\n"
            "| Metric | Definition |\n"
            "|--------|------------|\n"
            "| **Mean score** | Average final score across all (ship, case, seed) combinations |\n"
            "| **Std dev** | How much scores vary — lower = more predictable |\n"
            "| **Worst case** | Minimum score seen across all conditions |\n"
            "| **Range** | max − min score — lower = more consistent |\n"
            "| **Flex score†** | mean − 2×std — rewards high average, penalises variance |\n\n"
            "A **flexible** algorithm has a high flex score: it works well everywhere, "
            "not just in ideal conditions."
        )

    stats = _flex_stats(df_std, sel_ships, sel_solvers)
    if not stats.empty:
        st.plotly_chart(plot_flexibility_table(stats), use_container_width=True)

    st.divider()
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(
            plot_combo_heatmap(df_std, sel_ships, sel_solvers),
            use_container_width=True,
        )
        st.caption(
            "Every (ship × case) combination as a separate column. "
            "Horizontal uniform colour = consistent algorithm. "
            "Patchy rows = performance depends on the specific condition."
        )
    with col_b:
        radar = plot_radar(df_std, sel_ships, sel_solvers)
        if radar.data:
            st.plotly_chart(radar, use_container_width=True)
            st.caption(
                "Spider chart: each axis = one case type. "
                "Large, even polygon = flexible. Lopsided = specialist."
            )

    st.divider()
    st.plotly_chart(
        plot_runtime_vs_score_scatter(df_std, sel_ships, sel_solvers),
        use_container_width=True,
    )
    st.caption(
        "Colour = case type · Shape = ship (circle/square/diamond) · Border = solver. "
        "Vertical spread per solver = sensitivity to case type."
    )

    if has_transfer:
        st.divider()
        st.subheader("ML Solver Transfer Flexibility")
        with st.expander("ℹ️ What is transfer testing?", expanded=False):
            st.markdown(
                "ML solvers (**Neural Ranker**, **RL Bayesian**) are trained on a specific ship. "
                "Transfer tests apply them *unchanged* to ships of different sizes, measuring "
                "out-of-distribution generalisation.\n\n"
                "| Symbol | Meaning |\n"
                "|--------|--------|\n"
                "| ★ yellow border | In-speciality (model tested on the ship it was trained for) |\n"
                "| Δ < −0.03 | Significant degradation out-of-speciality |"
            )
        for solver in ML_SOLVERS:
            if solver not in df_tr["solver_name"].values:
                st.caption(f"No transfer data for **{SOLVER_DISPLAY.get(solver, solver)}**.")
                continue
            st.markdown(f"#### {SOLVER_DISPLAY.get(solver, solver)}")
            fig_s, fig_d, sigs = plot_transfer_pair(df_tr, solver)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_s, use_container_width=True)
            with col_b:
                st.plotly_chart(fig_d, use_container_width=True)
            if sigs:
                st.warning(
                    "**Out-of-speciality degradation (Δ < −0.03):**  \n"
                    + "  \n".join(
                        f"- Tested on **{SHIP_DISPLAY.get(sh, sh)}** with model trained on "
                        f"**{SHIP_DISPLAY.get(mo, mo)}**: "
                        f"score {sc:.3f}  (Δ {dl:+.3f})"
                        for sh, mo, sc, dl in sigs
                    )
                )

# ══════════════════════════════════════════════════════════════════════════════
# Raw Data
# ══════════════════════════════════════════════════════════════════════════════

with tab_raw:
    st.subheader("Raw Data")
    sub_raw = _filter(df_std, sel_ships, sel_solvers)

    # Display columns
    show_cols = [
        "ship_key", "scenario", "solver_name", "seed",
        "final_score", "ps_ratio", "fa_ratio", "diag_ratio",
        "pct_placed", "runtime_s", "weight_loaded", "cog_height_norm",
    ]
    show_cols = [c for c in show_cols if c in sub_raw.columns]

    col_filters = st.columns(3)
    with col_filters[0]:
        sc_filter = st.multiselect(
            "Scenarios", SCENARIO_ORDER,
            default=[s for s in SCENARIO_ORDER if s in sub_raw["scenario"].unique()],
            format_func=lambda s: CASE_DEFS.get(s, {}).get("title", s),
            key="raw_sc",
        )
    if sc_filter:
        sub_raw = sub_raw[sub_raw["scenario"].isin(sc_filter)]

    st.dataframe(
        sub_raw[show_cols].sort_values(["ship_key", "scenario", "solver_name", "seed"])
        .reset_index(drop=True)
        .style.background_gradient(subset=["final_score"], cmap="RdYlGn", vmin=0.85, vmax=1.0),
        use_container_width=True,
        height=420,
    )
    st.caption(f"{len(sub_raw)} rows shown.")

    csv = sub_raw[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv,
        "benchmark_results_filtered.csv",
        "text/csv",
    )
