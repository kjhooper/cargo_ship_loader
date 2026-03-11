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
SCENARIO_ORDER = [
    "balanced", "weight_limited", "space_limited", "mixed",
    "stop_early_heavy", "stop_early_light", "stop_uniform_3", "stop_many",
]
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
    "stop_early_heavy": {
        "icon": "🔴",
        "title": "Early Stop, Heavy",
        "summary": "2 stops. First-stop containers are heavy (20–28 t); second-stop are light (2–8 t).",
        "tests": "Worst case: weight-first sorting buries heavy first-stop containers at the bottom — they must come off first but are hardest to reach.",
        "constraint": "Unloading order conflict with weight-optimal loading.",
        "weights": "2,000 – 28,000 kg (bimodal by stop)",
        "mix": "20 ft + 40 ft",
        "dist": "Stop-stratified",
    },
    "stop_early_light": {
        "icon": "🟢",
        "title": "Early Stop, Light",
        "summary": "2 stops. First-stop containers are light (2–8 t); second-stop are heavy (20–28 t).",
        "tests": "Natural alignment: weight-first sorting places heavy later-stop containers deep; light early-stop ones float to the top automatically.",
        "constraint": "Unloading order naturally aligned with weight-optimal loading.",
        "weights": "2,000 – 28,000 kg (bimodal by stop)",
        "mix": "20 ft + 40 ft",
        "dist": "Stop-stratified",
    },
    "stop_uniform_3": {
        "icon": "🟡",
        "title": "Uniform 3 Stops",
        "summary": "3 port stops, uniform weight distribution across all containers.",
        "tests": "Moderate complexity: no weight bias — solver must rely on sort order and unloading penalty.",
        "constraint": "Unloading order with no weight shortcut.",
        "weights": "2,000 – 28,000 kg (uniform)",
        "mix": "20 ft + 40 ft",
        "dist": "Uniform",
    },
    "stop_many": {
        "icon": "🟠",
        "title": "Many Stops (5)",
        "summary": "5 port stops, uniform weight distribution.",
        "tests": "High complexity: five destination buckets multiply ordering conflicts.",
        "constraint": "Unloading order, high-complexity 5-way interleaving.",
        "weights": "2,000 – 28,000 kg (uniform)",
        "mix": "20 ft + 40 ft",
        "dist": "Uniform",
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
    "balanced":         "#60a5fa",
    "weight_limited":   "#f87171",
    "space_limited":    "#34d399",
    "mixed":            "#f59e0b",
    "stop_early_heavy": "#fb7185",
    "stop_early_light": "#4ade80",
    "stop_uniform_3":   "#fbbf24",
    "stop_many":        "#c084fc",
}

UNLOADING_SCENARIO_ORDER = [
    "stop_early_heavy", "stop_early_light", "stop_uniform_3", "stop_many",
]
UNLOADING_CASE_DEFS = {
    "stop_early_heavy": {
        "icon": "🔴",
        "title": "Early Stop, Heavy",
        "summary": "First-stop containers are heavy (20–28 t); second-stop containers are light (2–8 t).",
        "tests": "Worst case: weight-first sorting buries heavy first-stop containers at the bottom — they must come off first but are hardest to reach.",
        "stops": 2, "stop1_kg": "20,000–28,000", "stop2_kg": "2,000–8,000",
    },
    "stop_early_light": {
        "icon": "🟢",
        "title": "Early Stop, Light",
        "summary": "First-stop containers are light (2–8 t); second-stop containers are heavy (20–28 t).",
        "tests": "Natural alignment: weight-first sorting places heavy later-stop containers deep; light early-stop ones float to the top automatically.",
        "stops": 2, "stop1_kg": "2,000–8,000", "stop2_kg": "20,000–28,000",
    },
    "stop_uniform_3": {
        "icon": "🟡",
        "title": "Uniform 3 Stops",
        "summary": "3 port stops, uniform weight distribution across all containers.",
        "tests": "Moderate complexity: no weight bias to exploit — solver must rely on sort order and the unloading penalty to achieve good stacking.",
        "stops": 3, "stop1_kg": "2,000–28,000 (all)", "stop2_kg": "—",
    },
    "stop_many": {
        "icon": "🟠",
        "title": "Many Stops (5)",
        "summary": "5 port stops, uniform weight distribution.",
        "tests": "High complexity: five destination buckets multiply ordering conflicts and make rehandling harder to avoid.",
        "stops": 5, "stop1_kg": "2,000–28,000 (all)", "stop2_kg": "—",
    },
}
UNLOADING_SCENARIO_COLORS = {
    "stop_early_heavy": "#f87171",
    "stop_early_light": "#34d399",
    "stop_uniform_3":   "#f59e0b",
    "stop_many":        "#a78bfa",
}

_DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")

# Shared legend styling helpers
_LEG_H = dict(   # horizontal, anchored above-left of chart
    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    bgcolor="rgba(30,41,59,0.92)", bordercolor="#475569", borderwidth=1,
    font=dict(size=11, color="#e2e8f0"),
    title_font=dict(size=10, color="#94a3b8"),
)
_LEG_V = dict(   # vertical, right of chart
    orientation="v", x=1.02, xanchor="left", yanchor="middle", y=0.5,
    bgcolor="rgba(30,41,59,0.92)", bordercolor="#475569", borderwidth=1,
    font=dict(size=11, color="#e2e8f0"),
    title_font=dict(size=10, color="#94a3b8"),
)

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
            hovertemplate=(
                f"<b>{SHIP_DISPLAY.get(ship, ship)}</b><br>"
                "%{x}: %{y:.3g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Mean Runtime by Solver & Ship", font=dict(size=13)),
        yaxis=dict(
            title="Runtime (s)",
            type="log",
            tickformat=".3g",
            gridcolor="#334155",
        ),
        height=340, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Ship"),
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
        height=340, margin=dict(l=60, r=150, t=50, b=55),
        legend=dict(**_LEG_V, title_text="Solver"),
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
        height=340, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Ship"),
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
            hovertemplate=(
                f"<b>{SHIP_DISPLAY.get(ship, ship)}</b><br>"
                "%{x}: %{y:.3g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Runtime by Solver", font=dict(size=13)),
        yaxis=dict(
            title="Runtime (s)",
            type="log",
            tickformat=".3g",
            gridcolor="#334155",
        ),
        height=340, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Ship"),
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
        height=340, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Balance metric"),
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
        height=360, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Case"),
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
        sc_label = f"{d.get('icon', '')} {d.get('title', sc)}"
        fig.add_trace(go.Bar(
            name=sc_label,
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["runtime_s"],
            marker_color=SCENARIO_COLORS.get(sc, "#aaa"),
            hovertemplate=(
                f"<b>{sc_label}</b><br>"
                "%{x}: %{y:.3g} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Runtime by Case — {SHIP_DISPLAY.get(ship, ship)}", font=dict(size=13)),
        yaxis=dict(
            title="Runtime (s)",
            type="log",
            tickformat=".3g",
            gridcolor="#334155",
        ),
        height=360, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Case"),
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
    """Solver × (ship · case) heatmap.

    Model provenance (verified against benchmark_results.json):
    - ML solvers (Neural Ranker, RL Bayesian, RL Bayes+SA): each cell uses
      the model pre-trained on that ship (model_key == ship_key, is_transfer=False).
      The same model is applied to all 4 loading cases for that ship —
      there is NO case-specific fine-tuning.
    - All other solvers (Greedy, Beam Search, SA, Bayesian Opt): no
      pre-trained model; the algorithm runs from scratch on each problem.
    """
    sub = _filter(df, ships, solvers)
    if sub.empty:
        return go.Figure()
    sh = [s for s in SHIP_ORDER     if s in ships]
    sc = [s for s in SCENARIO_ORDER if s in sub["scenario"].unique()]
    sv = [s for s in SOLVER_ORDER   if s in solvers and s in sub["solver_name"].unique()]
    n_cases = len(sc)

    # Clearer column labels: "Coastal — ⚖️ Balanced"
    x_labels = [
        f"{SHIP_DISPLAY.get(ship, ship).split('(')[0].strip()} — "
        f"{CASE_DEFS.get(scenario, {}).get('icon', '')} "
        f"{CASE_DEFS.get(scenario, {}).get('title', scenario)}"
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
                # Mark ML solver cells to signal in-speciality model
                if solver in ("neural_ranker", "rl_bayesian", "rl_bayesian_sa"):
                    rt.append(f"{v:.3f}*" if not np.isnan(v) else "—")
                else:
                    rt.append(f"{v:.3f}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)

    title = (
        "Final Score — Every (Ship × Case) Combination<br>"
        "<sub>* ML solvers: score from the model trained on that ship "
        "(in-speciality, same model across all cases)  |  "
        "No asterisk: stateless algorithm, no pre-trained model</sub>"
    )
    fig = _heatmap(
        np.array(z, dtype=float), x_labels,
        [SOLVER_DISPLAY.get(s, s) for s in sv],
        text, title, "RdYlGn", 0.85, 1.0,
        colorbar_title="Score",
        height=max(320, 80 + 42 * len(sv)),
    )

    # Vertical dividers between ship groups
    for i in range(1, len(sh)):
        fig.add_shape(
            type="line",
            x0=i * n_cases - 0.5, x1=i * n_cases - 0.5,
            y0=-0.5, y1=len(sv) - 0.5,
            line=dict(color="#f59e0b", width=2),
        )

    # Ship group label annotations above the column groups
    for i, ship in enumerate(sh):
        mid_col = i * n_cases + (n_cases - 1) / 2
        fig.add_annotation(
            x=mid_col, y=len(sv) - 0.5,
            yref="y", xref="x",
            text=f"<b>{SHIP_DISPLAY.get(ship, ship).split('(')[0].strip()}</b>",
            showarrow=False,
            yshift=28,
            font=dict(size=11, color="#94a3b8"),
            xanchor="center",
        )

    fig.update_layout(margin=dict(l=120, r=20, t=90, b=80))
    return fig


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
        height=420, margin=dict(l=60, r=160, t=60, b=40),
        legend=dict(**_LEG_V, title_text="Solver"),
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
    ships_present    = [s for s in SHIP_ORDER     if s in agg["ship_key"].unique()]
    scenarios_present = [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]
    fig = go.Figure()

    # ── Data traces (one per solver) ──────────────────────────────────────────
    for solver in [s for s in SOLVER_ORDER if s in solvers]:
        rows = agg[agg["solver_name"] == solver]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["rt"], y=rows["score"],
            mode="markers",
            name=SOLVER_DISPLAY.get(solver, solver),
            legendgroup="solver",
            legendgrouptitle=dict(text="Solver", font=dict(size=11, color="#94a3b8")),
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

    # ── Dummy traces — case colour key ────────────────────────────────────────
    for sc in scenarios_present:
        d = CASE_DEFS.get(sc, {})
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"{d.get('icon', '')} {d.get('title', sc)}",
            legendgroup="case",
            legendgrouptitle=dict(text="Case (fill colour)",
                                  font=dict(size=11, color="#94a3b8")),
            marker=dict(size=10, color=SCENARIO_COLORS.get(sc, "#aaa"),
                        symbol="circle", line=dict(color="#475569", width=1)),
            showlegend=True,
        ))

    # ── Dummy traces — ship shape key ─────────────────────────────────────────
    for ship in ships_present:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=SHIP_DISPLAY.get(ship, ship),
            legendgroup="ship",
            legendgrouptitle=dict(text="Ship (marker shape)",
                                  font=dict(size=11, color="#94a3b8")),
            marker=dict(size=10, color="#94a3b8",
                        symbol=shape_map.get(ship, "circle"),
                        line=dict(color="#94a3b8", width=1)),
            showlegend=True,
        ))

    fig.update_xaxes(type="log", title="Mean Runtime (s) — log scale")
    fig.update_yaxes(title="Mean Final Score", range=[0.84, 1.01])
    fig.update_layout(
        title=dict(text="Score vs Runtime", font=dict(size=13)),
        height=420, margin=dict(l=60, r=200, t=50, b=55),
        legend=dict(**_LEG_V),
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


# ── Unloading Order plots ──────────────────────────────────────────────────────

def _ul_col_labels(scenarios: list) -> list:
    return [
        f"{UNLOADING_CASE_DEFS.get(s, {}).get('icon', '')} "
        f"{UNLOADING_CASE_DEFS.get(s, {}).get('title', s)}"
        for s in scenarios
    ]


@st.cache_data(show_spinner=False)
def plot_rehandles_heatmap(df_ul: pd.DataFrame, ships: tuple, solvers: tuple) -> go.Figure:
    """Avg rehandles per container — solver × scenario. Lower = better."""
    sub = _filter(df_ul, list(ships), list(solvers))
    if sub.empty or "avg_rehandles" not in sub.columns:
        return go.Figure()
    sc = [s for s in UNLOADING_SCENARIO_ORDER if s in sub["scenario"].unique()]
    sv = [s for s in SOLVER_ORDER if s in solvers and s in sub["solver_name"].unique()]
    g = sub.groupby(["solver_name", "scenario"])["avg_rehandles"].mean()
    z, text = [], []
    for solver in sv:
        rz, rt = [], []
        for scenario in sc:
            v = g.get((solver, scenario), np.nan)
            rz.append(v)
            rt.append(f"{v:.2f}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)
    z_arr = np.array(z, dtype=float)
    zmax = max(float(np.nanmax(z_arr)), 0.1) if not np.all(np.isnan(z_arr)) else 1.0
    return _heatmap(
        z_arr, _ul_col_labels(sc),
        [SOLVER_DISPLAY.get(s, s) for s in sv],
        text,
        "Avg Rehandles per Container  (lower = better unloading order)",
        "RdYlGn_r", 0.0, zmax,
        colorbar_title="Avg rehandles",
        height=max(280, 60 + 42 * len(sv)),
    )


@st.cache_data(show_spinner=False)
def plot_unload_score_heatmap(df_ul: pd.DataFrame, ships: tuple, solvers: tuple) -> go.Figure:
    """Unloading score heatmap — solver × scenario. Higher = better."""
    sub = _filter(df_ul, list(ships), list(solvers))
    if sub.empty or "unloading_score" not in sub.columns:
        return go.Figure()
    sc = [s for s in UNLOADING_SCENARIO_ORDER if s in sub["scenario"].unique()]
    sv = [s for s in SOLVER_ORDER if s in solvers and s in sub["solver_name"].unique()]
    g = sub.groupby(["solver_name", "scenario"])["unloading_score"].mean()
    z, text = [], []
    for solver in sv:
        rz, rt = [], []
        for scenario in sc:
            v = g.get((solver, scenario), np.nan)
            rz.append(v)
            rt.append(f"{v:.3f}" if not np.isnan(v) else "—")
        z.append(rz); text.append(rt)
    return _heatmap(
        np.array(z, dtype=float), _ul_col_labels(sc),
        [SOLVER_DISPLAY.get(s, s) for s in sv],
        text,
        "Unloading Score  (fraction of correctly-ordered stacked pairs — higher = better)",
        "RdYlGn", 0.5, 1.0,
        colorbar_title="Score",
        height=max(280, 60 + 42 * len(sv)),
    )


@st.cache_data(show_spinner=False)
def plot_conflict_vs_alignment(df_ul: pd.DataFrame, ships: tuple, solvers: tuple) -> go.Figure:
    """Grouped bars: early-heavy (conflict) vs early-light (alignment), per solver."""
    sub = _filter(df_ul, list(ships), list(solvers))
    if sub.empty or "avg_rehandles" not in sub.columns:
        return go.Figure()
    compare = ["stop_early_heavy", "stop_early_light"]
    sub = sub[sub["scenario"].isin(compare)]
    if sub.empty:
        return go.Figure()
    agg = sub.groupby(["solver_name", "scenario"])["avg_rehandles"].mean().reset_index()
    sv = [s for s in SOLVER_ORDER if s in solvers and s in agg["solver_name"].unique()]
    fig = go.Figure()
    for scenario in compare:
        d = UNLOADING_CASE_DEFS.get(scenario, {})
        rows = agg[agg["scenario"] == scenario].copy()
        rows = rows.set_index("solver_name").reindex(
            [s for s in sv if s in rows["solver_name"].values]
        ).reset_index()
        fig.add_trace(go.Bar(
            name=f"{d.get('icon', '')} {d.get('title', scenario)}",
            x=[SOLVER_DISPLAY.get(s, s) for s in rows["solver_name"]],
            y=rows["avg_rehandles"],
            marker_color=UNLOADING_SCENARIO_COLORS.get(scenario, "#aaa"),
            text=[f"{v:.2f}" if not pd.isna(v) else "—" for v in rows["avg_rehandles"]],
            textposition="outside",
            hovertemplate=(
                f"<b>{d.get('title', scenario)}</b><br>"
                "%{x}: %{y:.3f} avg rehandles<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Conflict vs Alignment — Avg Rehandles per Container",
            font=dict(size=13),
        ),
        yaxis=dict(title="Avg rehandles per container", rangemode="tozero"),
        height=360, margin=dict(l=60, r=20, t=68, b=65),
        legend=dict(**_LEG_H, title_text="Scenario"),
        **_DARK,
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_stop_balance_chart(
    df_ul: pd.DataFrame,
    ships: tuple,
    solvers: tuple,
    scenario: str,
) -> go.Figure:
    """Line chart: PS / FA / Diag balance ratios after each port stop is completed."""
    sub = _filter(df_ul, list(ships), list(solvers))
    sub = sub[sub["scenario"] == scenario]
    if sub.empty or "post_stop_balance" not in sub.columns:
        return go.Figure()

    rows = []
    for _, rec in sub.iterrows():
        psb = rec["post_stop_balance"]
        if not isinstance(psb, list):
            continue
        for entry in psb:
            rows.append({"solver_name": rec["solver_name"], **entry})
    if not rows:
        return go.Figure()

    exp = pd.DataFrame(rows)
    agg = (
        exp.groupby(["solver_name", "stop_completed"])[["ps_ratio", "fa_ratio", "diag_ratio"]]
        .mean()
        .reset_index()
    )

    metrics = [
        ("ps_ratio",   "PS",   "solid"),
        ("fa_ratio",   "FA",   "dash"),
        ("diag_ratio", "Diag", "dot"),
    ]
    fig = go.Figure()
    present_solvers = [s for s in SOLVER_ORDER if s in agg["solver_name"].unique()]
    for solver in present_solvers:
        d = agg[agg["solver_name"] == solver].sort_values("stop_completed")
        color = SOLVER_COLORS.get(solver, "#aaa")
        display = SOLVER_DISPLAY.get(solver, solver)
        for col, label, dash in metrics:
            fig.add_trace(go.Scatter(
                x=d["stop_completed"],
                y=d[col],
                mode="lines+markers",
                name=f"{display} — {label}",
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{display} — {label}</b><br>"
                    "After stop %{x}: %{y:.4f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(text="Balance Ratios After Each Port Stop", font=dict(size=13)),
        xaxis=dict(title="Stop completed", dtick=1),
        yaxis=dict(title="Balance ratio", range=[0.5, 1.02]),
        height=420,
        margin=dict(l=60, r=200, t=50, b=55),
        legend=dict(**_LEG_V, title_text="Solver — Metric"),
        **_DARK,
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_stop_rehandles_chart(
    df_ul: pd.DataFrame,
    ships: tuple,
    solvers: tuple,
    scenario: str,
) -> go.Figure:
    """Bar chart: rehandles_at_stop per stop, averaged across seeds and ships."""
    sub = _filter(df_ul, list(ships), list(solvers))
    sub = sub[sub["scenario"] == scenario]
    if sub.empty or "post_stop_balance" not in sub.columns:
        return go.Figure()

    rows = []
    for _, rec in sub.iterrows():
        psb = rec["post_stop_balance"]
        if not isinstance(psb, list):
            continue
        for entry in psb:
            if "rehandles_at_stop" not in entry:
                continue
            rows.append({
                "solver_name":      rec["solver_name"],
                "stop_completed":   entry["stop_completed"],
                "rehandles_at_stop": entry["rehandles_at_stop"],
            })
    if not rows:
        return go.Figure()

    exp = pd.DataFrame(rows)
    agg = (
        exp.groupby(["solver_name", "stop_completed"])["rehandles_at_stop"]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    present_solvers = [s for s in SOLVER_ORDER if s in agg["solver_name"].unique()]
    for solver in present_solvers:
        d = agg[agg["solver_name"] == solver].sort_values("stop_completed")
        fig.add_trace(go.Bar(
            name=SOLVER_DISPLAY.get(solver, solver),
            x=d["stop_completed"],
            y=d["rehandles_at_stop"],
            marker_color=SOLVER_COLORS.get(solver, "#aaa"),
            hovertemplate=(
                f"<b>{SOLVER_DISPLAY.get(solver, solver)}</b><br>"
                "Stop %{x}: %{y:.1f} rehandles<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Rehandles Required at Each Port Stop", font=dict(size=13)),
        xaxis=dict(title="Stop completed", dtick=1),
        yaxis=dict(title="Avg rehandles at stop", rangemode="tozero"),
        barmode="group",
        height=340,
        margin=dict(l=60, r=200, t=50, b=55),
        legend=dict(**_LEG_V, title_text="Solver"),
        **_DARK,
    )
    return fig


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
df_ul  = _df(data.get("unloading", []))
has_transfer  = not df_tr.empty
has_unloading = not df_ul.empty

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

tab_ov, tab_case, tab_ship, tab_flex, tab_unload, tab_raw = st.tabs([
    "🗺 Overview",
    "📋 Case Level",
    "🚢 Ship Level",
    "🔀 Flexibility",
    "⚓ Unloading Order",
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
        st.info(
            "**How to read this chart**\n\n"
            "Each column is one (ship, case) combination; each row is one solver. "
            "Uniform colour across a row = consistent solver. Patchy row = "
            "performance depends on ship size or case type.\n\n"
            "**Model provenance (verified)**\n\n"
            "| Solver | What runs | Model trained on |\n"
            "|--------|-----------|------------------|\n"
            "| Neural Ranker | Pre-trained MLP | **That ship only** — one model per ship, "
            "applied unchanged to all 4 cases |\n"
            "| RL Bayesian | Pre-trained MLP | **That ship only** — one model per ship, "
            "applied unchanged to all 4 cases |\n"
            "| RL Bayes + SA | RL Bayesian warm-start → SA | Same ship-specific pkl as RL Bayesian |\n"
            "| Greedy / Beam Search / SA / Bayesian Opt | Stateless algorithm | "
            "No pre-trained model — runs from scratch each time |\n\n"
            "Cells marked **\\*** are ML solver results. "
            "The ★ (in-speciality) notation used in the Transfer tab means the same thing: "
            "model tested on the ship it was trained for."
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
# Unloading Order
# ══════════════════════════════════════════════════════════════════════════════

with tab_unload:
    st.subheader("Unloading Order Analysis")
    st.caption(
        "How well does each solver stack containers in the correct unloading order? "
        "Each scenario stress-tests a different alignment between weight and port stop."
    )

    if not has_unloading:
        st.info(
            "**No unloading benchmark data found.**\n\n"
            "Generate it with:\n"
            "```\nconda run -n personal python benchmark.py\n```\n\n"
            "To run only unloading scenarios (faster):\n"
            "```\nconda run -n personal python benchmark.py "
            "--ships coastal --no-transfer\n```"
        )
    else:
        # ── Scenario definition cards ──────────────────────────────────────
        st.markdown("#### Scenario Definitions")
        card_cols = st.columns(4)
        for col, sc_key in zip(card_cols, UNLOADING_SCENARIO_ORDER):
            d = UNLOADING_CASE_DEFS.get(sc_key, {})
            colour = UNLOADING_SCENARIO_COLORS.get(sc_key, "#aaa")
            with col:
                st.markdown(
                    f"<div style='border-left: 4px solid {colour}; padding: 8px 12px; "
                    f"background: rgba(30,41,59,0.7); border-radius: 4px; margin-bottom: 8px;'>"
                    f"<b>{d.get('icon', '')} {d.get('title', sc_key)}</b><br>"
                    f"<span style='font-size:0.82em; color:#94a3b8;'>{d.get('summary', '')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Details"):
                    st.caption(f"**Tests:** {d.get('tests', '—')}")
                    st.caption(f"**Port stops:** {d.get('stops', '?')}")
                    stop1 = d.get("stop1_kg", "—")
                    stop2 = d.get("stop2_kg", "—")
                    if stop2 != "—":
                        st.caption(f"**Stop 1 weights:** {stop1} kg")
                        st.caption(f"**Stop 2+ weights:** {stop2} kg")
                    else:
                        st.caption(f"**All weights:** {stop1} kg")

        st.divider()

        # ── Metric explanation ─────────────────────────────────────────────
        with st.expander("ℹ️ How the metrics are calculated", expanded=False):
            st.markdown(
                "**Avg rehandles per container** *(primary metric, lower = better)*\n\n"
                "Simulates unloading stop-by-stop. At each stop, for every container "
                "being unloaded, count all containers above it in the same column that "
                "are still aboard and going to a *later* stop — those must be temporarily "
                "moved aside. Divide total moves by containers placed.\n\n"
                "> **Note:** containers going to the *same* stop do not count as rehandles — "
                "they unload together at that stop and come off first without extra moves.\n\n"
                "**Unloading score** *(secondary metric, higher = better)*\n\n"
                "Fraction of stacked container pairs (same column, overlapping bays) "
                "where the upper container unloads at the same or earlier port than "
                "the lower one. Score = 1.0 means every pair is correctly ordered."
            )

        # ── Primary heatmaps — both pre-built and cached, toggled by radio ────
        # Build both figures now (cache_data ensures this is O(1) on subsequent
        # runs with the same inputs — no algorithm re-runs, just a cache lookup).
        _ships_t   = tuple(sel_ships)
        _solvers_t = tuple(sel_solvers)
        fig_rehandles = plot_rehandles_heatmap(df_ul, _ships_t, _solvers_t)
        fig_unload_sc = plot_unload_score_heatmap(df_ul, _ships_t, _solvers_t)

        heatmap_view = st.radio(
            "Heatmap",
            ["Avg Rehandles", "Unloading Score"],
            horizontal=True,
            key="ul_heatmap_view",
        )
        if heatmap_view == "Avg Rehandles":
            st.plotly_chart(fig_rehandles, use_container_width=True)
            st.caption(
                "Mean avg rehandles per container across ships and seeds. "
                "**Green = fewer rehandles = better.** 0.0 = perfect ordering."
            )
        else:
            st.plotly_chart(fig_unload_sc, use_container_width=True)
            st.caption(
                "Fraction of correctly-ordered stacked pairs. "
                "**Green = more pairs in correct order = better.** 1.000 = no violations."
            )

        st.divider()

        # ── Conflict vs Alignment comparison ──────────────────────────────
        st.markdown("#### Conflict vs Alignment — Does weight bias help or hurt?")
        st.caption(
            "Compares the hardest case (🔴 early-stop containers are heavy — "
            "weight-first sort fights unloading order) against the easiest "
            "(🟢 early-stop containers are light — weight-first sort naturally "
            "aligns with unloading order). Gap = cost of the conflict."
        )
        fig_cva = plot_conflict_vs_alignment(df_ul, _ships_t, _solvers_t)
        if fig_cva.data:
            st.plotly_chart(fig_cva, use_container_width=True)
            st.caption(
                "Smaller gap between 🔴 and 🟢 bars = solver is better at "
                "overcoming the weight-vs-order conflict."
            )
        else:
            st.info("Both early-stop scenarios need data to show this chart.")

        # ── Balance after each port stop ───────────────────────────────────
        st.divider()
        st.subheader("Balance After Each Port Stop")
        st.caption(
            "Balance ratios (PS / FA / Diagonal) measured on the cargo that remains "
            "aboard after each port stop. A ship must stay balanced throughout the "
            "voyage, not just at departure."
        )
        _ul_filtered = _filter(df_ul, sel_ships, sel_solvers)
        _ul_scenarios_present = [
            s for s in UNLOADING_SCENARIO_ORDER
            if not _ul_filtered.empty and s in _ul_filtered["scenario"].unique()
        ]
        if _ul_scenarios_present:
            scenario_sel = st.selectbox(
                "Scenario",
                options=_ul_scenarios_present,
                format_func=lambda s: (
                    f"{UNLOADING_CASE_DEFS[s]['icon']} {UNLOADING_CASE_DEFS[s]['title']}"
                ),
                key="ul_stop_balance_scenario",
            )
            if scenario_sel:
                fig_sb = plot_stop_balance_chart(
                    df_ul, _ships_t, _solvers_t, scenario_sel
                )
                fig_sr = plot_stop_rehandles_chart(
                    df_ul, _ships_t, _solvers_t, scenario_sel
                )
                if fig_sb.data:
                    st.plotly_chart(fig_sb, use_container_width=True)
                    st.caption(
                        "Solid = PS ratio · Dashed = FA ratio · Dotted = Diagonal ratio. "
                        "Each point is the mean across selected ships and seeds."
                    )
                if fig_sr.data:
                    st.plotly_chart(fig_sr, use_container_width=True)
                    st.caption(
                        "Number of containers that must be moved aside at each stop to "
                        "access targets. Only containers staying aboard longer count. "
                        "Mean across selected ships and seeds."
                    )
                if not fig_sb.data and not fig_sr.data:
                    st.info(
                        "No per-stop balance data for this scenario. "
                        "Re-run benchmarks to populate."
                    )
        else:
            st.info("No unloading scenario data available.")

        # ── Raw unloading table ────────────────────────────────────────────
        st.divider()
        st.markdown("#### Raw Unloading Results")
        ul_raw = _filter(df_ul, sel_ships, sel_solvers)
        ul_show_cols = [
            c for c in [
                "ship_key", "scenario", "solver_name", "seed",
                "unloading_score", "avg_rehandles", "rehandle_count",
                "placed", "final_score", "runtime_s",
            ] if c in ul_raw.columns
        ]
        if not ul_raw.empty and ul_show_cols:
            style = ul_raw[ul_show_cols].sort_values(
                ["ship_key", "scenario", "solver_name", "seed"]
            ).reset_index(drop=True)
            gradient_cols = [c for c in ["unloading_score", "final_score"] if c in style.columns]
            st.dataframe(
                style.style.background_gradient(
                    subset=gradient_cols, cmap="RdYlGn", vmin=0.5, vmax=1.0
                ) if gradient_cols else style,
                use_container_width=True,
                height=380,
            )
            st.caption(f"{len(ul_raw)} rows shown.")


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
