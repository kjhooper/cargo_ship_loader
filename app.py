"""Cargo Ship Loader — Interactive Streamlit Demo

Run locally:
    conda run -n personal streamlit run app.py

Deploy:
    Push to GitHub, connect repo on share.streamlit.io.
"""

from __future__ import annotations

import io
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from algorithm import CargoLoader
from models import CargoShip, ShippingContainer
from visualizer import ComparisonVisualizer, Visualizer

try:
    from solvers import BeamSearchSolver, SimulatedAnnealingSolver
    _HEURISTICS = True
except ImportError:
    _HEURISTICS = False

try:
    from solvers import BayesianOptSolver
    _BAYESIAN = True
except ImportError:
    _BAYESIAN = False

try:
    from solvers.neural_ranker import NeuralRankerSolver
    _NEURAL = True
except ImportError:
    _NEURAL = False

try:
    from solvers.rl_bayesian import RLBayesianSolver
    _RL_BAYESIAN = True
except ImportError:
    _RL_BAYESIAN = False

try:
    from solvers.rl_bayesian_sa import RLBayesianSASolver
    _RL_BAYESIAN_SA = True
except ImportError:
    _RL_BAYESIAN_SA = False

MODELS_DIR = Path(__file__).parent / "models"

# ── Pre-built ship configurations ────────────────────────────────────────────

PREBUILT: Dict[str, Dict] = {
    "Coastal Feeder  (12 × 9 × 5)": dict(
        length=12, base_width=5, max_width=9, height=5,
        width_step=1, max_weight=500_000.0,
        model_key="coastal", default_20ft=12, default_40ft=4,
    ),
    "Handymax  (24 × 11 × 7)": dict(
        length=24, base_width=6, max_width=11, height=7,
        width_step=1, max_weight=1_500_000.0,
        model_key="handymax", default_20ft=35, default_40ft=12,
    ),
    "Panamax  (36 × 13 × 9)": dict(
        length=36, base_width=7, max_width=13, height=9,
        width_step=1, max_weight=3_000_000.0,
        model_key="panamax", default_20ft=60, default_40ft=25,
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ship(cfg: Dict) -> CargoShip:
    return CargoShip(
        length=cfg["length"], base_width=cfg["base_width"],
        max_width=cfg["max_width"], height=cfg["height"],
        width_step=cfg["width_step"], max_weight=cfg["max_weight"],
    )


def _closest_model_key(cfg: Dict) -> str:
    """Find which pre-built model best matches a custom ship config."""
    vol = cfg["length"] * cfg["max_width"] * cfg["height"]
    best, best_dist = "panamax", float("inf")
    for pb in PREBUILT.values():
        d = abs(pb["length"] * pb["max_width"] * pb["height"] - vol)
        if d < best_dist:
            best_dist = d
            best = pb["model_key"]
    return best


@st.cache_resource(show_spinner=False)
def load_rl_bayesian(model_key: str, ship_cfg_hash: str) -> Optional["RLBayesianSolver"]:
    """Load a pre-trained RLBayesianSolver from disk (cached per model key)."""
    if not _RL_BAYESIAN:
        return None
    path = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
    if not path.exists():
        return None
    dummy_ship = CargoShip(length=36, base_width=7, max_width=13,
                           height=9, width_step=1, max_weight=50_000.0)
    return RLBayesianSolver.load_model(dummy_ship, str(path))


@st.cache_resource(show_spinner=False)
def load_neural_ranker(model_key: str, ship_cfg_hash: str) -> Optional[NeuralRankerSolver]:
    """Load a pre-trained NeuralRankerSolver from disk (cached per model key)."""
    if not _NEURAL:
        return None
    path = MODELS_DIR / f"neural_ranker_{model_key}.pkl"
    if not path.exists():
        return None
    # Dummy ship — will be replaced on load()
    dummy_ship = CargoShip(length=36, base_width=7, max_width=13,
                           height=9, width_step=1, max_weight=50_000.0)
    return NeuralRankerSolver.load_model(dummy_ship, str(path))


def make_containers(n_20ft, n_40ft, dist_type, w_min, w_max,
                    w_mean, w_std, seed) -> List[ShippingContainer]:
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()

    def sample():
        if dist_type == "Uniform":
            return round(rng.uniform(w_min, w_max), 1)
        elif dist_type == "Normal":
            return round(max(w_min, min(w_max, rng.gauss(w_mean, w_std))), 1)
        elif dist_type == "Bimodal":
            mid = w_min + (w_max - w_min) * 0.35
            if rng.random() < 0.5:
                return round(rng.uniform(w_min, mid), 1)
            return round(rng.uniform(w_max - (w_max - w_min) * 0.35, w_max), 1)
        else:  # Heavy-biased
            t = rng.betavariate(3.0, 1.0)
            return round(w_min + t * (w_max - w_min), 1)

    containers = (
        [ShippingContainer(size=1, weight=sample()) for _ in range(n_20ft)]
        + [ShippingContainer(size=2, weight=sample()) for _ in range(n_40ft)]
    )
    rng.shuffle(containers)
    return containers


def run_solver(name: str, ship: CargoShip, containers: List[ShippingContainer],
               beam_k: int, sa_iters: int, n_trials: int,
               neural_ranker: Optional[NeuralRankerSolver],
               rl_bayesian=None, model_key: str = "panamax") -> Tuple[List[Dict], float]:
    t0 = time.perf_counter()
    if name == "Greedy":
        solver = CargoLoader(ship)
    elif name == "Beam Search":
        solver = BeamSearchSolver(ship, beam_width=beam_k)
    elif name == "Simulated Annealing":
        solver = SimulatedAnnealingSolver(ship, n_iterations=sa_iters, seed=42)
    elif name == "Bayesian Opt":
        solver = BayesianOptSolver(ship, n_trials=n_trials, seed=42)
    elif name == "Neural Ranker":
        if neural_ranker is None:
            return [], 0.0
        neural_ranker.ship = ship          # point at fresh ship
        neural_ranker._fitted = True
        solver = neural_ranker
    elif name == "RL Bayesian":
        if rl_bayesian is None:
            return [], 0.0
        rl_bayesian.ship = ship
        rl_bayesian._fitted = True
        solver = rl_bayesian
    elif name == "RL Bayesian + SA":
        if not _RL_BAYESIAN_SA:
            return [], 0.0
        rl_pkl = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        solver = RLBayesianSASolver(
            ship,
            n_iterations=sa_iters,
            seed=42,
            model_path=str(rl_pkl) if rl_pkl.exists() else None,
        )
    else:
        return [], 0.0
    manifest = solver.load(containers)
    return manifest, time.perf_counter() - t0


def collect_stats(manifest: List[Dict], ship: CargoShip, elapsed: float) -> Dict:
    placed = [e for e in manifest if e["placed"]]
    p, s = ship.port_starboard_balance()
    f, a = ship.fore_aft_balance()
    fp, fs, ap, as_ = ship.quadrant_balance()
    total = ship.total_weight or 1.0
    ps = min(p, s) / max(p, s) if max(p, s) > 0 else 1.0
    fa = min(f, a) / max(f, a) if max(f, a) > 0 else 1.0
    d1, d2 = fp + as_, fs + ap
    diag = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0
    gz = sum(
        e["weight"] * e["tier"] for e in placed
    ) / (total * max(ship.height - 1, 1))
    return {
        "Containers placed":   f"{len(placed)} / {len(manifest)}",
        "Total weight (kg)":   f"{total:,.0f}",
        "Port weight":         f"{p:,.0f}  ({100*p/total:.1f} %)",
        "Starboard weight":    f"{s:,.0f}  ({100*s/total:.1f} %)",
        "Fore weight":         f"{f:,.0f}  ({100*f/total:.1f} %)",
        "Aft weight":          f"{a:,.0f}  ({100*a/total:.1f} %)",
        "PS balance ratio":    f"{ps:.4f}",
        "FA balance ratio":    f"{fa:.4f}",
        "Diagonal ratio":      f"{diag:.4f}",
        "Final score":         f"{(ps+fa+diag)/3:.4f}",
        "CoG height (norm)":   f"{gz:.3f}",
        "Runtime (s)":         f"{elapsed:.2f}",
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

_DARK = dict(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")


def plot_hull_preview(cfg: Dict) -> go.Figure:
    """3D interactive empty-ship preview showing hull geometry and valid hold space."""
    ship    = make_ship(cfg)
    max_exp = (ship.width - ship.base_width) // 2

    def _box(x0, x1, y0, y1, z0, z1):
        vx = [x0,x1,x1,x0, x0,x1,x1,x0]
        vy = [y0,y0,y1,y1, y0,y0,y1,y1]
        vz = [z0,z0,z0,z0, z1,z1,z1,z1]
        ti = [0,0,4,4, 0,0,2,2, 0,0,1,1]
        tj = [1,2,5,6, 1,5,3,7, 3,7,2,6]
        tk = [2,3,6,7, 5,4,7,6, 7,4,6,5]
        return vx, vy, vz, ti, tj, tk

    def _batch(boxes, colors):
        ax,ay,az,ai,aj,ak,fc = [],[],[],[],[],[],[]
        off = 0
        for (x0,x1,y0,y1,z0,z1), c in zip(boxes, colors):
            vx,vy,vz,ti,tj,tk = _box(x0,x1,y0,y1,z0,z1)
            ax+=vx; ay+=vy; az+=vz
            ai+=[v+off for v in ti]; aj+=[v+off for v in tj]; ak+=[v+off for v in tk]
            fc+=[c]*12
            off += 8
        return ax, ay, az, ai, aj, ak, fc

    traces = []

    # ── Hull walls — shade gets lighter at higher tiers (less restrictive) ────
    h_boxes, h_cols = [], []
    for tier in range(ship.height):
        exp = min(tier // ship.width_step, max_exp)
        cw  = ship.base_width + 2 * exp
        lc  = (ship.width - cw) // 2
        rc  = lc + cw
        t   = tier / max(ship.height - 1, 1)
        a   = 0.90 - 0.35 * t          # more opaque at keel, less at deck
        col = f"rgba(51,65,85,{a:.2f})"
        if lc > 0:
            h_boxes.append((0, ship.length, 0, lc, tier, tier + 1))
            h_cols.append(col)
        if rc < ship.width:
            h_boxes.append((0, ship.length, rc, ship.width, tier, tier + 1))
            h_cols.append(col)

    if h_boxes:
        hx,hy,hz,hi,hj,hk,hfc = _batch(h_boxes, h_cols)
        traces.append(go.Mesh3d(
            x=hx, y=hy, z=hz, i=hi, j=hj, k=hk,
            facecolor=hfc, flatshading=True, opacity=0.88,
            name="Hull", showlegend=True, showscale=False,
            hoverinfo="skip",
        ))

    # ── Valid hold — thin teal floor at each tier level ───────────────────────
    v_boxes, v_cols = [], []
    for tier in range(ship.height):
        exp = min(tier // ship.width_step, max_exp)
        cw  = ship.base_width + 2 * exp
        lc  = (ship.width - cw) // 2
        rc  = lc + cw
        t   = tier / max(ship.height - 1, 1)
        a   = 0.10 + 0.18 * t          # brighter at top
        v_boxes.append((0, ship.length, lc, rc, tier, tier + 0.06))
        v_cols.append(f"rgba(34,211,238,{a:.2f})")

    vx,vy,vz,vi,vj,vk,vfc = _batch(v_boxes, v_cols)
    traces.append(go.Mesh3d(
        x=vx, y=vy, z=vz, i=vi, j=vj, k=vk,
        facecolor=vfc, flatshading=True, opacity=0.7,
        name="Valid hold", showlegend=True, showscale=False,
        hoverinfo="skip",
    ))

    # ── Ship boundary wireframe ───────────────────────────────────────────────
    L, W, H = ship.length, ship.width, ship.height
    pts   = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),
             (0,0,H),(L,0,H),(L,W,H),(0,W,H)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex += [pts[a][0], pts[b][0], None]
        ey += [pts[a][1], pts[b][1], None]
        ez += [pts[a][2], pts[b][2], None]
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(148,163,184,0.7)", width=2),
        name="Ship outline", hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        title=dict(
            text=(
                f"Empty cargo hold — "
                f"{ship.length}L × {ship.width}W × {ship.height}H  "
                f"(keel width {ship.base_width})"
            ),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title="Bay (fore → aft)",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(title="Column (port → stbd)",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            zaxis=dict(title="Tier",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            bgcolor="#1e293b",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.2)),
        ),
        height=420,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#334155", borderwidth=1,
        ),
        font=dict(color="#e2e8f0"),
    )
    return fig


def plot_weight_dist(containers: List[ShippingContainer]) -> go.Figure:
    twenty = [c.weight for c in containers if c.size == 1]
    forty  = [c.weight for c in containers if c.size == 2]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=twenty, name="20 ft", nbinsx=20,
                               marker_color="#60a5fa", opacity=0.8))
    fig.add_trace(go.Histogram(x=forty, name="40 ft", nbinsx=20,
                               marker_color="#f59e0b", opacity=0.8))
    fig.update_layout(
        barmode="overlay",
        title="Container weight distribution",
        xaxis_title="Weight (kg)",
        yaxis_title="Count",
        height=260,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_DARK,
    )
    return fig


def plot_final_state(manifest: List[Dict], ship: CargoShip,
                     title: str = "") -> go.Figure:
    placed = [e for e in manifest if e["placed"]]
    weights = [e["weight"] for e in placed]
    max_w = max(weights) if weights else 1.0

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.73, 0.27],
        subplot_titles=[title or "Final loading state", "Weight balance"],
        horizontal_spacing=0.08,
    )

    # ── Left: top-down ship grid ──────────────────────────────────────────────
    for col in range(ship.width):
        frac = float((ship.cargo_hold[0, col, :] == -1.0).sum()) / ship.height
        if frac > 0:
            fig.add_shape(type="rect", x0=0, x1=ship.length, y0=col, y1=col + 1,
                          fillcolor=f"rgba(51,65,85,{min(frac * 1.8, 0.85):.2f})",
                          line_width=0, row=1, col=1)

    # Invisible scatter for hover + colorbar; shapes for container visuals
    hover_x, hover_y, hover_text, norm_vals = [], [], [], []
    for e in placed:
        pos = e["bay"] * 2 + (e.get("half") or 0)
        t = e["weight"] / max_w
        color = pc.sample_colorscale("RdBu_r", [t])[0]
        fig.add_shape(
            type="rect",
            x0=pos + 0.06, x1=pos + e["size"] - 0.06,
            y0=e["col"] + 0.08, y1=e["col"] + 0.92,
            fillcolor=color, line=dict(color="#0f172a", width=0.5),
            row=1, col=1,
        )
        hover_x.append(pos + e["size"] / 2)
        hover_y.append(e["col"] + 0.5)
        half = e.get("half")
        half_str = "F" if half == 0 else "B" if half == 1 else "-"
        hover_text.append(
            f"Weight: {e['weight']:,.0f} kg<br>"
            f"Size: {'40 ft' if e['size'] == 2 else '20 ft'}<br>"
            f"Bay {e['bay']} ({half_str}), Col {e['col']}, Tier {e.get('tier', '?')}"
        )
        norm_vals.append(t)

    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers",
        marker=dict(
            size=10, opacity=0,
            color=norm_vals, colorscale="RdBu_r", cmin=0, cmax=1,
            colorbar=dict(
                title=dict(text="Weight (kg)", side="right"),
                thickness=12, len=0.7,
                tickvals=[0, 0.5, 1],
                ticktext=["0", f"{max_w / 2:,.0f}", f"{max_w:,.0f}"],
                x=0.69,
            ),
            showscale=True,
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        showlegend=False,
    ), row=1, col=1)

    fig.update_xaxes(title_text="Bay (fore → aft)", range=[0, ship.length],
                     dtick=2, row=1, col=1)
    fig.update_yaxes(title_text="Column (port → starboard)", range=[0, ship.width],
                     row=1, col=1)

    # ── Right: balance bars ───────────────────────────────────────────────────
    p, s = ship.port_starboard_balance()
    f, a = ship.fore_aft_balance()
    total = ship.total_weight or 1.0
    pf, sf = p / total, s / total
    ff, af = f / total, a / total
    ps_ok = abs(pf - sf) <= 0.10
    fa_ok = abs(ff - af) <= 0.10

    fig.add_trace(go.Bar(
        name="Port / Fore", orientation="h",
        x=[pf, ff], y=["Port / Starboard", "Fore / Aft"],
        marker_color=["#3b82f6" if ps_ok else "#ef4444",
                      "#3b82f6" if fa_ok else "#ef4444"],
        text=[f"Port {100*pf:.1f}%", f"Fore {100*ff:.1f}%"],
        textposition="inside", showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name="Stbd / Aft", orientation="h",
        x=[sf, af], y=["Port / Starboard", "Fore / Aft"],
        marker_color=["#f97316" if ps_ok else "#f59e0b",
                      "#f97316" if fa_ok else "#f59e0b"],
        text=[f"Stbd {100*sf:.1f}%", f"Aft {100*af:.1f}%"],
        textposition="inside", showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Weight fraction", range=[0, 1], row=1, col=2)
    fig.update_layout(
        barmode="stack",
        height=430,
        margin=dict(l=50, r=20, t=60, b=40),
        **_DARK,
    )
    return fig


def plot_3d_state(manifest: List[Dict], ship: CargoShip, title: str = "") -> go.Figure:
    """3D interactive ship loading visualisation — hull taper + coloured container boxes."""
    placed = [e for e in manifest if e["placed"]]
    max_w  = max((e["weight"] for e in placed), default=1.0)

    # ── Box-mesh helpers ──────────────────────────────────────────────────────
    def _box(x0, x1, y0, y1, z0, z1):
        vx = [x0,x1,x1,x0, x0,x1,x1,x0]
        vy = [y0,y0,y1,y1, y0,y0,y1,y1]
        vz = [z0,z0,z0,z0, z1,z1,z1,z1]
        ti = [0,0,4,4, 0,0,2,2, 0,0,1,1]
        tj = [1,2,5,6, 1,5,3,7, 3,7,2,6]
        tk = [2,3,6,7, 5,4,7,6, 7,4,6,5]
        return vx, vy, vz, ti, tj, tk

    def _batch(boxes, colors):
        ax,ay,az,ai,aj,ak,fc = [],[],[],[],[],[],[]
        off = 0
        for (x0,x1,y0,y1,z0,z1), c in zip(boxes, colors):
            vx,vy,vz,ti,tj,tk = _box(x0,x1,y0,y1,z0,z1)
            ax+=vx; ay+=vy; az+=vz
            ai+=[v+off for v in ti]; aj+=[v+off for v in tj]; ak+=[v+off for v in tk]
            fc+=[c]*12
            off += 8
        return ax, ay, az, ai, aj, ak, fc

    traces = []

    # ── Hull walls (invalid cells shown as tinted solid blocks) ──────────────
    max_exp = (ship.width - ship.base_width) // 2
    h_boxes, h_cols = [], []
    for tier in range(ship.height):
        exp = min(tier // ship.width_step, max_exp)
        cw  = ship.base_width + 2 * exp
        lc  = (ship.width - cw) // 2
        rc  = lc + cw
        if lc > 0:
            h_boxes.append((0, ship.length, 0, lc, tier, tier + 1))
            h_cols.append("rgba(51,65,85,0.6)")
        if rc < ship.width:
            h_boxes.append((0, ship.length, rc, ship.width, tier, tier + 1))
            h_cols.append("rgba(51,65,85,0.6)")
    if h_boxes:
        hx,hy,hz,hi,hj,hk,hfc = _batch(h_boxes, h_cols)
        traces.append(go.Mesh3d(
            x=hx, y=hy, z=hz, i=hi, j=hj, k=hk,
            facecolor=hfc, flatshading=True, opacity=0.55,
            name="Hull", showlegend=True, showscale=False,
            hoverinfo="skip",
        ))

    # ── Containers (coloured boxes) ───────────────────────────────────────────
    if placed:
        c_boxes, c_cols = [], []
        hov_x, hov_y, hov_z, hov_text = [], [], [], []
        for e in placed:
            pos   = e["bay"] * 2 + (e.get("half") or 0)
            t     = e["weight"] / max_w
            color = pc.sample_colorscale("RdBu_r", [t])[0]
            c_boxes.append((
                pos, pos + e["size"],
                e["col"], e["col"] + 1,
                e["tier"], e["tier"] + 1,
            ))
            c_cols.append(color)
            hov_x.append(pos + e["size"] / 2)
            hov_y.append(e["col"] + 0.5)
            hov_z.append(e["tier"] + 1.05)  # just above container top face
            half = e.get("half")
            half_str = "F" if half == 0 else "B" if half == 1 else "-"
            hov_text.append(
                f"ID {e['container_id']} · "
                f"{'40 ft' if e['size'] == 2 else '20 ft'}<br>"
                f"Weight: {e['weight']:,.0f} kg<br>"
                f"Bay {e['bay']} ({half_str}), Col {e['col']}, Tier {e['tier']}"
            )

        cx,cy,cz,ci,cj,ck,cfc = _batch(c_boxes, c_cols)
        traces.append(go.Mesh3d(
            x=cx, y=cy, z=cz, i=ci, j=cj, k=ck,
            facecolor=cfc, flatshading=True, opacity=0.92,
            name="Containers", showlegend=True, showscale=False,
            hoverinfo="skip",
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.5),
        ))
        # Near-invisible markers just above each container — carry hover text
        traces.append(go.Scatter3d(
            x=hov_x, y=hov_y, z=hov_z,
            mode="markers",
            marker=dict(size=8, opacity=0.01, color="white"),
            customdata=hov_text,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ))
        # Dummy trace for weight colorbar
        wts = [e["weight"] for e in placed]
        traces.append(go.Scatter3d(
            x=[None]*len(wts), y=[None]*len(wts), z=[None]*len(wts),
            mode="markers",
            marker=dict(
                color=wts, colorscale="RdBu_r",
                cmin=min(wts), cmax=max(wts),
                colorbar=dict(
                    title=dict(text="Weight (kg)", side="right"),
                    thickness=14, len=0.55, x=1.0, tickformat=",d",
                ),
                showscale=True, size=0,
            ),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Ship boundary wireframe ───────────────────────────────────────────────
    L, W, H = ship.length, ship.width, ship.height
    pts   = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),
             (0,0,H),(L,0,H),(L,W,H),(0,W,H)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex += [pts[a][0], pts[b][0], None]
        ey += [pts[a][1], pts[b][1], None]
        ez += [pts[a][2], pts[b][2], None]
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(148,163,184,0.7)", width=2),
        name="Ship outline", hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    p, s = ship.port_starboard_balance()
    f, a_w = ship.fore_aft_balance()
    total  = ship.total_weight or 1.0
    n_pl   = len(placed)
    n_tot  = len(manifest)
    ps_r   = min(p, s) / max(p, s) if max(p, s) > 0 else 1.0
    fa_r   = min(f, a_w) / max(f, a_w) if max(f, a_w) > 0 else 1.0

    max_dim = max(L, W, H)
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        uirevision=title or "3D Loading State",  # reset camera when solver changes
        title=dict(
            text=(
                f"{title or '3D Loading State'}<br>"
                f"<sup>Placed {n_pl}/{n_tot} · "
                f"PS {ps_r:.3f} · FA {fa_r:.3f} · "
                f"Total weight {total:,.0f} kg</sup>"
            ),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title="Bay (fore → aft)",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(title="Column (port → stbd)",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            zaxis=dict(title="Tier",
                       backgroundcolor="#1e293b", gridcolor="#334155", zerolinecolor="#475569"),
            bgcolor="#1e293b",
            aspectmode="manual",
            aspectratio=dict(
                x=L / max_dim * 2,
                y=W / max_dim * 2,
                z=H / max_dim * 2,
            ),
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        height=540,
        margin=dict(l=0, r=0, t=75, b=0),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#334155", borderwidth=1,
        ),
        font=dict(color="#e2e8f0"),
    )
    return fig


def plot_solver_metrics(stats: Dict, name: str) -> go.Figure:
    METRIC_KEYS = ["PS balance ratio", "FA balance ratio", "Diagonal ratio", "Final score"]
    labels      = ["PS ratio", "FA ratio", "Diag ratio", "Final score"]
    values      = [float(stats[m]) for m in METRIC_KEYS]
    colors = [
        "#22c55e" if v >= 0.97 else ("#f59e0b" if v >= 0.92 else "#ef4444")
        for v in values
    ]
    rt     = stats.get("Runtime (s)", "—")
    placed = stats.get("Containers placed", "—")

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.8,
                  annotation_text="0.92", annotation_font_color="#f59e0b",
                  annotation_position="top right")
    fig.update_layout(
        title=dict(
            text=f"{name}<br><sup>Placed: {placed}  ·  Runtime: {rt} s</sup>",
            font=dict(size=13),
        ),
        xaxis=dict(title="Ratio", range=[0.78, 1.06]),
        height=230,
        margin=dict(l=80, r=70, t=65, b=30),
        showlegend=False,
        **_DARK,
    )
    return fig


def plot_summary_comparison(results: Dict) -> go.Figure:
    METRIC_KEYS = ["PS balance ratio", "FA balance ratio", "Diagonal ratio", "Final score"]
    x_labels    = ["PS ratio", "FA ratio", "Diag ratio", "Final score"]
    palette     = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]

    fig = go.Figure()
    for i, (sname, (_, _, stats)) in enumerate(results.items()):
        values = [float(stats[m]) for m in METRIC_KEYS]
        fig.add_trace(go.Bar(
            name=sname, x=x_labels, y=values,
            marker_color=palette[i % len(palette)],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate=f"<b>{sname}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    lo = min(float(results[s][2][m]) for s in results for m in METRIC_KEYS)
    fig.add_hline(y=0.92, line_dash="dash", line_color="#f59e0b", opacity=0.8,
                  annotation_text="0.92 threshold", annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_layout(
        barmode="group",
        title="Solver comparison",
        yaxis=dict(title="Ratio", range=[max(lo - 0.02, 0.60), 1.06]),
        height=400,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_DARK,
    )
    return fig


def make_gif(manifest, ship, label="", interval_ms=200) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = tmp.name
    Visualizer(manifest, ship_length=ship.length, ship_width=ship.width,
               ship_height=ship.height, hull=ship.cargo_hold.copy()
               ).animate(interval_ms=interval_ms, save_path=path)
    plt.close("all")
    with open(path, "rb") as fh:
        return fh.read()


def make_comparison_gif(m1, s1, lbl1, m2, s2, lbl2, interval_ms=200) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = tmp.name
    ComparisonVisualizer(
        left=(m1, lbl1), right=(m2, lbl2),
        ship_length=s1.length, ship_width=s1.width, ship_height=s1.height,
        hull=s1.cargo_hold.copy(),
    ).animate(interval_ms=interval_ms, save_path=path)
    plt.close("all")
    with open(path, "rb") as fh:
        return fh.read()


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Manifest",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* tighten up the default streamlit padding */
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  /* stat value emphasis */
  .stat-val { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
  .stat-lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
               letter-spacing: 0.06em; }
  /* score colour helpers */
  .good  { color: #34d399 !important; }
  .warn  { color: #fbbf24 !important; }
  .bad   { color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚢  Manifest")
    st.caption("Configure the ship, containers, and algorithms, then click **Run**.")
    st.divider()

    # ── Ship ──────────────────────────────────────────────────
    st.subheader("Ship")
    ship_mode = st.radio("Ship size", ["Pre-built", "Custom"],
                         horizontal=True, label_visibility="collapsed")

    if ship_mode == "Pre-built":
        pb_name = st.selectbox("Select ship", list(PREBUILT.keys()), index=1)
        ship_cfg = dict(PREBUILT[pb_name])
        model_key = ship_cfg.pop("model_key")
        _def20 = ship_cfg.pop("default_20ft")
        _def40 = ship_cfg.pop("default_40ft")
        st.caption(
            f"Length {ship_cfg['length']} · Beam {ship_cfg['max_width']} · "
            f"Height {ship_cfg['height']} · Keel {ship_cfg['base_width']}"
        )
    else:
        length    = st.slider("Length (20 ft positions)", 4, 36, 24, step=2)
        max_width = st.slider("Beam (columns)", 7, 13, 11)
        base_width = st.slider("Keel width", 3, max_width - 2, 6)
        height    = st.slider("Height (tiers)", 4, 9, 7)
        ship_cfg  = dict(length=length, base_width=base_width,
                         max_width=max_width, height=height,
                         width_step=1, max_weight=1_500_000.0)
        model_key = _closest_model_key(ship_cfg)
        _def20, _def40 = 35, 12

    max_weight_kg = st.slider(
        "Max cargo weight (kg)", 100_000, 10_000_000,
        int(ship_cfg["max_weight"]), step=100_000,
        help="Total weight limit. Containers that would push the ship over this limit are left unloaded.",
    )
    ship_cfg["max_weight"] = float(max_weight_kg)

    st.divider()

    # ── Containers ────────────────────────────────────────────
    st.subheader("Containers")
    _cap_ship        = make_ship(ship_cfg)
    _valid_slots     = int(np.sum(_cap_ship.cargo_hold >= 0))
    _sp_n20          = _valid_slots + 200          # space-limited preset: just over physical capacity
    _slider_max_20ft = max(1000, _valid_slots + 500)
    st.caption(f"Ship capacity: ~{_valid_slots} TEU slots (20 ft equivalent)")

    SCENARIO_PRESETS = {
        "Custom":            None,
        "⚖️ Balanced":       dict(n_20ft=_def20,  n_40ft=_def40, w_min=2_000,  w_max=28_000, dist="Uniform"),
        "🏋️ Weight-limited":  dict(n_20ft=150,    n_40ft=60,     w_min=18_000, w_max=30_000, dist="Uniform"),
        "📦 Space-limited":   dict(n_20ft=_sp_n20, n_40ft=0,     w_min=100,    w_max=500,    dist="Uniform"),
        "🎲 Mixed":           dict(n_20ft=200,     n_40ft=70,     w_min=500,    w_max=30_000, dist="Bimodal"),
    }

    scenario = st.selectbox("Scenario preset", list(SCENARIO_PRESETS.keys()), key="scenario")
    preset   = SCENARIO_PRESETS[scenario]
    if preset is not None and st.session_state.get("_last_preset") != scenario:
        st.session_state["_last_preset"] = scenario
        st.session_state["cnt_20ft"] = min(preset["n_20ft"], _slider_max_20ft)
        st.session_state["cnt_40ft"] = preset["n_40ft"]
        st.session_state["wt_dist"]  = preset["dist"]
        st.session_state["wt_min"]   = preset["w_min"]
        st.session_state["wt_max"]   = preset["w_max"]
    elif preset is None:
        st.session_state["_last_preset"] = "Custom"

    n_20ft = st.slider("20 ft containers", 0, _slider_max_20ft, _def20, key="cnt_20ft")
    n_40ft = st.slider("40 ft containers", 0, 500, _def40, key="cnt_40ft")
    if n_20ft + n_40ft == 0:
        st.warning("Add at least one container.")

    dist_type = st.selectbox(
        "Weight distribution",
        ["Uniform", "Normal", "Bimodal", "Heavy-biased"],
        key="wt_dist",
    )
    if dist_type == "Normal":
        w_mean = st.slider("Mean weight (kg)", 1_000, 28_000, 14_000, step=500)
        w_std  = st.slider("Std deviation (kg)", 500, 8_000, 4_000, step=500)
        w_min, w_max = 100, 30_000
    else:
        w_min = st.slider("Min weight (kg)", 100, 20_000, 2_000, step=100, key="wt_min")
        # Clamp stored wt_max if it's now below the new w_min floor
        if "wt_max" in st.session_state and st.session_state["wt_max"] < w_min + 100:
            st.session_state["wt_max"] = w_min + 100
        w_max = st.slider("Max weight (kg)", w_min + 100, 30_000, 28_000, step=100, key="wt_max")
        w_mean = (w_min + w_max) // 2
        w_std  = (w_max - w_min) // 4

    seed = st.number_input("Random seed", 0, 99_999, 42)

    if n_20ft + n_40ft > 0:
        _preview = make_containers(n_20ft, n_40ft, dist_type, w_min, w_max, w_mean, w_std, seed)
        _total_w = sum(c.weight for c in _preview)
        _delta_w = _total_w - max_weight_kg
        st.metric(
            "Total container weight",
            f"{_total_w:,.0f} kg",
            delta=f"{_delta_w:+,.0f} kg vs limit",
            delta_color="inverse",
            help="Exact total weight of this manifest. Red = over the ship's weight limit.",
        )

    st.divider()

    # ── Algorithms ────────────────────────────────────────────
    st.subheader("Algorithms")
    solver_options = ["Greedy", "Beam Search"]
    if _HEURISTICS:
        solver_options.append("Simulated Annealing")
    if _BAYESIAN:
        solver_options.append("Bayesian Opt")
    if _NEURAL:
        model_path = MODELS_DIR / f"neural_ranker_{model_key}.pkl"
        if model_path.exists():
            solver_options.append("Neural Ranker")
        else:
            st.caption(f"⚠ No pre-trained Neural Ranker for {model_key}. Run pretrain_models.py first.")

    if _RL_BAYESIAN:
        rl_model_path = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        if rl_model_path.exists():
            solver_options.append("RL Bayesian")
        else:
            st.caption(f"⚠ No pre-trained RL Bayesian for {model_key}. Run pretrain_models.py first.")

    if _RL_BAYESIAN_SA:
        solver_options.append("RL Bayesian + SA")
        rl_sa_pkl = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        if not rl_sa_pkl.exists():
            st.caption(f"⚠ No pre-trained RL Bayesian for {model_key}. RL+SA will fall back to greedy warm start.")

    selected_solvers = st.multiselect(
        "Solvers to compare", solver_options,
        default=["Greedy", "Beam Search"],
    )

    if "Beam Search" in selected_solvers:
        beam_k = st.slider("Beam width K", 2, 20, 5)
    else:
        beam_k = 5

    if "Simulated Annealing" in selected_solvers or "RL Bayesian + SA" in selected_solvers:
        sa_iters = st.slider("SA iterations", 200, 5_000, 2_000, step=200)
    else:
        sa_iters = 2_000

    if "Bayesian Opt" in selected_solvers:
        n_trials = st.slider("Bayesian trials", 10, 100, 30, step=10)
        st.caption("⏱ ~0.5–2 min depending on trial count.")
    else:
        n_trials = 30

    if "Neural Ranker" in selected_solvers:
        st.caption(f"🧠 Using pre-trained model: **{model_key}**")

    if "RL Bayesian" in selected_solvers:
        st.caption(f"🤖 Using pre-trained RL Bayesian model: **{model_key}**")

    if "RL Bayesian + SA" in selected_solvers:
        rl_sa_pkl_info = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        if rl_sa_pkl_info.exists():
            st.caption(f"🤖 RL+SA: RL Bayesian warm start (**{model_key}**) → SA refinement")
        else:
            st.caption(f"🤖 RL+SA: Greedy warm start → SA refinement (no **{model_key}** pkl)")

    st.divider()

    # ── Display options ───────────────────────────────────────
    st.subheader("Display")
    show_anim = st.toggle("Show loading animations", value=True,
                          help="Generates a GIF per solver (~10–15 s each).")

    st.divider()
    run_btn = st.button("▶  Run solvers", type="primary",
                        use_container_width=True,
                        disabled=(len(selected_solvers) == 0 or n_20ft + n_40ft == 0))

# ── Main area — pre-run ───────────────────────────────────────────────────────

st.header("Cargo Ship Loader", divider="gray")
st.caption(
    "Configure the ship and manifest in the sidebar, then click **▶ Run solvers** "
    "to compare loading algorithms side-by-side."
)

# Always show hull preview and weight distribution preview
prev_col, dist_col = st.columns([3, 2])
with prev_col:
    st.subheader("Hull preview")
    st.plotly_chart(plot_hull_preview(ship_cfg), use_container_width=True)

containers_preview = make_containers(
    n_20ft, n_40ft, dist_type, w_min, w_max, w_mean, w_std, seed
)
with dist_col:
    st.subheader("Container weights")
    n_placed = n_20ft + n_40ft
    st.caption(f"{n_20ft} × 20 ft  +  {n_40ft} × 40 ft  =  {n_placed} total")
    if containers_preview:
        st.plotly_chart(plot_weight_dist(containers_preview), use_container_width=True)

# ── Main area — results ───────────────────────────────────────────────────────

if not run_btn:
    st.stop()

if len(selected_solvers) == 0:
    st.warning("Select at least one solver in the sidebar.")
    st.stop()

# Load neural ranker (from cache)
neural_ranker_model = None
if "Neural Ranker" in selected_solvers and _NEURAL:
    with st.spinner("Loading Neural Ranker model…"):
        neural_ranker_model = load_neural_ranker(model_key, model_key)

rl_bayesian_model = None
if "RL Bayesian" in selected_solvers and _RL_BAYESIAN:
    with st.spinner("Loading RL Bayesian model…"):
        rl_bayesian_model = load_rl_bayesian(model_key, model_key)

# Run all solvers
results: Dict[str, Tuple[List[Dict], CargoShip, Dict]] = {}
containers = make_containers(
    n_20ft, n_40ft, dist_type, w_min, w_max, w_mean, w_std, seed
)

with st.spinner(f"Running {len(selected_solvers)} solver(s)…"):
    for name in selected_solvers:
        ship = make_ship(ship_cfg)
        # Neural ranker needs the containers to use the same IDs — regenerate fresh
        ShippingContainer.reset_id_counter()
        conts = make_containers(n_20ft, n_40ft, dist_type, w_min, w_max,
                                w_mean, w_std, seed)
        manifest, elapsed = run_solver(
            name, ship, conts, beam_k, sa_iters, n_trials,
            neural_ranker_model, rl_bayesian=rl_bayesian_model,
            model_key=model_key,
        )
        stats = collect_stats(manifest, ship, elapsed)
        results[name] = (manifest, ship, stats)

st.success(f"✓ Done — {len(results)} solver(s) completed.")
st.divider()

# ── Summary comparison chart ──────────────────────────────────────────────────
st.subheader("Summary")

if results:
    st.plotly_chart(plot_summary_comparison(results), use_container_width=True)

st.divider()

# ── Per-solver results ────────────────────────────────────────────────────────
st.subheader("Per-solver results")

n_solvers = len(results)
if n_solvers == 1:
    cols = [st.container()]
elif n_solvers == 2:
    cols = list(st.columns(2))
else:
    cols = list(st.columns(min(n_solvers, 3)))

for i, (name, (manifest, ship, stats)) in enumerate(results.items()):
    col = cols[i % len(cols)]
    with col:
        st.markdown(f"### {name}")

        st.plotly_chart(plot_solver_metrics(stats, name), use_container_width=True)
        st.plotly_chart(plot_3d_state(manifest, ship, title=name),
                        use_container_width=True)

        if show_anim:
            with st.spinner(f"Generating {name} animation…"):
                gif_bytes = make_gif(manifest, ship, label=name)
            st.image(gif_bytes, caption=f"{name} — loading sequence",
                     use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Source code on [GitHub](https://github.com/kjhooper/cargo_ship_loader) · "
    "Built with [Streamlit](https://streamlit.io)"
)
