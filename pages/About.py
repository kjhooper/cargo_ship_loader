"""About page — explains the app's purpose, scoring, and the 0.92 balance threshold."""

import streamlit as st

st.header("About — Cargo Ship Loader", divider="gray")

st.markdown("""
This app explores algorithmic approaches to **container ship stowage planning** — the
problem of deciding which container goes where in a ship's cargo hold so that the
vessel is stable, trimmed correctly, and can efficiently unload at each port of call.

It is a research and visualisation tool, not a certified planning system.  All
algorithms, scoring functions, and thresholds are implemented for educational and
exploratory purposes.
""")

st.divider()

# ── What the app does ──────────────────────────────────────────────────────────
st.subheader("What the app does")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**Simulation page**
- Build a ship (pre-built Coastal, Handymax, or Panamax profile, or custom geometry)
- Generate a container manifest (20 ft and 40 ft containers, configurable weight
  distribution, optional multi-stop port routes)
- Run one or more loading algorithms and compare the results side-by-side

**Benchmarks page**
- Pre-computed results across 9 solvers × 3 ship sizes × 6 scenarios × 4 random seeds
- Transfer analysis: how well ML models trained on one ship size generalise to another
- Unloading order analysis: how solvers handle multi-stop routes
- Version comparison: v0.1.0 (balance-only) vs v1.0.0 (with unloading constraint)
""")
with col_b:
    st.markdown("""
**Classic Simulation page**
- The original v0.1.0 version of the app for direct comparison
- Balance-only objective — no port-stop ordering constraint

**Solvers implemented**

| Solver | Type |
|---|---|
| Greedy | Deterministic heuristic |
| Beam Search | Tree search (width *K*) |
| Simulated Annealing | Metaheuristic |
| Bayesian Opt | Black-box optimisation (Optuna) |
| Neural Ranker | Imitation learning (beam-search teacher) |
| RL Bayesian | Reward-weighted regression (SIR) |
| RL Bayesian + SA | RL warm-start → SA refinement |
| Defer | Queue-based greedy with deferral |
| Learned Defer | Defer with learned deferral policy (MLP) |
""")

st.divider()

# ── Scoring ────────────────────────────────────────────────────────────────────
st.subheader("Scoring methodology")

st.markdown("""
Each loaded manifest receives a **final score** — the mean of three balance ratios,
each in [0, 1] where 1.0 is perfect:

| Metric | Formula | What it measures |
|---|---|---|
| PS ratio | min(P, S) / max(P, S) | Port ↔ starboard weight symmetry |
| FA ratio | min(F, A) / max(F, A) | Fore ↔ aft weight symmetry (trim) |
| Diagonal ratio | min(FP+AS, FS+AP) / max(FP+AS, FS+AP) | Cross-diagonal weight symmetry |

Where P = port weight, S = starboard weight, F = fore weight, A = aft weight,
FP/FS/AP/AS = fore-port / fore-starboard / aft-port / aft-starboard quadrant weights.

The **placement objective** used during loading adds physical terms:

```
score = −5·gz_norm − 4·trim_norm − 4·list_norm − 6·diag_norm + 0.5·stacking_bonus [+ k_unload · unload_penalty]
```

- `gz_norm` — normalised centre-of-gravity height (metacentric stability proxy)
- `trim_norm`, `list_norm` — direct fore-aft and port-starboard weight imbalance
- `diag_norm` — diagonal imbalance (the K=6 weight provably prevents corner-heavy loading)
- `unload_penalty` — fraction of containers below a position that have a *later* port stop
  (v1.0.0 only; zero when all containers go to the same destination)
""")

st.divider()

# ── The 0.92 threshold ─────────────────────────────────────────────────────────
st.subheader("The 0.92 balance threshold")

st.info("""
The **0.92** line shown on balance ratio charts is a practical operational threshold:
a loaded condition is considered *acceptable* when the lighter side carries at least
**92 %** of the heavier side's weight (i.e. imbalance ≤ 8 %).
""")

st.markdown("""
#### Derivation and references

**Transverse (port/starboard) list**

For a freely-floating vessel, a transverse weight imbalance *ΔW* over a lever arm *d*
produces a static list angle *θ* satisfying:

> tan θ  =  ΔW · d  /  (Δ · GM_T)

where Δ is displacement and GM_T is the transverse metacentric height.

For a typical loaded container ship (GM_T ≈ 1.5–3 m, beam ≈ 25–35 m), an 8 %
port/starboard imbalance produces a static list of roughly **2–5°**.

The **IMO 2008 Intact Stability Code** (MSC.267(85), Part A, Chapter 2) requires,
among other criteria, that:
- the angle of steady heel due to beam wind shall not exceed **16°**, and
- the initial metacentric height GM₀ shall be ≥ **0.15 m**.

An 8 % weight imbalance sits comfortably within these limits for most vessel classes,
and is consistent with the **≤ 5° list** target that most operators adopt as a
practical loading guideline *(IACS, Common Structural Rules; Lloyd's Register,
Ship Stability — A Guide for Masters, 2023)*.

**Longitudinal (fore/aft) trim**

The same 8 % threshold is applied for trim.  Classification society guidance
(e.g. DNV Rules for Classification — Ships, Pt. 3 Ch. 1, §4) typically requires
trim to remain within the range [−0.5 %, +1.5 %] of ship length for departure
conditions.  For the ship sizes modelled here (12–36 bay lengths), 8 % weight
imbalance translates to a trim well within these bounds.

**Diagonal balance**

No direct IMO requirement governs diagonal loading, but an asymmetric distribution
across the fore-port / aft-starboard diagonal can produce combined list *and* trim
that exceed individual thresholds even when each axis appears balanced.  The
diagonal ratio acts as a second-order safeguard and uses the same 0.92 cut-off for
consistency.

---

#### Summary of relevant standards

| Document | Issuer | Relevance |
|---|---|---|
| [2008 IS Code](https://www.imo.org/en/OurWork/Safety/Pages/IntactStability.aspx) (MSC.267(85)) | IMO | Intact stability criteria for all vessels |
| [CSS Code](https://www.imo.org/en/OurWork/Safety/Pages/CargoSecuring.aspx) (MSC.1/Circ.1533) | IMO | Safe stowage and securing of cargo |
| SOLAS Chapter VI | IMO | Carriage of cargoes — general provisions |
| [Common Structural Rules](https://www.iacs.org.uk/publications/common-structural-rules/) | IACS | Structural loading envelope for bulk carriers and tankers |
| *Ship Stability for Masters and Mates*, Barrass & Derrett (7th ed.) | Butterworth-Heinemann | Standard reference text for stability calculations |

> **Disclaimer:** The 0.92 threshold is a *simplified operational heuristic* used for
> scoring in this research tool.  It is not a certified stability criterion and should
> not be used for real vessel loading decisions.  Actual stability assessment requires
> a class-approved loading computer and compliance with the vessel's approved stability
> booklet.
""")

st.divider()

# ── Hull geometry ──────────────────────────────────────────────────────────────
st.subheader("Hull geometry model")

st.markdown("""
The ships modelled here use a **stepped trapezoidal hull cross-section**:
- The hold is widest at the beam (`max_width` columns)
- Below-waterline tiers are narrower — the hold expands outward by one column per
  side every `width_step` tiers, starting from `base_width` at the keel
- This approximates the double-hull taper of real container ships without full 3D
  hydrostatics

Three pre-built profiles are provided, loosely based on published dimensions:

| Profile | Length | Beam | Height | Approx. TEU | Real-world analogue |
|---|---|---|---|---|---|
| Coastal Feeder | 12 bays | 9 cols | 5 tiers | ~100 TEU | Feeder vessels, coastal trades |
| Handymax | 24 bays | 11 cols | 7 tiers | ~800 TEU | Short-sea / regional trades |
| Panamax | 36 bays | 13 cols | 9 tiers | ~3 000 TEU | Post-Panamax feeder class |

*One "bay" = one 20 ft position (two per 40 ft slot).  Actual TEU capacity depends
on the proportion of 40 ft containers loaded.*
""")

st.divider()

st.caption(
    "Cargo Ship Loader v1.0.0 · "
    "Built with [Streamlit](https://streamlit.io) · "
    "For questions or feedback, open an issue on GitHub."
)
