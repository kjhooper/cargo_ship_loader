"""Generate case study benchmark data and GIFs for the demo page.

Case 1 — Well-fitted:   Panamax, 85 containers, generous weight limit.
                        All containers placed; algorithms compete purely on balance.

Case 2 — Volume-bound:  Coastal Feeder, 120 containers, weight limit effectively
                        infinite. More containers than slots — algorithms choose
                        which to leave on the dock.

Case 3 — Weight-bound:  Panamax, 85 heavy containers, tight weight cap.
                        Total manifest weight exceeds limit — algorithms choose
                        which containers to reject.

Usage:
    conda run -n personal python generate_case_studies.py <output_dir>
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

import random
import sys
import time
from pathlib import Path

from models import CargoShip, ShippingContainer
from algorithm import CargoLoader
from visualizer import ComparisonVisualizer

try:
    from solvers import BeamSearchSolver, SimulatedAnnealingSolver
except ImportError:
    BeamSearchSolver = SimulatedAnnealingSolver = None

try:
    from solvers import BayesianOptSolver
except ImportError:
    BayesianOptSolver = None

try:
    from solvers.neural_ranker import NeuralRankerSolver
except ImportError:
    NeuralRankerSolver = None

try:
    from solvers.rl_bayesian import RLBayesianSolver
except ImportError:
    RLBayesianSolver = None

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).parent / "models"

# ── Case study definitions ────────────────────────────────────────────────────

CASES = {
    "case1": {
        "label": "Case 1 — Well-fitted manifest",
        "ship": dict(length=36, base_width=7, max_width=13, height=9,
                     width_step=1, max_weight=3_000_000.0),
        "model_key": "panamax",
        "n_20ft": 60, "n_40ft": 25,
        "w_min": 2_000, "w_max": 28_000,
        "seed": 42,
        # Smaller manifest for GIF legibility
        "gif_n_20ft": 20, "gif_n_40ft": 8, "gif_seed": 42,
    },
    "case2": {
        "label": "Case 2 — Volume-constrained (Coastal Feeder)",
        "ship": dict(length=12, base_width=5, max_width=9, height=5,
                     width_step=1, max_weight=100_000_000.0),
        "model_key": "coastal",
        "n_20ft": 120, "n_40ft": 0,
        "w_min": 1_000, "w_max": 5_000,
        "seed": 42,
        "gif_n_20ft": 120, "gif_n_40ft": 0, "gif_seed": 42,
    },
    "case3": {
        "label": "Case 3 — Weight-constrained manifest",
        "ship": dict(length=36, base_width=7, max_width=13, height=9,
                     width_step=1, max_weight=800_000.0),
        "model_key": "panamax",
        "n_20ft": 60, "n_40ft": 25,
        "w_min": 5_000, "w_max": 20_000,
        "seed": 42,
        "gif_n_20ft": 20, "gif_n_40ft": 8, "gif_seed": 42,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_containers(n_20ft, n_40ft, w_min, w_max, seed):
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    conts = (
        [ShippingContainer(size=1, weight=round(rng.uniform(w_min, w_max), 1))
         for _ in range(n_20ft)]
        + [ShippingContainer(size=2, weight=round(rng.uniform(w_min, w_max), 1))
           for _ in range(n_40ft)]
    )
    rng.shuffle(conts)
    return conts


def get_stats(manifest, ship, elapsed):
    placed = [e for e in manifest if e["placed"]]
    total  = ship.total_weight or 1.0
    p, s   = ship.port_starboard_balance()
    f, a   = ship.fore_aft_balance()
    fp, fs, ap, as_ = ship.quadrant_balance()
    ps   = min(p, s)   / max(p, s)   if max(p, s)   > 0 else 1.0
    fa   = min(f, a)   / max(f, a)   if max(f, a)   > 0 else 1.0
    d1, d2 = fp + as_, fs + ap
    diag = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0
    return {
        "placed":    len(placed),
        "total":     len(manifest),
        "weight_kg": total,
        "ps":        ps,
        "fa":        fa,
        "diag":      diag,
        "final":     (ps + fa + diag) / 3,
        "time":      elapsed,
    }


def load_neural(model_key, ship):
    path = MODELS_DIR / f"neural_ranker_{model_key}.pkl"
    if not path.exists():
        return None
    dummy = CargoShip(length=36, base_width=7, max_width=13, height=9,
                      width_step=1, max_weight=50_000.0)
    solver = NeuralRankerSolver.load_model(dummy, str(path))
    solver.ship    = ship
    solver._fitted = True
    return solver


def load_rl(model_key, ship):
    path = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
    if not path.exists():
        return None
    dummy = CargoShip(length=36, base_width=7, max_width=13, height=9,
                      width_step=1, max_weight=50_000.0)
    solver = RLBayesianSolver.load_model(dummy, str(path))
    solver.ship    = ship
    solver._fitted = True
    return solver


def run_solver(name, ship, containers, model_key):
    t0 = time.perf_counter()
    if name == "Greedy":
        solver = CargoLoader(ship)
    elif name == "Beam Search":
        solver = BeamSearchSolver(ship, beam_width=5)
    elif name == "Simulated Annealing":
        solver = SimulatedAnnealingSolver(ship, n_iterations=2000, seed=42)
    elif name == "Bayesian Opt":
        solver = BayesianOptSolver(ship, n_trials=30, seed=42)
    elif name == "Neural Ranker":
        solver = load_neural(model_key, ship)
        if solver is None:
            return None, 0.0
    elif name == "RL Bayesian":
        solver = load_rl(model_key, ship)
        if solver is None:
            return None, 0.0
    else:
        return None, 0.0
    manifest = solver.load(containers)
    return manifest, time.perf_counter() - t0


# ── Run benchmarks ────────────────────────────────────────────────────────────

SOLVER_NAMES = [
    "Greedy", "Beam Search", "Simulated Annealing",
    "Bayesian Opt", "Neural Ranker", "RL Bayesian",
]

all_results = {}   # case_key -> solver_name -> stats dict

for case_key, case in CASES.items():
    print(f"\n{'='*65}")
    print(f"  {case['label']}")
    print(f"  Ship: {case['ship']}")
    print(f"  Containers: {case['n_20ft']}×20ft + {case['n_40ft']}×40ft, "
          f"w=[{case['w_min']:,}–{case['w_max']:,}] kg, seed={case['seed']}")
    print(f"{'='*65}")

    all_results[case_key] = {}

    for sname in SOLVER_NAMES:
        ShippingContainer.reset_id_counter()
        conts = make_containers(
            case["n_20ft"], case["n_40ft"],
            case["w_min"], case["w_max"], case["seed"]
        )
        ship = CargoShip(**case["ship"])
        try:
            manifest, elapsed = run_solver(sname, ship, conts, case["model_key"])
            if manifest is None:
                print(f"  {sname:<25}: skipped (model not found)")
                continue
            st = get_stats(manifest, ship, elapsed)
            all_results[case_key][sname] = st
            print(f"  {sname:<25}: placed={st['placed']:>3}/{st['total']}, "
                  f"PS={st['ps']:.4f}  FA={st['fa']:.4f}  "
                  f"diag={st['diag']:.4f}  final={st['final']:.4f}  "
                  f"t={st['time']:.2f}s")
        except Exception as exc:
            print(f"  {sname:<25}: ERROR — {exc}")


# ── Print machine-readable summary ───────────────────────────────────────────

print("\n\n" + "="*65)
print("SUMMARY (copy into HTML)")
print("="*65)
for case_key, case in CASES.items():
    print(f"\n--- {case['label']} ---")
    for sname, st in all_results.get(case_key, {}).items():
        print(f"  {sname}: placed={st['placed']}/{st['total']} "
              f"PS={st['ps']:.4f} FA={st['fa']:.4f} "
              f"diag={st['diag']:.4f} final={st['final']:.4f} "
              f"weight={st['weight_kg']:,.0f}kg time={st['time']:.2f}s")


# ── GIF generation ────────────────────────────────────────────────────────────
# For each case, generate a side-by-side GIF: Greedy vs best ML/heuristic solver.
# Uses a smaller manifest for GIF legibility.

def best_solver_name(case_key):
    """Return the name of the non-Greedy solver with the highest final score."""
    res = all_results.get(case_key, {})
    candidates = {k: v for k, v in res.items() if k != "Greedy" and v is not None}
    if not candidates:
        return "Beam Search"
    return max(candidates, key=lambda k: candidates[k]["final"])


GIF_INTERVAL_MS = 250

for case_key, case in CASES.items():
    best = best_solver_name(case_key)
    print(f"\nGenerating GIF for {case['label']}: Greedy vs {best} …")

    ship_cfg = case["ship"]
    n20, n40, seed = case["gif_n_20ft"], case["gif_n_40ft"], case["gif_seed"]
    w_min, w_max   = case["w_min"], case["w_max"]

    # Greedy
    ShippingContainer.reset_id_counter()
    conts_g = make_containers(n20, n40, w_min, w_max, seed)
    ship_g  = CargoShip(**ship_cfg)
    manifest_g, _ = run_solver("Greedy", ship_g, conts_g, case["model_key"])

    # Best solver
    ShippingContainer.reset_id_counter()
    conts_b = make_containers(n20, n40, w_min, w_max, seed)
    ship_b  = CargoShip(**ship_cfg)
    manifest_b, _ = run_solver(best, ship_b, conts_b, case["model_key"])

    if manifest_g is None or manifest_b is None:
        print(f"  Skipping — solver returned None")
        continue

    gif_path = OUTPUT_DIR / f"{case_key}-comparison.gif"
    ComparisonVisualizer(
        left=(manifest_g, "Greedy"),
        right=(manifest_b, best),
        ship_length=ship_g.length,
        ship_width=ship_g.width,
        ship_height=ship_g.height,
        hull=CargoShip(**ship_cfg).cargo_hold,
    ).animate(interval_ms=GIF_INTERVAL_MS, save_path=str(gif_path))
    plt.close("all")
    print(f"  → {gif_path}")

print("\nDone.")
