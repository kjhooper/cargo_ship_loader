"""Cargo Ship Loader — demo entry point.

Usage examples
--------------
# Default: greedy solver, seed 42
conda run -n personal python main.py

# Run beam search with K=5, compare vs greedy side-by-side
conda run -n personal python main.py --solver beam_search --compare

# Simulated annealing with custom iteration budget
conda run -n personal python main.py --solver simulated_annealing --n-iterations 3000

# Bayesian optimisation (requires optuna)
conda run -n personal python main.py --solver bayesian_opt --n-trials 100

# Cross-solver comparison table across 5 seeds
conda run -n personal python main.py --benchmark
"""

import argparse
import random
import sys
import time

from algorithm import CargoLoader
from models import CargoShip, ShippingContainer
from visualizer import ComparisonVisualizer, visualize_hull_3d

# Optional solvers
try:
    from solvers import BeamSearchSolver, SimulatedAnnealingSolver
    _HEURISTIC_SOLVERS = True
except ImportError:
    _HEURISTIC_SOLVERS = False

try:
    from solvers import BayesianOptSolver
    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

try:
    from solvers import NeuralRankerSolver
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PANAMAX_PARAMS = dict(
    length=36,
    base_width=7,
    max_width=13,
    height=9,
    width_step=1,
    max_weight=3_000_000.0,
)


def make_panamax_ship() -> CargoShip:
    """Panamax-class ship with realistic hull geometry.

    length=36  : 18 real 40ft bays × 2 = 36 20ft positions
    base_width=7: ~17m beam at keel ÷ 2.44m per TEU ≈ 7 columns
    max_width=13: Panamax beam ~32m ÷ 2.44m ≈ 13 columns
    height=9    : 4 tiers below deck + 5 above
    width_step=1: hull widens every tier (smooth expansion)
    """
    return CargoShip(**PANAMAX_PARAMS)


def generate_containers(
    n_20ft: int,
    n_40ft: int,
    weight_min: float,
    weight_max: float,
    seed: int = 42,
) -> list:
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    containers = []
    for _ in range(n_20ft):
        w = round(rng.uniform(weight_min, weight_max), 1)
        containers.append(ShippingContainer(size=1, weight=w))
    for _ in range(n_40ft):
        w = round(rng.uniform(weight_min, weight_max), 1)
        containers.append(ShippingContainer(size=2, weight=w))
    rng.shuffle(containers)
    return containers


def print_manifest(manifest: list, ship: CargoShip, label: str = "") -> None:
    placed   = [e for e in manifest if e["placed"]]
    unplaced = [e for e in manifest if not e["placed"]]

    col_w   = 70
    heading = f"CARGO MANIFEST{f'  [{label}]' if label else ''}"
    print(f"\n{'=' * col_w}")
    print(f"{heading:^{col_w}}")
    print(f"{'=' * col_w}")
    header = f"{'ID':>4}  {'Size':>4}  {'Weight':>8}  {'Bay':>4}  {'Col':>4}  {'Tier':>4}  {'Slot':>4}"
    print(header)
    print("-" * col_w)
    for e in placed:
        print(
            f"{e['container_id']:>4}  {e['size']:>4}  {e['weight']:>8.1f}  "
            f"{e['bay']:>4}  {e['col']:>4}  {e['tier']:>4}  {e['slot']:>4}"
        )

    if unplaced:
        print(f"\n  Unplaced containers ({len(unplaced)}):")
        for e in unplaced:
            ft = "20ft" if e["size"] == 1 else "40ft"
            print(f"    ID={e['container_id']}  {ft}  weight={e['weight']:.1f}")

    port_w, stbd_w = ship.port_starboard_balance()
    fore_w, aft_w  = ship.fore_aft_balance()
    total = ship.total_weight or 1.0

    ps_ratio = (
        min(port_w, stbd_w) / max(port_w, stbd_w)
        if max(port_w, stbd_w) > 0 else 1.0
    )
    fa_ratio = (
        min(fore_w, aft_w) / max(fore_w, aft_w)
        if max(fore_w, aft_w) > 0 else 1.0
    )

    print(f"\n{'=' * col_w}")
    print(f"  Containers placed   : {len(placed)} / {len(manifest)}")
    print(f"  Total weight loaded : {ship.total_weight:.1f} kg")
    print(f"  Port weight         : {port_w:.1f}  ({100 * port_w / total:.1f}%)")
    print(f"  Starboard weight    : {stbd_w:.1f}  ({100 * stbd_w / total:.1f}%)")
    print(f"  Fore weight         : {fore_w:.1f}  ({100 * fore_w / total:.1f}%)")
    print(f"  Aft weight          : {aft_w:.1f}  ({100 * aft_w / total:.1f}%)")
    print(f"  Port/Stbd ratio     : {ps_ratio:.3f}  (1.000 = perfect)")
    print(f"  Fore/Aft ratio      : {fa_ratio:.3f}  (1.000 = perfect)")
    print(f"{'=' * col_w}\n")


def _build_solver(name: str, ship: CargoShip, args: argparse.Namespace):
    """Instantiate the requested solver."""
    if name == "greedy":
        return CargoLoader(ship)

    if name == "beam_search":
        if not _HEURISTIC_SOLVERS:
            sys.exit("solvers package not found")
        return BeamSearchSolver(ship, beam_width=args.beam_width)

    if name == "simulated_annealing":
        if not _HEURISTIC_SOLVERS:
            sys.exit("solvers package not found")
        return SimulatedAnnealingSolver(ship, n_iterations=args.n_iterations, seed=args.seed)

    if name == "bayesian_opt":
        if not _BAYESIAN_AVAILABLE:
            sys.exit("BayesianOptSolver requires optuna: pip install optuna")
        return BayesianOptSolver(ship, n_trials=args.n_trials, seed=args.seed)

    if name == "neural_ranker":
        if not _NEURAL_AVAILABLE:
            sys.exit("NeuralRankerSolver requires scikit-learn: pip install scikit-learn")
        solver = NeuralRankerSolver(ship)
        print("  [neural_ranker] Training MLP (this may take a minute)…")
        solver.fit(n_episodes=200, beam_width=5, seed=args.seed,
                   ship_params=PANAMAX_PARAMS)
        return solver

    sys.exit(f"Unknown solver: {name}")


# ---------------------------------------------------------------------------
# Benchmark across seeds
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [42, 99, 7, 1234, 5678]

_SOLVER_SPECS = [
    ("greedy",               {}),
    ("beam_search(K=5)",     {"beam_width": 5}),
    ("beam_search(K=10)",    {"beam_width": 10}),
    ("simulated_annealing",  {"n_iterations": 2000, "seed": None}),
]


def run_benchmark() -> None:
    """Print a cross-solver comparison table: balance ratios + runtime."""
    header_cols = ["Solver"] + [f"seed={s}" for s in BENCHMARK_SEEDS] + ["mean", "time(s)"]
    col_w = [24] + [10] * len(BENCHMARK_SEEDS) + [8, 8]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print("\n" + "=" * 90)
    print(" Cross-solver benchmark  —  final_score() = mean(PS, FA, diag) ratios")
    print("=" * 90)
    print(fmt_row(header_cols))
    print("-" * 90)

    for spec_name, spec_kwargs in _SOLVER_SPECS:
        scores = []
        t0 = time.perf_counter()
        for seed in BENCHMARK_SEEDS:
            ship = make_panamax_ship()
            containers = generate_containers(60, 25, 2000.0, 28000.0, seed=seed)

            if spec_name.startswith("greedy"):
                solver = CargoLoader(ship)
            elif spec_name.startswith("beam_search"):
                if not _HEURISTIC_SOLVERS:
                    scores.append(float("nan"))
                    continue
                solver = BeamSearchSolver(ship, **spec_kwargs)
            elif spec_name.startswith("simulated_annealing"):
                if not _HEURISTIC_SOLVERS:
                    scores.append(float("nan"))
                    continue
                kw = dict(spec_kwargs)
                kw["seed"] = seed
                solver = SimulatedAnnealingSolver(ship, **kw)
            else:
                scores.append(float("nan"))
                continue

            solver.load(containers)
            scores.append(solver.final_score())

        elapsed = time.perf_counter() - t0
        mean_score = sum(s for s in scores if s == s) / max(1, len(scores))
        row = [spec_name] + [f"{s:.4f}" for s in scores] + [f"{mean_score:.4f}", f"{elapsed:.2f}"]
        print(fmt_row(row))

    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cargo ship container loader — multi-solver demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solver",
        choices=["greedy", "beam_search", "simulated_annealing", "bayesian_opt", "neural_ranker"],
        default="greedy",
        help="Solver to use (default: greedy)",
    )
    parser.add_argument("--compare", action="store_true",
                        help="Compare chosen solver vs greedy in ComparisonVisualizer")
    parser.add_argument("--benchmark", action="store_true",
                        help="Print cross-solver comparison table and exit")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--beam-width",   type=int,   default=5,
                        help="Beam width K for beam_search (default: 5)")
    parser.add_argument("--n-trials",     type=int,   default=50,
                        help="Optuna trials for bayesian_opt (default: 50)")
    parser.add_argument("--n-iterations", type=int,   default=2000,
                        help="SA iterations for simulated_annealing (default: 2000)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualisation (useful in CI)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    # --- 3D hull geometry (unless suppressed) ---
    if not args.no_viz:
        ship_hull = make_panamax_ship()
        visualize_hull_3d(
            ship_hull.cargo_hold,
            ship_hull.length,
            ship_hull.width,
            ship_hull.height,
            title="Panamax Hull — Empty Cargo Hold (3D)",
        )

    containers = generate_containers(
        n_20ft=60, n_40ft=25, weight_min=2000.0, weight_max=28000.0,
        seed=args.seed,
    )

    # --- Primary solver ---
    ship = make_panamax_ship()
    solver = _build_solver(args.solver, ship, args)
    t0 = time.perf_counter()
    manifest = solver.load(containers)
    elapsed = time.perf_counter() - t0

    label = f"{args.solver} — seed {args.seed}  ({elapsed:.2f}s)"
    print_manifest(manifest, ship, label=label)
    print(f"  final_score() = {solver.final_score():.4f}")

    if args.compare or args.solver == "greedy":
        # --- Greedy reference run (second seed for standalone greedy demo) ---
        if args.solver == "greedy" and not args.compare:
            # Original demo: show two seeds side-by-side
            seed_b = 99
            ship_b = make_panamax_ship()
            containers_b = generate_containers(60, 25, 2000.0, 28000.0, seed=seed_b)
            manifest_b = CargoLoader(ship_b).load(containers_b)
            print_manifest(manifest_b, ship_b, label=f"Greedy — seed {seed_b}")

            if not args.no_viz:
                viz = ComparisonVisualizer(
                    left=(manifest, f"Greedy — seed {args.seed}"),
                    right=(manifest_b, f"Greedy — seed {seed_b}"),
                    ship_length=ship.length,
                    ship_width=ship.width,
                    ship_height=ship.height,
                    hull=ship.cargo_hold,
                )
                viz.animate(interval_ms=400)
        else:
            # Compare chosen solver against greedy on the same containers
            ship_g = make_panamax_ship()
            ShippingContainer.reset_id_counter()
            containers_g = generate_containers(60, 25, 2000.0, 28000.0, seed=args.seed)
            manifest_g = CargoLoader(ship_g).load(containers_g)
            print_manifest(manifest_g, ship_g, label=f"Greedy — seed {args.seed}")

            if not args.no_viz:
                viz = ComparisonVisualizer(
                    left=(manifest_g, f"Greedy — seed {args.seed}"),
                    right=(manifest, label),
                    ship_length=ship.length,
                    ship_width=ship.width,
                    ship_height=ship.height,
                    hull=ship.cargo_hold,
                )
                viz.animate(interval_ms=400)


if __name__ == "__main__":
    main()
