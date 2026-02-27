"""Generate demo assets for the cargo ship loader website.

Outputs (all written to OUTPUT_DIR):
  hull.png           — 3-D voxel render of empty Panamax hull
  loading-greedy.gif — animated greedy loading sequence
  comparison.gif     — side-by-side greedy vs beam search

Also prints a benchmark table to stdout.

Usage:
  conda run -n personal python generate_demo_assets.py <output_dir>
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None   # no-op so visualize_hull_3d doesn't block

import random
import sys
import time
from pathlib import Path

from models import CargoShip, ShippingContainer
from algorithm import CargoLoader
from solvers import BeamSearchSolver, SimulatedAnnealingSolver
from visualizer import Visualizer, ComparisonVisualizer, visualize_hull_3d

OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANAMAX = dict(
    length=36, base_width=7, max_width=13,
    height=9, width_step=1, max_weight=50_000.0,
)
# Smaller manifest keeps GIFs fast to generate and watch
DEMO_N_20FT = 20
DEMO_N_40FT = 8
DEMO_SEED   = 42
GIF_INTERVAL_MS = 220


def make_ship() -> CargoShip:
    return CargoShip(**PANAMAX)


def make_containers(n_20ft=DEMO_N_20FT, n_40ft=DEMO_N_40FT, seed=DEMO_SEED):
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    containers = []
    for _ in range(n_20ft):
        containers.append(ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1)))
    for _ in range(n_40ft):
        containers.append(ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1)))
    rng.shuffle(containers)
    return containers


def balance_ratios(ship):
    p, s = ship.port_starboard_balance()
    f, a = ship.fore_aft_balance()
    ps = min(p, s) / max(p, s) if max(p, s) > 0 else 1.0
    fa = min(f, a) / max(f, a) if max(f, a) > 0 else 1.0
    return ps, fa


# ── 1. Hull PNG ──────────────────────────────────────────────────────────────
print("1/3  hull.png …")
ship = make_ship()
visualize_hull_3d(
    ship.cargo_hold, ship.length, ship.width, ship.height,
    title="Panamax Hull — Empty Cargo Hold (3-D)",
)
fig = plt.gcf()
fig.patch.set_facecolor("#0f172a")
fig.savefig(OUTPUT_DIR / "hull.png", dpi=110, bbox_inches="tight", facecolor="#0f172a")
plt.close("all")
print(f"     → {OUTPUT_DIR / 'hull.png'}")


# ── 2. Greedy GIF ────────────────────────────────────────────────────────────
print("2/3  loading-greedy.gif …")
ship_g = make_ship()
containers = make_containers()
manifest_g = CargoLoader(ship_g).load(containers)
ps_g, fa_g = balance_ratios(ship_g)

Visualizer(
    manifest_g,
    ship_length=ship_g.length,
    ship_width=ship_g.width,
    ship_height=ship_g.height,
    hull=make_ship().cargo_hold,
).animate(interval_ms=GIF_INTERVAL_MS, save_path=str(OUTPUT_DIR / "loading-greedy.gif"))
plt.close("all")
print(f"     → {OUTPUT_DIR / 'loading-greedy.gif'}  (PS={ps_g:.3f}  FA={fa_g:.3f})")


# ── 3. Comparison GIF ────────────────────────────────────────────────────────
print("3/3  comparison.gif …")
ship_b = make_ship()
ShippingContainer.reset_id_counter()
containers_b = make_containers()
manifest_b = BeamSearchSolver(ship_b, beam_width=5).load(containers_b)
ps_b, fa_b = balance_ratios(ship_b)

ComparisonVisualizer(
    left=(manifest_g,  "Greedy"),
    right=(manifest_b, "Beam Search  K=5"),
    ship_length=ship_g.length,
    ship_width=ship_g.width,
    ship_height=ship_g.height,
    hull=make_ship().cargo_hold,
).animate(interval_ms=GIF_INTERVAL_MS, save_path=str(OUTPUT_DIR / "comparison.gif"))
plt.close("all")
print(f"     → {OUTPUT_DIR / 'comparison.gif'}  (BS PS={ps_b:.3f}  FA={fa_b:.3f})")


# ── 4. Benchmark table (full-size manifest, 5 seeds) ────────────────────────
print("\nBenchmark — 60 20ft + 25 40ft containers, seeds [42, 99, 7, 1234, 5678]\n")

BENCH_SEEDS = [42, 99, 7, 1234, 5678]

SOLVERS = [
    ("Greedy",                lambda ship: CargoLoader(ship)),
    ("Beam Search  K=5",      lambda ship: BeamSearchSolver(ship, beam_width=5)),
    ("Simulated Annealing",   lambda ship: SimulatedAnnealingSolver(ship, n_iterations=2000, seed=42)),
]

header = f"{'Solver':<24}" + "".join(f"  seed={s:>4}" for s in BENCH_SEEDS) + "   mean    time(s)"
print(header)
print("-" * len(header))

results = {}
for name, factory in SOLVERS:
    ps_scores, fa_scores = [], []
    t0 = time.perf_counter()
    for seed in BENCH_SEEDS:
        ship = make_ship()
        rng = random.Random(seed)
        ShippingContainer.reset_id_counter()
        conts = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1)) for _ in range(60)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1)) for _ in range(25)]
        )
        rng.shuffle(conts)
        solver = factory(ship)
        solver.load(conts)
        ps, fa = balance_ratios(ship)
        ps_scores.append(ps)
        fa_scores.append(fa)
    elapsed = time.perf_counter() - t0

    mean_ps = sum(ps_scores) / len(ps_scores)
    mean_fa = sum(fa_scores) / len(fa_scores)
    scores_str = "".join(f"  {ps:.3f}/{fa:.3f}" for ps, fa in zip(ps_scores, fa_scores))
    print(f"{name:<24}{scores_str}   {mean_ps:.3f}/{mean_fa:.3f}   {elapsed:.1f}s")
    results[name] = (ps_scores, fa_scores, elapsed)

print("\n(PS/FA format: port-starboard ratio / fore-aft ratio)")
print("Done.")
