"""Comprehensive benchmarking framework for the Cargo Ship Loader.

Usage
-----
# Full run (all ships × scenarios × seeds + transfer tests)
conda run -n personal python benchmark.py

# Quick smoke test
conda run -n personal python benchmark.py --ships coastal --scenarios balanced --no-transfer

# Subset
conda run -n personal python benchmark.py --ships coastal panamax --scenarios balanced mixed
conda run -n personal python benchmark.py --seeds 42 99
conda run -n personal python benchmark.py --no-color
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from algorithm import BaseSolver, CargoLoader
from models import CargoShip, ShippingContainer

# ---------------------------------------------------------------------------
# Optional solver imports
# ---------------------------------------------------------------------------

try:
    from solvers import BeamSearchSolver, SimulatedAnnealingSolver
    _HEURISTIC_OK = True
except ImportError:
    _HEURISTIC_OK = False

try:
    from solvers import BayesianOptSolver
    _BAYESIAN_OK = True
except ImportError:
    _BAYESIAN_OK = False

try:
    from solvers import NeuralRankerSolver
    _NEURAL_OK = True
except ImportError:
    _NEURAL_OK = False

try:
    from solvers import RLBayesianSolver
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    from solvers import RLBayesianSASolver
    _RL_SA_OK = True
except ImportError:
    _RL_SA_OK = False

try:
    from solvers import LearnedDeferSolver, DeferSolver
    _DEFER_OK = True
except ImportError:
    _DEFER_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHIP_CONFIGS = {
    "coastal":  dict(length=12, base_width=5, max_width=9,  height=5, width_step=1, max_weight=500_000.0),
    "handymax": dict(length=24, base_width=6, max_width=11, height=7, width_step=1, max_weight=1_500_000.0),
    "panamax":  dict(length=36, base_width=7, max_width=13, height=9, width_step=1, max_weight=3_000_000.0),
}

SHIP_DEFAULTS = {
    "coastal":  dict(default_20ft=12,  default_40ft=4),
    "handymax": dict(default_20ft=35,  default_40ft=12),
    "panamax":  dict(default_20ft=60,  default_40ft=25),
}

SOLVERS   = ["greedy", "beam_search", "simulated_annealing", "bayesian_opt", "neural_ranker", "rl_bayesian", "rl_bayesian_sa", "defer", "learned_defer"]
SCENARIOS = ["balanced", "weight_limited", "space_limited", "mixed"]
SEEDS     = [42, 99, 7]

# Scenarios specifically designed to stress-test the unloading order constraint.
# Each assigns containers to 2–5 port stops with deliberate weight distributions.
UNLOADING_SCENARIOS = [
    "stop_early_heavy",   # stop-1 containers are heavy → direct conflict with weight-first sort
    "stop_early_light",   # stop-1 containers are light → weight-first naturally satisfies order
    "stop_uniform_3",     # 3 stops, uniform weights   → no weight pattern to exploit
    "stop_many",          # 5 stops, uniform weights   → high-complexity ordering
]

MODELS_DIR = Path(__file__).parent / "models"

# Increment when algorithm logic changes and all benchmarks should re-run
ALGO_VERSION = "0.3.0"

# ML solvers that support cross-ship transfer tests
ML_SOLVERS = ["neural_ranker", "rl_bayesian", "learned_defer"]

# All 9 ship × model combos for transfer tests
TRANSFER_COMBOS = [
    ("coastal",  "coastal"),
    ("coastal",  "handymax"),
    ("coastal",  "panamax"),
    ("handymax", "coastal"),
    ("handymax", "handymax"),
    ("handymax", "panamax"),
    ("panamax",  "coastal"),
    ("panamax",  "handymax"),
    ("panamax",  "panamax"),
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    ship_key:    str
    scenario:    str
    solver_name: str
    model_key:   str   # which .pkl to load (may differ from ship_key in transfer tests)
    seed:        int


@dataclass
class BenchmarkResult:
    config:          BenchmarkConfig
    ps_ratio:        float = 0.0
    fa_ratio:        float = 0.0
    diag_ratio:      float = 0.0
    final_score:     float = 0.0
    placed:          int   = 0
    total:           int   = 0
    weight_loaded:   float = 0.0   # total cargo weight placed (kg)
    cog_height_norm: float = 0.0   # normalised centre-of-gravity height [0, 1]
    unloading_score:  float = 1.0   # fraction of stacked pairs in correct order [0, 1]
    rehandle_count:   int   = 0     # total moves needed across all stops
    post_stop_balance: List[Dict] = field(default_factory=list)
    runtime_s:        float = 0.0
    error:            Optional[str] = None

    def to_dict(self) -> dict:
        avg_rehandles = (
            round(self.rehandle_count / self.placed, 4) if self.placed else 0.0
        )
        return {
            "algo_version":    ALGO_VERSION,
            "ship_key":        self.config.ship_key,
            "scenario":        self.config.scenario,
            "solver_name":     self.config.solver_name,
            "model_key":       self.config.model_key,
            "seed":            self.config.seed,
            "is_transfer":     self.config.ship_key != self.config.model_key,
            "ps_ratio":        round(self.ps_ratio,        6),
            "fa_ratio":        round(self.fa_ratio,        6),
            "diag_ratio":      round(self.diag_ratio,      6),
            "final_score":     round(self.final_score,     6),
            "placed":          self.placed,
            "total":           self.total,
            "pct_placed":      round(100.0 * self.placed / self.total, 2) if self.total else 0.0,
            "weight_loaded":   round(self.weight_loaded,   1),
            "cog_height_norm": round(self.cog_height_norm, 6),
            "unloading_score": round(self.unloading_score, 6),
            "rehandle_count":    self.rehandle_count,
            "avg_rehandles":     avg_rehandles,
            "post_stop_balance": self.post_stop_balance,
            "runtime_s":         round(self.runtime_s,       4),
            "error":             self.error,
        }


# ---------------------------------------------------------------------------
# Internal exception for graceful solver skips
# ---------------------------------------------------------------------------

class _SolverSkip(Exception):
    """Raised by _create_solver when the solver cannot run; carries error code."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _load_existing_results(path: Path) -> dict:
    """Load benchmark_results.json if present; return empty section lists otherwise.

    Backfills 'algo_version': '0.2.0' on any record that predates versioning so that
    existing benchmark data is preserved without re-running.
    """
    import json
    if not path.exists() or path.stat().st_size == 0:
        return {"standard": [], "transfer": [], "unloading": []}
    with open(path) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            print(f"Warning: {path} is corrupt — starting fresh.", flush=True)
            return {"standard": [], "transfer": [], "unloading": []}

    def _backfill(records: list) -> list:
        for r in records:
            r.setdefault("algo_version", "0.2.0")
        return records

    return {
        "standard":  _backfill(data.get("standard",  [])),
        "transfer":  _backfill(data.get("transfer",  [])),
        "unloading": _backfill(data.get("unloading", [])),
    }


def _done_keys(records: list) -> set:
    """Return the set of config keys already completed for the current ALGO_VERSION."""
    return {
        (r["ship_key"], r["scenario"], r["solver_name"], r["model_key"], r["seed"])
        for r in records
        if r.get("algo_version") == ALGO_VERSION
    }


# ---------------------------------------------------------------------------
# Ship & container factories
# ---------------------------------------------------------------------------

def _make_ship(ship_key: str) -> CargoShip:
    return CargoShip(**SHIP_CONFIGS[ship_key])


def _scenario_containers(
    ship_key: str, scenario: str, seed: int
) -> List[ShippingContainer]:
    """Generate containers for a given ship/scenario/seed combination."""
    rng = random.Random(seed)
    defaults = SHIP_DEFAULTS[ship_key]
    n20_def  = defaults["default_20ft"]
    n40_def  = defaults["default_40ft"]

    if scenario == "balanced":
        n20, n40    = n20_def, n40_def
        w_min, w_max = 2_000.0, 28_000.0
        dist = "uniform"

    elif scenario == "weight_limited":
        # Many heavy containers — weight cap will bind before space runs out
        n20, n40    = 4 * n20_def, 2 * n40_def
        w_min, w_max = 18_000.0, 30_000.0
        dist = "uniform"

    elif scenario == "space_limited":
        # Fill every valid slot with ultra-light containers (weight cap irrelevant)
        ship_tmp = _make_ship(ship_key)
        valid_slots = int(np.sum(ship_tmp.cargo_hold >= 0))
        n20, n40    = valid_slots, 0
        w_min, w_max = 100.0, 500.0
        dist = "uniform"

    elif scenario == "mixed":
        # Bimodal weight distribution; both constraints compete
        n20, n40    = 3 * n20_def, n40_def
        w_min, w_max = 500.0, 30_000.0
        dist = "bimodal"

    # ── Unloading-order stress scenarios ───────────────────────────────────────
    # These scenarios assign containers to port stops with deliberate weight
    # distributions to expose conflicts between weight-optimal loading and
    # correct port-stop unloading order.

    elif scenario == "stop_early_heavy":
        # 2 stops.  Stop-1 containers are heavy, stop-2 are light.
        # Maximum conflict: greedy heavy-first buries stop-1 containers at the
        # bottom, but they must come off first — requiring many rehandles.
        n20, n40     = n20_def, n40_def
        ShippingContainer.reset_id_counter()
        containers: List[ShippingContainer] = []
        for _ in range(n20):
            stop = rng.randint(1, 2)
            w = round(rng.uniform(20_000.0, 28_000.0) if stop == 1
                      else rng.uniform(2_000.0, 8_000.0), 1)
            containers.append(ShippingContainer(size=1, weight=w, facility=stop))
        for _ in range(n40):
            stop = rng.randint(1, 2)
            w = round(rng.uniform(20_000.0, 28_000.0) if stop == 1
                      else rng.uniform(2_000.0, 8_000.0), 1)
            containers.append(ShippingContainer(size=2, weight=w, facility=stop))
        rng.shuffle(containers)
        return containers

    elif scenario == "stop_early_light":
        # 2 stops.  Stop-1 containers are light, stop-2 are heavy.
        # Natural alignment: greedy heavy-first places stop-2 containers deep
        # and stop-1 containers on top — unloading order is satisfied for free.
        # Measures how much headroom exists vs the hard case.
        n20, n40     = n20_def, n40_def
        ShippingContainer.reset_id_counter()
        containers = []
        for _ in range(n20):
            stop = rng.randint(1, 2)
            w = round(rng.uniform(2_000.0, 8_000.0) if stop == 1
                      else rng.uniform(20_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=1, weight=w, facility=stop))
        for _ in range(n40):
            stop = rng.randint(1, 2)
            w = round(rng.uniform(2_000.0, 8_000.0) if stop == 1
                      else rng.uniform(20_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=2, weight=w, facility=stop))
        rng.shuffle(containers)
        return containers

    elif scenario == "stop_uniform_3":
        # 3 stops, uniform weight distribution across all stops.
        # No systematic weight pattern to exploit; tests the unloading-order
        # penalty in isolation without a weight-based shortcut.
        n20, n40     = n20_def, n40_def
        ShippingContainer.reset_id_counter()
        containers = []
        for _ in range(n20):
            stop = rng.randint(1, 3)
            w = round(rng.uniform(2_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=1, weight=w, facility=stop))
        for _ in range(n40):
            stop = rng.randint(1, 3)
            w = round(rng.uniform(2_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=2, weight=w, facility=stop))
        rng.shuffle(containers)
        return containers

    elif scenario == "stop_many":
        # 5 stops, uniform weight distribution.
        # High-complexity ordering challenge: the solver must resolve a 5-way
        # interleaving of port-stop constraints alongside balance.
        n20, n40     = n20_def, n40_def
        ShippingContainer.reset_id_counter()
        containers = []
        for _ in range(n20):
            stop = rng.randint(1, 5)
            w = round(rng.uniform(2_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=1, weight=w, facility=stop))
        for _ in range(n40):
            stop = rng.randint(1, 5)
            w = round(rng.uniform(2_000.0, 28_000.0), 1)
            containers.append(ShippingContainer(size=2, weight=w, facility=stop))
        rng.shuffle(containers)
        return containers

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    ShippingContainer.reset_id_counter()
    containers: List[ShippingContainer] = []

    for _ in range(n20):
        w = round(_sample_weight(rng, dist, w_min, w_max), 1)
        containers.append(ShippingContainer(size=1, weight=w))
    for _ in range(n40):
        w = round(_sample_weight(rng, dist, w_min, w_max), 1)
        containers.append(ShippingContainer(size=2, weight=w))

    rng.shuffle(containers)
    return containers


def _sample_weight(rng: random.Random, dist: str, w_min: float, w_max: float) -> float:
    if dist == "uniform":
        return rng.uniform(w_min, w_max)
    elif dist == "bimodal":
        # Exact match to app.py:138-142
        spread = (w_max - w_min) * 0.35
        if rng.random() < 0.5:
            return rng.uniform(w_min, w_min + spread)
        return rng.uniform(w_max - spread, w_max)
    raise ValueError(f"Unknown dist: {dist!r}")


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def _create_solver(solver_name: str, ship: CargoShip, model_key: str):
    """Return a ready-to-use solver.

    Raises _SolverSkip with an error code if the solver cannot run.
    """
    if solver_name == "greedy":
        return CargoLoader(ship)

    if solver_name == "beam_search":
        if not _HEURISTIC_OK:
            raise _SolverSkip("IMPORT_ERROR")
        return BeamSearchSolver(ship, beam_width=5)

    if solver_name == "simulated_annealing":
        if not _HEURISTIC_OK:
            raise _SolverSkip("IMPORT_ERROR")
        return SimulatedAnnealingSolver(ship, n_iterations=1000, seed=42)

    if solver_name == "bayesian_opt":
        if not _BAYESIAN_OK:
            raise _SolverSkip("IMPORT_ERROR")
        return BayesianOptSolver(ship, n_trials=30, seed=42)

    if solver_name == "neural_ranker":
        if not _NEURAL_OK:
            raise _SolverSkip("IMPORT_ERROR")
        path = MODELS_DIR / f"neural_ranker_{model_key}.pkl"
        if not path.exists():
            raise _SolverSkip("MISSING_PKL")
        dummy = CargoShip(length=36, base_width=7, max_width=13,
                          height=9, width_step=1, max_weight=50_000.0)
        solver = NeuralRankerSolver.load_model(dummy, str(path))
        solver.ship = ship
        return solver

    if solver_name == "rl_bayesian":
        if not _RL_OK:
            raise _SolverSkip("IMPORT_ERROR")
        path = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        if not path.exists():
            raise _SolverSkip("MISSING_PKL")
        dummy = CargoShip(length=36, base_width=7, max_width=13,
                          height=9, width_step=1, max_weight=50_000.0)
        solver = RLBayesianSolver.load_model(dummy, str(path))
        solver.ship = ship
        return solver

    if solver_name == "rl_bayesian_sa":
        if not _RL_SA_OK:
            raise _SolverSkip("IMPORT_ERROR")
        path = MODELS_DIR / f"rl_bayesian_{model_key}.pkl"
        # rl_bayesian_sa runs even without a pkl (greedy warm start fallback)
        return RLBayesianSASolver(
            ship,
            n_iterations=1000,
            seed=42,
            model_path=str(path) if path.exists() else None,
        )

    if solver_name == "defer":
        if not _DEFER_OK:
            raise _SolverSkip("IMPORT_ERROR")
        return DeferSolver(ship)

    if solver_name == "learned_defer":
        if not _DEFER_OK:
            raise _SolverSkip("IMPORT_ERROR")
        path = MODELS_DIR / f"learned_defer_{model_key}.pkl"
        if not path.exists():
            raise _SolverSkip("MISSING_PKL")
        dummy = CargoShip(length=36, base_width=7, max_width=13,
                          height=9, width_step=1, max_weight=50_000.0)
        solver = LearnedDeferSolver.load_model(dummy, str(path))
        solver.ship = ship
        return solver

    raise ValueError(f"Unknown solver: {solver_name!r}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute one benchmark trial; never raises — errors go into result.error."""
    result = BenchmarkResult(config=config)

    try:
        ship       = _make_ship(config.ship_key)
        containers = _scenario_containers(config.ship_key, config.scenario, config.seed)
        solver     = _create_solver(config.solver_name, ship, config.model_key)

        t0 = time.perf_counter()
        manifest = solver.load(containers)
        result.runtime_s = time.perf_counter() - t0

        # Compute balance ratios directly from the ship state
        port_w, stbd_w = ship.port_starboard_balance()
        fore_w, aft_w  = ship.fore_aft_balance()
        fp, fs, ap, as_ = ship.quadrant_balance()

        max_ps = max(port_w, stbd_w)
        max_fa = max(fore_w, aft_w)
        d1, d2 = fp + as_, fs + ap
        max_d  = max(d1, d2)

        result.ps_ratio   = min(port_w, stbd_w) / max_ps if max_ps > 0 else 1.0
        result.fa_ratio   = min(fore_w, aft_w)  / max_fa if max_fa > 0 else 1.0
        result.diag_ratio = min(d1, d2)          / max_d  if max_d  > 0 else 1.0
        result.final_score = (result.ps_ratio + result.fa_ratio + result.diag_ratio) / 3.0

        placed_entries = [e for e in manifest if e["placed"]]
        result.placed = len(placed_entries)
        result.total  = len(manifest)

        result.weight_loaded = ship.total_weight
        if placed_entries and ship.height > 1 and ship.total_weight > 0:
            gz_sum = sum(e["weight"] * e["tier"] for e in placed_entries)
            result.cog_height_norm = gz_sum / (ship.total_weight * (ship.height - 1))

        result.unloading_score = BaseSolver.unloading_score(manifest)
        result.rehandle_count  = BaseSolver.rehandle_count(manifest)
        result.post_stop_balance = BaseSolver.per_stop_balance(
            manifest, ship.length, ship.width
        )

    except _SolverSkip as exc:
        result.error = exc.code
    except Exception as exc:  # noqa: BLE001
        result.error = f"ERROR:{exc}"

    return result


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

def _print_progress(done: int, total: int, cfg: BenchmarkConfig, result: BenchmarkResult) -> None:
    pct = 100 * done / total
    tag = f"[{done}/{total} {pct:.0f}%]"
    if result.error:
        status = f"SKIP({result.error})"
    else:
        status = f"score={result.final_score:.3f}  {result.runtime_s:.2f}s"
    print(f"  {tag:<14}  {cfg.ship_key:<8}  {cfg.scenario:<15}  "
          f"{cfg.solver_name:<22}  seed={cfg.seed}  {status}", flush=True)


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"

_USE_COLOR = True   # toggled in main() based on TTY / --no-color


def _maybe_color(s: str, color: str) -> str:
    return f"{color}{s}{_RESET}" if _USE_COLOR else s


def _fmt_ratio(v: Optional[float]) -> str:
    if v is None:
        return "  N/A  "
    s = f"{v:.3f}"
    if v >= 0.97:
        return _maybe_color(s, _GREEN)
    if v >= 0.92:
        return _maybe_color(s, _YELLOW)
    return _maybe_color(s, _RED)


# ---------------------------------------------------------------------------
# Standard benchmark table
# ---------------------------------------------------------------------------

def print_standard_table(results: List[BenchmarkResult]) -> None:
    """Print one row per (ship, scenario, solver), mean across seeds."""
    print()
    print("=" * 100)
    print(" STANDARD BENCHMARK — mean across seeds")
    print("=" * 100)

    header = (
        f"{'Ship':<10}  {'Scenario':<16}  {'Solver':<22}  "
        f"{'PS':>7}  {'FA':>7}  {'Diag':>7}  {'Score':>7}  "
        f"{'Placed':>7}  {'Time(s)':>7}"
    )
    print(header)
    print("-" * 100)

    # Group by (ship, scenario, solver)
    groups: dict = {}
    for r in results:
        key = (r.config.ship_key, r.config.scenario, r.config.solver_name, r.config.model_key)
        groups.setdefault(key, []).append(r)

    ship_order     = ["coastal", "handymax", "panamax"]
    scenario_order = ["balanced", "weight_limited", "space_limited", "mixed"]
    solver_order   = SOLVERS

    prev_ship = None
    for ship_key in ship_order:
        for scenario in scenario_order:
            for solver_name in solver_order:
                key = (ship_key, scenario, solver_name, ship_key)
                if key not in groups:
                    continue
                seed_results = groups[key]

                # Filter to non-error results for metric means
                ok  = [r for r in seed_results if not r.error]
                skipped_all = all(r.error for r in seed_results)

                if ship_key != prev_ship:
                    if prev_ship is not None:
                        print()
                    prev_ship = ship_key

                if skipped_all:
                    err_code = seed_results[0].error if seed_results else "SKIP"
                    print(
                        f"{ship_key:<10}  {scenario:<16}  {solver_name:<22}  "
                        + "  ".join(["   N/A "] * 5)
                        + f"  ({err_code})"
                    )
                    continue

                mean_ps   = sum(r.ps_ratio   for r in ok) / len(ok)
                mean_fa   = sum(r.fa_ratio   for r in ok) / len(ok)
                mean_diag = sum(r.diag_ratio for r in ok) / len(ok)
                mean_sc   = sum(r.final_score for r in ok) / len(ok)
                mean_time = sum(r.runtime_s  for r in ok) / len(ok)

                total_placed = sum(r.placed for r in ok)
                total_total  = sum(r.total  for r in ok)
                pct_placed   = f"{100 * total_placed / total_total:.0f}%" if total_total else "N/A"

                print(
                    f"{ship_key:<10}  {scenario:<16}  {solver_name:<22}  "
                    f"{_fmt_ratio(mean_ps):>7}  {_fmt_ratio(mean_fa):>7}  "
                    f"{_fmt_ratio(mean_diag):>7}  {_fmt_ratio(mean_sc):>7}  "
                    f"{pct_placed:>7}  {mean_time:>7.2f}"
                )

    print("=" * 100)


# ---------------------------------------------------------------------------
# Transfer benchmark table
# ---------------------------------------------------------------------------

def print_transfer_table(results: List[BenchmarkResult]) -> None:
    """Print cross-ship transfer results, one block per ML solver."""
    for solver_name in ML_SOLVERS:
        solver_results = [r for r in results if r.config.solver_name == solver_name]
        if not solver_results:
            continue

        print()
        print("=" * 80)
        print(f" CROSS-SHIP TRANSFER — {solver_name} (balanced, seeds {SEEDS})")
        print("=" * 80)

        header = (
            f"{'Ship':<10}  {'Model':<10}  "
            f"{'PS':>7}  {'FA':>7}  {'Diag':>7}  {'Score':>7}  {'Match':>6}"
        )
        print(header)
        print("-" * 80)

        for ship_key, model_key in TRANSFER_COMBOS:
            seed_results = [
                r for r in solver_results
                if r.config.ship_key == ship_key and r.config.model_key == model_key
            ]
            if not seed_results:
                continue

            ok = [r for r in seed_results if not r.error]
            if not ok:
                err_code = seed_results[0].error if seed_results else "SKIP"
                print(
                    f"{ship_key:<10}  {model_key:<10}  "
                    + "  ".join(["   N/A "] * 4)
                    + f"  ({err_code})"
                )
                continue

            mean_ps   = sum(r.ps_ratio   for r in ok) / len(ok)
            mean_fa   = sum(r.fa_ratio   for r in ok) / len(ok)
            mean_diag = sum(r.diag_ratio for r in ok) / len(ok)
            mean_sc   = sum(r.final_score for r in ok) / len(ok)

            is_match = ship_key == model_key
            if is_match:
                match_str = _maybe_color("YES ✓", _GREEN)
            else:
                match_str = "no"

            print(
                f"{ship_key:<10}  {model_key:<10}  "
                f"{_fmt_ratio(mean_ps):>7}  {_fmt_ratio(mean_fa):>7}  "
                f"{_fmt_ratio(mean_diag):>7}  {_fmt_ratio(mean_sc):>7}  "
                f"{match_str}"
            )

    print("=" * 80)


# ---------------------------------------------------------------------------
# Unloading benchmark table
# ---------------------------------------------------------------------------

UNLOADING_CASE_LABELS = {
    "stop_early_heavy": "EarlyHeavy (hard)",
    "stop_early_light": "EarlyLight (easy)",
    "stop_uniform_3":   "Uniform-3stop",
    "stop_many":        "Uniform-5stop",
}


def print_unloading_table(results: List[BenchmarkResult]) -> None:
    """Print unloading-focused metrics: unload score and avg rehandles per container."""
    ul_results = [r for r in results
                  if r.config.scenario in UNLOADING_SCENARIOS]
    if not ul_results:
        return

    print()
    print("=" * 110)
    print(" UNLOADING ORDER BENCHMARK — avg rehandles per container  (lower = better)")
    print(" Rehandle = container that must be moved aside to access an earlier-stop container.")
    print(" Same-stop containers above do NOT count (they unload together at that stop).")
    print("=" * 110)

    header = (
        f"{'Ship':<10}  {'Scenario':<22}  {'Solver':<22}  "
        f"{'Unload':>8}  {'Avg-Rhdl':>9}  {'Tot-Rhdl':>9}  {'Balance':>8}  {'Time(s)':>7}"
    )
    print(header)
    print("-" * 110)

    groups: dict = {}
    for r in ul_results:
        key = (r.config.ship_key, r.config.scenario, r.config.solver_name)
        groups.setdefault(key, []).append(r)

    prev_ship = None
    for ship_key in ["coastal", "handymax", "panamax"]:
        for scenario in UNLOADING_SCENARIOS:
            for solver_name in SOLVERS:
                key = (ship_key, scenario, solver_name)
                if key not in groups:
                    continue
                seed_results = groups[key]
                ok = [r for r in seed_results if not r.error]
                skipped_all = all(r.error for r in seed_results)

                if ship_key != prev_ship:
                    if prev_ship is not None:
                        print()
                    prev_ship = ship_key

                label = UNLOADING_CASE_LABELS.get(scenario, scenario)

                if skipped_all:
                    err_code = seed_results[0].error if seed_results else "SKIP"
                    print(f"{ship_key:<10}  {label:<22}  {solver_name:<22}  "
                          + "  ".join(["      N/A"] * 5)
                          + f"  ({err_code})")
                    continue

                mean_ul    = sum(r.unloading_score for r in ok) / len(ok)
                total_rh   = sum(r.rehandle_count  for r in ok)
                total_pl   = sum(r.placed           for r in ok)
                avg_rh     = total_rh / total_pl if total_pl else 0.0
                mean_sc    = sum(r.final_score      for r in ok) / len(ok)
                mean_time  = sum(r.runtime_s        for r in ok) / len(ok)

                ul_str  = _fmt_ratio(mean_ul)
                sc_str  = _fmt_ratio(mean_sc)
                # Colour avg rehandles: green = 0, yellow < 0.5, red >= 0.5
                rh_raw  = f"{avg_rh:.3f}"
                if avg_rh == 0:
                    rh_str = _maybe_color(rh_raw, _GREEN)
                elif avg_rh < 0.5:
                    rh_str = _maybe_color(rh_raw, _YELLOW)
                else:
                    rh_str = _maybe_color(rh_raw, _RED)

                print(
                    f"{ship_key:<10}  {label:<22}  {solver_name:<22}  "
                    f"{ul_str:>8}  {rh_str:>9}  {total_rh:>9}  "
                    f"{sc_str:>8}  {mean_time:>7.2f}"
                )

    print("=" * 110)


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------

def _run_parallel(configs: List[BenchmarkConfig], n_workers: int) -> List[BenchmarkResult]:
    """Run benchmark configs in parallel (or sequentially when n_workers==1).

    Results are returned in the same order as *configs*.  Progress lines are
    printed as each trial completes (arrival order when parallel).
    """
    total = len(configs)
    results: List[Optional[BenchmarkResult]] = [None] * total

    if n_workers == 1:
        for i, cfg in enumerate(configs):
            result = _run_one(cfg)
            results[i] = result
            _print_progress(i + 1, total, cfg, result)
        return results  # type: ignore[return-value]

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_run_one, cfg): i
                         for i, cfg in enumerate(configs)}
        for future in as_completed(future_to_idx):
            idx    = future_to_idx[future]
            result = future.result()
            results[idx] = result
            done += 1
            _print_progress(done, total, configs[idx], result)

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_standard_benchmark(
    ships: List[str],
    scenarios: List[str],
    solvers: List[str],
    seeds: List[int],
    n_workers: int = 1,
    skip_keys: set = None,
) -> List[BenchmarkResult]:
    """Run all (ship × scenario × solver × seed) combinations."""
    configs = [
        BenchmarkConfig(
            ship_key=ship,
            scenario=scenario,
            solver_name=solver,
            model_key=ship,   # standard: model trained on same ship
            seed=seed,
        )
        for ship     in ships
        for scenario in scenarios
        for solver   in solvers
        for seed     in seeds
    ]

    if skip_keys:
        all_count = len(configs)
        configs = [c for c in configs
                   if (c.ship_key, c.scenario, c.solver_name, c.model_key, c.seed)
                   not in skip_keys]
        skipped = all_count - len(configs)
        if skipped:
            print(f"  (Skipping {skipped} trials already cached for version {ALGO_VERSION})")

    total = len(configs)
    print(f"\nRunning standard benchmark: {total} trials "
          f"({len(ships)} ships × {len(scenarios)} scenarios × "
          f"{len(solvers)} solvers × {len(seeds)} seeds)"
          + (f"  [workers={n_workers}]" if n_workers > 1 else "") + "\n")

    return _run_parallel(configs, n_workers)


def run_unloading_benchmark(
    ships: List[str],
    solvers: List[str],
    seeds: List[int],
    n_workers: int = 1,
    skip_keys: set = None,
) -> List[BenchmarkResult]:
    """Run all (ship × unloading_scenario × solver × seed) combinations."""
    configs = [
        BenchmarkConfig(
            ship_key=ship,
            scenario=scenario,
            solver_name=solver,
            model_key=ship,
            seed=seed,
        )
        for ship     in ships
        for scenario in UNLOADING_SCENARIOS
        for solver   in solvers
        for seed     in seeds
    ]

    if skip_keys:
        all_count = len(configs)
        configs = [c for c in configs
                   if (c.ship_key, c.scenario, c.solver_name, c.model_key, c.seed)
                   not in skip_keys]
        skipped = all_count - len(configs)
        if skipped:
            print(f"  (Skipping {skipped} trials already cached for version {ALGO_VERSION})")

    total = len(configs)
    print(f"\nRunning unloading benchmark: {total} trials "
          f"({len(ships)} ships × {len(UNLOADING_SCENARIOS)} scenarios × "
          f"{len(solvers)} solvers × {len(seeds)} seeds)"
          + (f"  [workers={n_workers}]" if n_workers > 1 else "") + "\n")

    return _run_parallel(configs, n_workers)


def run_transfer_benchmark(
    seeds: List[int],
    n_workers: int = 1,
    skip_keys: set = None,
) -> List[BenchmarkResult]:
    """Run all 9 ship × model combos for each ML solver on the balanced scenario."""
    configs = [
        BenchmarkConfig(
            ship_key=ship_key,
            scenario="balanced",
            solver_name=solver_name,
            model_key=model_key,
            seed=seed,
        )
        for solver_name           in ML_SOLVERS
        for (ship_key, model_key) in TRANSFER_COMBOS
        for seed                  in seeds
    ]

    if skip_keys:
        all_count = len(configs)
        configs = [c for c in configs
                   if (c.ship_key, c.scenario, c.solver_name, c.model_key, c.seed)
                   not in skip_keys]
        skipped = all_count - len(configs)
        if skipped:
            print(f"  (Skipping {skipped} trials already cached for version {ALGO_VERSION})")

    total = len(configs)
    print(f"\nRunning transfer benchmark: {total} trials "
          f"(ML solvers × 9 ship/model combos × {len(seeds)} seeds)"
          + (f"  [workers={n_workers}]" if n_workers > 1 else "") + "\n")

    return _run_parallel(configs, n_workers)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    standard: List[BenchmarkResult],
    transfer: List[BenchmarkResult],
    unloading: List[BenchmarkResult],
    output_path: Path,
    existing: dict = None,
) -> None:
    """Serialise benchmark results to JSON for the Streamlit benchmarks page."""
    import json
    from datetime import datetime

    existing = existing or {"standard": [], "transfer": [], "unloading": []}

    new_std_dicts = [r.to_dict() for r in standard]
    new_ul_dicts  = [r.to_dict() for r in unloading]
    new_tr_dicts  = [r.to_dict() for r in transfer]

    # "standard" key stores standard + unloading results (existing structure preserved)
    all_std_dicts = existing["standard"] + new_std_dicts + new_ul_dicts
    all_tr_dicts  = existing["transfer"] + new_tr_dicts
    all_ul_dicts  = existing["unloading"] + new_ul_dicts

    payload = {
        "metadata": {
            "generated_at":    datetime.now().isoformat(timespec="seconds"),
            "algo_version":    ALGO_VERSION,
            "ships_tested":    sorted({r["ship_key"]    for r in all_std_dicts}),
            "scenarios_tested": sorted({r["scenario"]   for r in all_std_dicts}),
            "solvers_tested":  sorted({r["solver_name"] for r in all_std_dicts}),
            "seeds":           sorted({r["seed"]        for r in all_std_dicts}),
        },
        "standard":  all_std_dicts,
        "transfer":  all_tr_dicts,
        "unloading": all_ul_dicts,
    }
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="Comprehensive cargo ship loader benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ships", nargs="+",
        choices=list(SHIP_CONFIGS.keys()),
        default=list(SHIP_CONFIGS.keys()),
        help="Ship sizes to benchmark (default: all)",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        choices=SCENARIOS,
        default=SCENARIOS,
        help="Scenarios to benchmark (default: all)",
    )
    parser.add_argument(
        "--solvers", nargs="+",
        choices=SOLVERS,
        default=SOLVERS,
        help="Solvers to include (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds (default: 42 99 7)",
    )
    parser.add_argument(
        "--no-transfer", action="store_true",
        help="Skip cross-ship transfer tests",
    )
    parser.add_argument(
        "--no-unloading", action="store_true",
        help="Skip unloading-order stress tests",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour output",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=None, metavar="N",
        help="Parallel worker processes (default: all CPU cores). Use 1 for sequential.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all trials even if cached for the current version",
    )
    parser.add_argument(
        "--version", type=str, default=None, metavar="VER",
        help=f"Override algo version tag for caching (default: {ALGO_VERSION}). "
             "Use to re-run a specific historical version, e.g. --version 0.1.0",
    )
    args = parser.parse_args()

    # Allow overriding ALGO_VERSION at runtime (must happen before workers fork)
    if args.version:
        import sys as _sys
        _sys.modules[__name__].ALGO_VERSION = args.version

    # Disable color when not a TTY or explicitly requested
    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False

    n_workers = args.workers if args.workers is not None else os.cpu_count() or 1
    if n_workers > 1:
        print(f"Parallel mode: {n_workers} worker processes  (use --workers 1 to disable)")

    # --- Load existing results and compute skip sets ---
    existing = _load_existing_results(args.output)
    if args.force:
        skip_std = skip_ul = skip_tr = set()
        print(f"--force: ignoring cached results for version {ALGO_VERSION}")
    else:
        skip_std = _done_keys(existing["standard"])
        skip_ul  = _done_keys(existing["unloading"])
        skip_tr  = _done_keys(existing["transfer"])
        if skip_std or skip_ul or skip_tr:
            print(f"Version {ALGO_VERSION}: {len(skip_std)} standard, "
                  f"{len(skip_ul)} unloading, {len(skip_tr)} transfer trials cached → will skip")

    t_start = time.perf_counter()

    # --- Standard benchmark ---
    std_results = run_standard_benchmark(
        ships=args.ships,
        scenarios=args.scenarios,
        solvers=args.solvers,
        seeds=args.seeds,
        n_workers=n_workers,
        skip_keys=skip_std,
    )
    print_standard_table(std_results)

    # --- Unloading-order benchmark ---
    unloading_results: List[BenchmarkResult] = []
    if not args.no_unloading:
        unloading_results = run_unloading_benchmark(
            ships=args.ships,
            solvers=args.solvers,
            seeds=args.seeds,
            n_workers=n_workers,
            skip_keys=skip_ul,
        )
        print_unloading_table(unloading_results)

    # --- Transfer benchmark ---
    transfer_results: List[BenchmarkResult] = []
    if not args.no_transfer:
        transfer_results = run_transfer_benchmark(
            seeds=args.seeds,
            n_workers=n_workers,
            skip_keys=skip_tr,
        )
        print_transfer_table(transfer_results)

    # --- Persist to JSON ---
    save_results(std_results, transfer_results, unloading_results, args.output, existing)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
