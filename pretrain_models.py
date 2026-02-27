"""Pre-train NeuralRankerSolver models for the three standard ship sizes.

Run once before deploying the Streamlit app:
    conda run -n personal python pretrain_models.py

Saves .pkl files to models/ so the app can load them instantly.
Training uses Beam Search (K=5) behavioural cloning.  Smaller ships
train faster; runtime is ~2 min total for all three.
"""

import json
import time
from pathlib import Path

from models import CargoShip
from solvers.neural_ranker import NeuralRankerSolver
from solvers.rl_bayesian import RLBayesianSolver

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Ship configurations ──────────────────────────────────────────────────────
CONFIGS = [
    {
        "key":        "coastal",
        "label":      "Coastal Feeder  (12 × 9 × 5)",
        "ship_params": dict(length=12, base_width=5, max_width=9,  height=5,
                            width_step=1, max_weight=500_000.0),
        "n_episodes": 300,   # fast — small ship
        "n_20ft":     12,    # min containers per episode (normal load)
        "n_40ft":     4,
        "n_20ft_max": 40,    # max containers per episode (overloaded: 40×28k=1.12M > 500k)
        "n_40ft_max": 15,
        "weight_max": 28_000.0,
    },
    {
        "key":        "handymax",
        "label":      "Handymax  (24 × 11 × 7)",
        "ship_params": dict(length=24, base_width=6, max_width=11, height=7,
                            width_step=1, max_weight=1_500_000.0),
        "n_episodes": 200,
        "n_20ft":     35,    # min (normal load)
        "n_40ft":     12,
        "n_20ft_max": 100,   # max (overloaded: 100×28k=2.8M > 1.5M)
        "n_40ft_max": 40,
        "weight_max": 28_000.0,
    },
    {
        "key":        "panamax",
        "label":      "Panamax  (36 × 13 × 9)",
        "ship_params": dict(length=36, base_width=7, max_width=13, height=9,
                            width_step=1, max_weight=3_000_000.0),
        "n_episodes": 150,
        "n_20ft":     60,    # min (normal load)
        "n_40ft":     25,
        "n_20ft_max": 180,   # max (overloaded: 180×28k=5.04M > 3M)
        "n_40ft_max": 70,
        "weight_max": 28_000.0,
    },
]


def _print_stats(stats: dict) -> None:
    """Pretty-print training_stats_ dict."""
    if not stats:
        return
    print(f"    elapsed:        {stats.get('elapsed_s', '?')}s")
    if "il_elapsed_s" in stats:
        print(f"    IL elapsed:     {stats['il_elapsed_s']}s")
        print(f"    RL elapsed:     {stats['rl_elapsed_s']}s")
        il_loss = stats.get("il_loss_curve", [])
        print(f"    IL loss:        {il_loss[0]:.4f} → {il_loss[-1]:.4f}  ({len(il_loss)} iters)")
        il_val = stats.get("il_val_score_curve", [])
        if il_val:
            print(f"    IL val score:   {il_val[0]:.4f} → {il_val[-1]:.4f}  best={stats.get('il_best_val_score','?')}")
        rl_r = stats.get("rl_reward_history", [])
        if rl_r:
            print(f"    RL reward:      {rl_r[0]:.4f} → {rl_r[-1]:.4f}  (mean={sum(rl_r)/len(rl_r):.4f})")
        final_loss = stats.get("final_loss_curve", [])
        print(f"    Final loss:     {final_loss[0]:.4f} → {final_loss[-1]:.4f}  ({len(final_loss)} iters)")
        final_val = stats.get("final_val_score_curve", [])
        if final_val:
            print(f"    Final val:      best={stats.get('final_best_val_score','?')}")
    else:
        loss = stats.get("loss_curve", [])
        if loss:
            print(f"    loss:           {loss[0]:.4f} → {loss[-1]:.4f}  ({stats.get('n_iter','?')} iters)")
        val = stats.get("val_score_curve", [])
        if val:
            print(f"    val score:      {val[0]:.4f} → {val[-1]:.4f}  best={stats.get('best_val_score','?')}")
        print(f"    samples:        {stats.get('n_samples','?')}  features: {stats.get('n_features','?')}")


def pretrain_all():
    total_t0 = time.perf_counter()
    all_stats = {}

    for cfg in CONFIGS:
        out_path = MODELS_DIR / f"neural_ranker_{cfg['key']}.pkl"
        print(f"\n{'─' * 60}")
        print(f"Training: {cfg['label']}")
        print(f"  episodes={cfg['n_episodes']}  beam_width=5")
        print(f"  containers: {cfg['n_20ft']} × 20 ft  +  {cfg['n_40ft']} × 40 ft")

        ship = CargoShip(**cfg["ship_params"])
        solver = NeuralRankerSolver(ship, max_weight=cfg["weight_max"])
        solver.fit(
            n_episodes  = cfg["n_episodes"],
            beam_width  = 5,
            seed        = 42,
            ship_params = cfg["ship_params"],
            n_20ft      = cfg["n_20ft"],
            n_40ft      = cfg["n_40ft"],
            n_20ft_max  = cfg.get("n_20ft_max", 0),
            n_40ft_max  = cfg.get("n_40ft_max", 0),
            weight_min  = 2_000.0,
        )
        solver.save(str(out_path))

        stats = getattr(solver, "training_stats_", {})
        all_stats[f"neural_ranker_{cfg['key']}"] = stats
        print(f"  ✓ saved → {out_path}  ({stats.get('elapsed_s', '?')}s)")
        _print_stats(stats)

    total = time.perf_counter() - total_t0
    print(f"\n{'─' * 60}")
    print(f"All Neural Ranker models trained in {total:.1f}s")
    print(f"Files in {MODELS_DIR}:")
    for f in sorted(MODELS_DIR.glob("neural_ranker_*.pkl")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    return all_stats


RL_CONFIGS = [
    {
        "key":        "coastal",
        "label":      "RL Bayesian — Coastal Feeder  (12 × 9 × 5)",
        "ship_params": dict(length=12, base_width=5, max_width=9,  height=5,
                            width_step=1, max_weight=500_000.0),
        "n_il":       80,
        "n_bayes":    25,
        "n_rl":       40,
        "n_samples":  20,
        "n_20ft":     12,
        "n_40ft":     4,
        "n_20ft_max": 40,
        "n_40ft_max": 15,
    },
    {
        "key":        "handymax",
        "label":      "RL Bayesian — Handymax  (24 × 11 × 7)",
        "ship_params": dict(length=24, base_width=6, max_width=11, height=7,
                            width_step=1, max_weight=1_500_000.0),
        "n_il":       60,
        "n_bayes":    20,
        "n_rl":       30,
        "n_samples":  15,
        "n_20ft":     35,
        "n_40ft":     12,
        "n_20ft_max": 100,
        "n_40ft_max": 40,
    },
    {
        "key":        "panamax",
        "label":      "RL Bayesian — Panamax  (36 × 13 × 9)",
        "ship_params": dict(length=36, base_width=7, max_width=13, height=9,
                            width_step=1, max_weight=3_000_000.0),
        "n_il":       40,
        "n_bayes":    15,
        "n_rl":       20,
        "n_samples":  10,
        "n_20ft":     60,
        "n_40ft":     25,
        "n_20ft_max": 180,
        "n_40ft_max": 70,
    },
]


def pretrain_rl_bayesian():
    total_t0 = time.perf_counter()
    all_stats = {}

    for cfg in RL_CONFIGS:
        out_path = MODELS_DIR / f"rl_bayesian_{cfg['key']}.pkl"
        print(f"\n{'─' * 60}")
        print(f"Training: {cfg['label']}")
        print(f"  n_il={cfg['n_il']}  n_bayes={cfg['n_bayes']}  "
              f"n_rl={cfg['n_rl']}  n_samples={cfg['n_samples']}")
        print(f"  containers: {cfg['n_20ft']} × 20 ft  +  {cfg['n_40ft']} × 40 ft")

        ship   = CargoShip(**cfg["ship_params"])
        solver = RLBayesianSolver(ship)
        solver.fit(
            n_il        = cfg["n_il"],
            n_bayes     = cfg["n_bayes"],
            n_rl        = cfg["n_rl"],
            n_samples   = cfg["n_samples"],
            n_20ft      = cfg["n_20ft"],
            n_40ft      = cfg["n_40ft"],
            n_20ft_max  = cfg.get("n_20ft_max", 0),
            n_40ft_max  = cfg.get("n_40ft_max", 0),
            weight_min  = 2_000.0,
            seed        = 42,
            ship_params = cfg["ship_params"],
        )
        solver.save(str(out_path))

        stats = getattr(solver, "training_stats_", {})
        all_stats[f"rl_bayesian_{cfg['key']}"] = stats
        print(f"  ✓ saved → {out_path}  ({stats.get('elapsed_s', '?')}s)")
        _print_stats(stats)

    total = time.perf_counter() - total_t0
    print(f"\n{'─' * 60}")
    print(f"All RL Bayesian models trained in {total:.1f}s")
    print(f"Files in {MODELS_DIR}:")
    for f in sorted(MODELS_DIR.glob("rl_bayesian_*.pkl")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    return all_stats


if __name__ == "__main__":
    nr_stats  = pretrain_all()
    rl_stats  = pretrain_rl_bayesian()

    stats_path = MODELS_DIR / "training_stats.json"
    with open(stats_path, "w") as fh:
        json.dump({**nr_stats, **rl_stats}, fh, indent=2)
    print(f"\n✓ Training stats saved → {stats_path}")
