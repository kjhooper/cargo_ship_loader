"""Bayesian Optimisation solver (M1).

Treats the five CargoLoader scorer weights as continuous hyperparameters and
uses a Gaussian-Process surrogate (via ``optuna``) with Expected-Improvement
acquisition to find the weight vector that maximises ``final_score()`` on
the supplied container manifest.

The underlying search algorithm is still the greedy CargoLoader — only the
weights are learned.  Each trial takes milliseconds, so 50–200 GP evaluations
are practical.

Requires: ``pip install optuna`` (or ``conda install -c conda-forge optuna``).
"""

from __future__ import annotations

import copy
from typing import List, Dict

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from models import CargoShip, ShippingContainer
from algorithm import BaseSolver, CargoLoader


class BayesianOptSolver(BaseSolver):
    """Bayesian Optimisation over CargoLoader scorer weights.

    Parameters
    ----------
    ship : CargoShip
        Empty ship to load containers into.
    n_trials : int
        Number of optuna trials (GP evaluations).
    seed : int
        Optuna sampler seed for reproducibility.
    """

    def __init__(self, ship: CargoShip, n_trials: int = 50, seed: int = 42):
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for BayesianOptSolver.  "
                "Install with: conda run -n personal pip install optuna"
            )
        super().__init__(ship)
        self.n_trials = n_trials
        self.seed     = seed
        self.manifest: List[Dict] = []
        self.best_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_ship(self) -> CargoShip:
        """Return an empty copy of self.ship with identical geometry."""
        return CargoShip(
            length     = self.ship.length,
            base_width = self.ship.base_width,
            max_width  = self.ship.width,
            height     = self.ship.height,
            width_step = self.ship.width_step,
            max_weight = self.ship.max_weight,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        """Run Bayesian optimisation then do a final greedy pass on self.ship."""

        def objective(trial: optuna.Trial) -> float:
            k_gz       = trial.suggest_float("k_gz",       0.5, 12.0)
            k_trim     = trial.suggest_float("k_trim",     0.5, 12.0)
            k_list     = trial.suggest_float("k_list",     0.5, 12.0)
            k_diag     = trial.suggest_float("k_diag",     0.5, 12.0)
            k_stacking = trial.suggest_float("k_stacking", 0.0,  2.0)

            trial_ship = self._fresh_ship()
            loader = CargoLoader(
                trial_ship,
                k_gz=k_gz, k_trim=k_trim,
                k_list=k_list, k_diag=k_diag,
                k_stacking=k_stacking,
            )
            loader.load(containers)
            # BaseSolver.final_score() — called on trial_ship via loader
            port_w, stbd_w = trial_ship.port_starboard_balance()
            fore_w, aft_w  = trial_ship.fore_aft_balance()
            fp, fs, ap, as_ = trial_ship.quadrant_balance()

            max_ps = max(port_w, stbd_w)
            max_fa = max(fore_w, aft_w)
            d1, d2 = fp + as_, fs + ap
            max_d  = max(d1, d2)

            ps_ratio   = min(port_w, stbd_w) / max_ps if max_ps > 0 else 1.0
            fa_ratio   = min(fore_w, aft_w)  / max_fa if max_fa > 0 else 1.0
            diag_ratio = min(d1, d2)         / max_d  if max_d  > 0 else 1.0
            return (ps_ratio + fa_ratio + diag_ratio) / 3.0

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study   = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_weights = study.best_params

        # Final authoritative run on self.ship with the best weights
        loader = CargoLoader(self.ship, **self.best_weights)
        self.manifest = loader.load(containers)
        return self.manifest
