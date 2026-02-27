"""RL-Bayesian + Simulated Annealing hybrid solver (H3+M3).

Uses a pre-trained RLBayesianSolver to construct a high-quality initial
placement, then refines it with Simulated Annealing's perturbation loop.

This combines the complementary strengths of both approaches:
  - RL Bayesian: fast (~40 ms), generalises from BayesOpt training,
    produces globally well-balanced placements.
  - SA: local search that escapes from suboptimal positions, particularly
    effective in constrained scenarios (weight-limited, space-limited,
    or cross-ship transfer where the RL model is out-of-distribution).

Falls back to a greedy warm start when no pre-trained model is available
(identical behaviour to plain SimulatedAnnealingSolver in that case).

Requires: ``scikit-learn`` + ``optuna`` (for RLBayesianSolver).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from models import CargoShip, ShippingContainer
from algorithm import CargoLoader
from solvers.simulated_annealing import SimulatedAnnealingSolver

try:
    from solvers.rl_bayesian import RLBayesianSolver
    _RL_BAYESIAN_AVAILABLE = True
except ImportError:
    _RL_BAYESIAN_AVAILABLE = False

# Dummy ship geometry used when loading a pre-trained pkl — will be swapped
# out immediately after loading, so the exact dimensions don't matter.
_DUMMY_SHIP_PARAMS = dict(
    length=36, base_width=7, max_width=13, height=9, width_step=1, max_weight=50_000.0
)


class RLBayesianSASolver(SimulatedAnnealingSolver):
    """Simulated Annealing with an RL-Bayesian warm start.

    Parameters
    ----------
    ship : CargoShip
        Empty ship to load containers into.
    n_iterations : int
        SA perturbation iterations (applied *after* the warm start).
    T_start : float
        Initial SA temperature.
    cooling : float
        Geometric cooling factor per iteration.
    seed : int | None
        RNG seed for SA perturbations.
    model_path : str | Path | None
        Path to a pre-trained ``RLBayesianSolver`` ``.pkl`` file.
        When supplied and the file exists, it is used for the warm start.
        Otherwise the solver falls back to a greedy warm start.
    k_gz, k_trim, k_list, k_diag, k_stacking : float
        CargoLoader scorer weights used for the *fallback* greedy warm start
        (ignored when the RL Bayesian model is available).
    """

    def __init__(
        self,
        ship: CargoShip,
        n_iterations: int   = 2000,
        T_start: float      = 0.05,
        cooling: float      = 0.999,
        seed: Optional[int] = None,
        model_path: Optional[str] = None,
        k_gz: float         = 5.0,
        k_trim: float       = 4.0,
        k_list: float       = 4.0,
        k_diag: float       = 6.0,
        k_stacking: float   = 0.5,
    ):
        super().__init__(
            ship, n_iterations, T_start, cooling, seed,
            k_gz, k_trim, k_list, k_diag, k_stacking,
        )
        self.model_path: Optional[Path] = Path(model_path) if model_path else None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        """RL Bayesian warm start → SA refinement."""
        # --- Step 1: RL Bayesian (or greedy fallback) warm start ---
        if (
            _RL_BAYESIAN_AVAILABLE
            and self.model_path is not None
            and self.model_path.exists()
        ):
            dummy = CargoShip(**_DUMMY_SHIP_PARAMS)
            rl_solver = RLBayesianSolver.load_model(dummy, str(self.model_path))
            rl_solver.ship    = self.ship
            rl_solver._fitted = True
            self.manifest = rl_solver.load(containers)
        else:
            loader = CargoLoader(
                self.ship,
                k_gz=self.k_gz, k_trim=self.k_trim,
                k_list=self.k_list, k_diag=self.k_diag,
                k_stacking=self.k_stacking,
            )
            self.manifest = loader.load(containers)

        # --- Steps 2 + 3: SA refinement + restore best ---
        return self._run_sa_loop()
