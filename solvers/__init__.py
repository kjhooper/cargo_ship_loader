"""Solver implementations for the Cargo Ship Loader.

Always available
----------------
  BeamSearchSolver           — Heuristic beam search (H2)
  SimulatedAnnealingSolver   — Simulated annealing    (H3)

Require optional dependencies
-------------------------------
  BayesianOptSolver   — needs ``optuna``        (M1)
  NeuralRankerSolver  — needs ``scikit-learn``  (M2)
  generate_training_data — data generation helper for M2
"""

from .beam_search import BeamSearchSolver
from .simulated_annealing import SimulatedAnnealingSolver

try:
    from .bayesian_opt import BayesianOptSolver
except ImportError:
    pass

try:
    from .neural_ranker import NeuralRankerSolver, generate_training_data
except ImportError:
    pass

__all__ = [
    "BeamSearchSolver",
    "SimulatedAnnealingSolver",
    "BayesianOptSolver",
    "NeuralRankerSolver",
    "generate_training_data",
]
