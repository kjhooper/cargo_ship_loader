"""RL-Bayesian Hybrid Solver (M3).

Manifest-conditioned weight predictor combining:
  Phase 1 — Imitation Learning from BayesianOptSolver (teacher)
  Phase 2 — RL fine-tuning via Reward-Weighted Regression with SIR

How it differs from NeuralRankerSolver (M2)
---------------------------------------------
M2 imitates Beam Search's *position* choices (replaces ``_score_position``).
M3 imitates BayesOpt's *weight* choices (sets scorer coefficients):

    manifest features  →  MLP  →  [k_gz, k_trim, k_list, k_diag, k_stacking]
                                              ↓
                               CargoLoader(ship, **weights)

The MLP takes 8 manifest-level statistics (computed before any placement) and
predicts the five CargoLoader scorer weights that BayesOpt would have found
for that manifest.  At inference the full loading is near-instantaneous.

Phase 2 — RWR with SIR
-----------------------
After the IL pre-train, ``sklearn.MLPRegressor`` does not support
``sample_weight``, so reward-weighted regression is implemented via
Sequential Importance Resampling (SIR):

  For each RL episode:
    1. Sample K weight vectors:  w_i ~ N(μ=MLP(features), σ)
    2. Evaluate:                 r_i = final_score(CargoLoader(w_i))
    3. Reward-weight:            α_i = softmax(β · r_i)
    4. Resample with replacement: w_rl_i ~ Categorical(α_i)

  After all episodes, refit MLP on IL data + resampled RL data (all equal
  weight).  This is mathematically equivalent to reward-weighted regression
  and requires no modification to sklearn.

Requires: ``scikit-learn``, ``joblib``, ``optuna`` (for IL teacher).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from models import CargoShip, ShippingContainer
from algorithm import BaseSolver, CargoLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W_KEYS = ["k_gz", "k_trim", "k_list", "k_diag", "k_stacking"]
W_MIN  = np.array([0.5,  0.5,  0.5,  0.5,  0.0],  dtype=np.float64)
W_MAX  = np.array([12.0, 12.0, 12.0, 12.0, 2.0],  dtype=np.float64)
N_MANIFEST_FEATURES = 8


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _compute_manifest_features(
    containers: List[ShippingContainer],
    ship: CargoShip,
) -> np.ndarray:
    """Return an 8-dim float64 vector summarising the manifest.

    Features are computed before any placement, enabling the solver to
    configure greedy weights before a single container is placed.

    Dimensions
    ----------
    0  n / 100                 total container count (normalised)
    1  n_20ft / n              fraction that are 20 ft
    2  weight_mean / max_w     mean weight
    3  weight_std  / max_w     weight spread
    4  weight_max  / max_w     heaviest container
    5  weight_min  / max_w     lightest container
    6  total_w    / max_w      aggregate load fraction
    7  n_20ft / ship_length    20 ft bay density
    """
    weights = np.array([c.weight for c in containers], dtype=np.float64)
    n    = max(len(containers), 1)
    n20  = sum(1 for c in containers if c.size == 1)
    mw   = ship.max_weight + 1e-9
    return np.array([
        n   / 100.0,
        n20 / n,
        weights.mean()  / mw,
        weights.std()   / mw,
        weights.max()   / mw,
        weights.min()   / mw,
        weights.sum()   / mw,
        n20 / max(ship.length, 1),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fresh_ship(ship: CargoShip) -> CargoShip:
    return CargoShip(
        length     = ship.length,
        base_width = ship.base_width,
        max_width  = ship.width,
        height     = ship.height,
        width_step = ship.width_step,
        max_weight = ship.max_weight,
    )


def _eval_weights(
    w: np.ndarray,
    containers: List[ShippingContainer],
    ref_ship: CargoShip,
) -> float:
    """Run CargoLoader with weight vector w; return final_score in [0, 1]."""
    trial_ship = _fresh_ship(ref_ship)
    weights_dict = dict(zip(W_KEYS, w.tolist()))
    loader = CargoLoader(trial_ship, **weights_dict)
    loader.load(containers)
    return loader.final_score()


def _random_containers(
    n_20ft: int, n_40ft: int,
    weight_min: float, weight_max: float,
    seed: int,
) -> List[ShippingContainer]:
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    conts = (
        [ShippingContainer(size=1,
                           weight=round(rng.uniform(weight_min, weight_max), 1))
         for _ in range(n_20ft)]
        + [ShippingContainer(size=2,
                             weight=round(rng.uniform(weight_min, weight_max), 1))
           for _ in range(n_40ft)]
    )
    rng.shuffle(conts)
    return conts


# ---------------------------------------------------------------------------
# IL data generation
# ---------------------------------------------------------------------------

def _generate_il_data(
    ref_ship: CargoShip,
    n_il: int,
    n_bayes: int,
    weight_min: float,
    weight_max: float,
    n_20ft: int,
    n_40ft: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run BayesOpt on n_il random manifests; collect (features, best_weights).

    Returns
    -------
    X : float64 array of shape (n_il, N_MANIFEST_FEATURES)
    y : float64 array of shape (n_il, 5)  — one row per W_KEYS vector
    """
    from solvers.bayesian_opt import BayesianOptSolver

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for _ in range(n_il):
        ep_seed = rng.randint(0, 2**31)
        conts   = _random_containers(n_20ft, n_40ft, weight_min, weight_max, ep_seed)

        teach_ship = _fresh_ship(ref_ship)
        teacher    = BayesianOptSolver(teach_ship, n_trials=n_bayes, seed=ep_seed)
        teacher.load(conts)

        w = np.array([teacher.best_weights[k] for k in W_KEYS], dtype=np.float64)
        f = _compute_manifest_features(conts, ref_ship)
        X_list.append(f)
        y_list.append(w)

    return np.stack(X_list), np.stack(y_list)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class RLBayesianSolver(BaseSolver):
    """Manifest-conditioned greedy weight predictor (IL from BayesOpt + RWR).

    Usage
    -----
    ::

        solver = RLBayesianSolver(ship)
        solver.fit(n_il=80, n_bayes=25, n_rl=40, n_samples=20,
                   n_20ft=60, n_40ft=25, ship_params=PANAMAX)
        manifest = solver.load(containers)

        solver.save("rl_bayesian.pkl")
        solver2 = RLBayesianSolver.load_model(ship, "rl_bayesian.pkl")

    Parameters
    ----------
    ship : CargoShip
    hidden_layer_sizes : tuple
        MLP architecture for the weight predictor (default: two 64-unit layers).
    sigma : float
        Exploration noise std for RL sampling in raw weight space.
    beta : float
        Softmax temperature for reward-weighting (higher = greedier selection).
    """

    def __init__(
        self,
        ship: CargoShip,
        hidden_layer_sizes: Tuple[int, ...] = (64, 64),
        sigma: float = 1.0,
        beta:  float = 5.0,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for RLBayesianSolver.  "
                "Install with: conda run -n personal pip install scikit-learn"
            )
        super().__init__(ship)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.sigma  = sigma
        self.beta   = beta
        self.manifest: List[Dict] = []
        self._fitted = False

        self._scaler = StandardScaler()
        self._mlp    = MLPRegressor(
            hidden_layer_sizes = hidden_layer_sizes,
            activation         = "relu",
            max_iter           = 1000,
            random_state       = 42,
            early_stopping     = True,
            n_iter_no_change   = 20,
            validation_fraction= 0.15,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        n_il:      int   = 100,
        n_bayes:   int   = 25,
        n_rl:      int   = 50,
        n_samples: int   = 10,
        n_20ft:    int   = 60,
        n_40ft:    int   = 25,
        weight_min: float = 2_000.0,
        seed:      int   = 42,
        ship_params: Optional[Dict[str, Any]] = None,
    ) -> "RLBayesianSolver":
        """Run IL pre-training then RWR fine-tuning."""
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for RLBayesianSolver.fit() (IL teacher).  "
                "Install with: conda run -n personal pip install optuna"
            )
        if ship_params is None:
            ship_params = dict(
                length     = self.ship.length,
                base_width = self.ship.base_width,
                max_width  = self.ship.width,
                height     = self.ship.height,
                width_step = self.ship.width_step,
                max_weight = self.ship.max_weight,
            )

        ref_ship   = CargoShip(**ship_params)
        weight_max = ref_ship.max_weight
        rng        = random.Random(seed)
        np_rng     = np.random.default_rng(seed)

        # ── Phase 1: Imitation Learning ──────────────────────────────────
        print(f"  [IL] {n_il} BayesOpt episodes × {n_bayes} trials…")
        X_il, y_il = _generate_il_data(
            ref_ship, n_il, n_bayes, weight_min, weight_max,
            n_20ft, n_40ft, rng,
        )
        X_il_scaled = self._scaler.fit_transform(X_il)
        self._mlp.fit(X_il_scaled, y_il)
        self._fitted = True

        # ── Phase 2: RL fine-tuning via SIR ──────────────────────────────
        print(f"  [RL] {n_rl} episodes × {n_samples} samples (SIR)…")
        X_rl_list: List[np.ndarray] = []
        y_rl_list: List[np.ndarray] = []

        for _ in range(n_rl):
            ep_seed    = rng.randint(0, 2**31)
            conts      = _random_containers(n_20ft, n_40ft, weight_min, weight_max, ep_seed)
            feat       = _compute_manifest_features(conts, ref_ship)
            feat_scaled = self._scaler.transform(feat.reshape(1, -1))
            mu         = np.clip(self._mlp.predict(feat_scaled)[0], W_MIN, W_MAX)

            # Sample K weight vectors from Gaussian(mu, sigma^2 I)
            ep_np_rng = np.random.default_rng(rng.randint(0, 2**31))
            noise     = ep_np_rng.standard_normal((n_samples, 5))
            w_samples = np.clip(mu + self.sigma * noise, W_MIN, W_MAX)

            # Evaluate each candidate
            rewards = np.array([_eval_weights(w, conts, ref_ship) for w in w_samples])

            # Softmax reward-weighting (temperature = self.beta)
            log_alpha = self.beta * (rewards - rewards.max())  # stability shift
            alpha     = np.exp(log_alpha)
            alpha    /= alpha.sum()

            # SIR: resample n_samples indices with replacement
            sir_rng  = np.random.default_rng(rng.randint(0, 2**31))
            indices  = sir_rng.choice(n_samples, size=n_samples, replace=True, p=alpha)
            w_resampled = w_samples[indices]

            X_rl_list.append(np.tile(feat, (n_samples, 1)))
            y_rl_list.append(w_resampled)

        # ── Final refit on IL + RL data ───────────────────────────────────
        if X_rl_list:
            X_rl = np.vstack(X_rl_list)
            y_rl = np.vstack(y_rl_list)
            X_all = np.vstack([X_il, X_rl])
            y_all = np.vstack([y_il, y_rl])
            X_all_scaled = self._scaler.transform(X_all)
            self._mlp.fit(X_all_scaled, y_all)

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted scaler + MLP to path (via joblib)."""
        joblib.dump({
            "scaler": self._scaler,
            "mlp":    self._mlp,
            "sigma":  self.sigma,
            "beta":   self.beta,
        }, path)

    @classmethod
    def load_model(
        cls, ship: CargoShip, path: str, **kwargs
    ) -> "RLBayesianSolver":
        """Load a previously saved model from path."""
        data   = joblib.load(path)
        solver = cls(
            ship,
            sigma = data.get("sigma", 1.0),
            beta  = data.get("beta",  5.0),
            **kwargs,
        )
        solver._scaler = data["scaler"]
        solver._mlp    = data["mlp"]
        solver._fitted = True
        return solver

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_weights(self, containers: List[ShippingContainer]) -> np.ndarray:
        """Predict scorer weights from manifest statistics."""
        feat        = _compute_manifest_features(containers, self.ship)
        feat_scaled = self._scaler.transform(feat.reshape(1, -1))
        raw         = self._mlp.predict(feat_scaled)[0]
        return np.clip(raw, W_MIN, W_MAX)

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        """Predict weights from manifest → delegate to CargoLoader."""
        if not self._fitted:
            raise RuntimeError(
                "RLBayesianSolver must be fitted before calling load().  "
                "Call solver.fit(...) or RLBayesianSolver.load_model(...)."
            )
        w = self._predict_weights(containers)
        weights_dict = dict(zip(W_KEYS, w.tolist()))
        loader       = CargoLoader(self.ship, **weights_dict)
        self.manifest = loader.load(containers)
        return self.manifest
