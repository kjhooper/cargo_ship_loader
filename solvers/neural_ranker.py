"""Neural Position Ranker solver (M2).

Replaces the hand-crafted ``_score_position`` function with a small MLP
trained via behavioural cloning from Beam Search.

Data generation
---------------
``generate_training_data(ship_params, n_episodes, beam_width, seed)`` runs
Beam Search (K=beam_width) on *n_episodes* random container manifests.  For
each placement step it records:

  * All valid positions as feature vectors  (shape: (n_candidates, N_FEATURES))
  * A binary label: 1.0 for the position the best beam chose, 0.0 for all others

The MLP is trained as a binary classifier.  At inference ``predict_proba``
column 1 replaces ``_score_position``.

Feature vector (16 dimensions)
-------------------------------
  State (7):  moment_z_norm, fore_frac, port_frac, fp_frac, fs_frac, ap_frac, as_frac
  Container (2): size (1 or 2), weight_norm
  Position (3):  tier_norm, bay_c_norm, col_norm
  Post-placement (4): gz_norm, trim_norm, list_norm, diag_norm

Requires: ``scikit-learn`` (``conda install scikit-learn``).
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from models import CargoShip, ShippingContainer
from algorithm import BaseSolver, CargoLoader

N_FEATURES = 16


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(
    moment_z: float,
    fp_w: float, fs_w: float, ap_w: float, as_w: float, total_w: float,
    container: ShippingContainer,
    bay: int, col: int, tier: int,
    ship_length: int, ship_width: int, ship_height: int,
    max_weight: float,
) -> np.ndarray:
    """Return a 1D float array of N_FEATURES=16 values."""
    eps = 1e-9
    sz  = container.size
    w   = container.weight

    # --- state features (7) ---
    fore_w = fp_w + fs_w
    port_w = fp_w + ap_w
    moment_z_norm = (moment_z / (ship_height - 1)) / (total_w + eps)
    fore_frac = fore_w / (total_w + eps)
    port_frac = port_w / (total_w + eps)
    fp_frac   = fp_w   / (total_w + eps)
    fs_frac   = fs_w   / (total_w + eps)
    ap_frac   = ap_w   / (total_w + eps)
    as_frac   = as_w   / (total_w + eps)

    # --- container features (2) ---
    weight_norm = w / (max_weight + eps)

    # --- position features (3) ---
    bay_c      = bay + (sz - 1) / 2.0
    tier_norm  = tier  / (ship_height - 1) if ship_height > 1 else 0.0
    bay_c_norm = bay_c / ship_length
    col_norm   = col   / ship_width

    # --- post-placement features (4) ---
    total_new = total_w + w
    is_fore = bay_c < ship_length / 2.0
    is_port = col   < ship_width  / 2.0

    new_fp = fp_w + (w if is_fore and     is_port else 0.0)
    new_fs = fs_w + (w if is_fore and not is_port else 0.0)
    new_ap = ap_w + (w if not is_fore and     is_port else 0.0)
    new_as = as_w + (w if not is_fore and not is_port else 0.0)

    gz_norm   = ((moment_z + w * tier) / total_new) / (ship_height - 1) if ship_height > 1 else 0.0
    trim_norm = abs((new_fp + new_fs) - (new_ap + new_as)) / total_new
    list_norm = abs((new_fp + new_ap) - (new_fs + new_as)) / total_new
    diag_norm = abs((new_fp + new_as) - (new_fs + new_ap)) / total_new

    return np.array([
        moment_z_norm, fore_frac, port_frac, fp_frac, fs_frac, ap_frac, as_frac,
        float(sz), weight_norm,
        tier_norm, bay_c_norm, col_norm,
        gz_norm, trim_norm, list_norm, diag_norm,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_training_data(
    ship_params: Dict[str, Any],
    n_episodes: int = 200,
    beam_width: int = 5,
    n_20ft: int     = 60,
    n_40ft: int     = 25,
    n_20ft_max: int = 0,
    n_40ft_max: int = 0,
    weight_min: float = 2000.0,
    weight_max: float = 28000.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Beam Search on *n_episodes* random manifests; collect (features, label) pairs.

    Each episode's container count is sampled uniformly between n_20ft and
    n_20ft_max (n_40ft and n_40ft_max), so the training set spans both normal
    loads and weight-overloaded manifests where the ship's max_weight cap binds.
    If n_20ft_max / n_40ft_max are 0 (or <= the min), the count is fixed.

    Labels are binary: 1 for the position chosen by the best beam at each step,
    0 for all other valid positions considered at that step.

    Returns
    -------
    X : float32 array of shape (n_samples, N_FEATURES)
    y : float32 array of shape (n_samples,)   values in {0.0, 1.0}
    """
    # Import here to avoid circular imports at module load time
    from solvers.beam_search import BeamSearchSolver, _BeamState

    rng_top = random.Random(seed)
    X_list: List[np.ndarray] = []
    y_list: List[float]      = []

    for episode in range(n_episodes):
        ep_seed = rng_top.randint(0, 2**31)
        rng     = random.Random(ep_seed)

        ep_n20 = rng_top.randint(n_20ft, n_20ft_max) if n_20ft_max > n_20ft else n_20ft
        ep_n40 = rng_top.randint(n_40ft, n_40ft_max) if n_40ft_max > n_40ft else n_40ft

        ShippingContainer.reset_id_counter()
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(weight_min, weight_max), 1))
             for _ in range(ep_n20)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(weight_min, weight_max), 1))
               for _ in range(ep_n40)]
        )
        rng.shuffle(containers)

        ship = CargoShip(**ship_params)
        solver = BeamSearchSolver(ship, beam_width=beam_width)

        # --- Replicate the beam search loop with feature collection ---
        sorted_containers = sorted(containers, key=lambda c: (-c.weight, -c.size))
        beams: List[_BeamState] = [solver._initial_state()]

        for container in sorted_containers:
            candidates = []
            for state in beams:
                loader = solver._loader_for_state(state)
                for (bay, half, col, tier) in loader._enumerate_valid_positions(container):
                    score = loader._score_position(container, bay, half, col, tier)
                    candidates.append((state, bay, half, col, tier, score))

            if not candidates:
                for state in beams:
                    state.manifest.append({
                        "container_id": container.container_id,
                        "size": container.size, "weight": container.weight,
                        "bay": None, "half": None, "col": None, "tier": None,
                        "placed": False,
                    })
                continue

            candidates.sort(
                key=lambda c: c[0].cumulative_score + c[5], reverse=True
            )
            top_k = candidates[:beam_width]

            # Best beam's chosen position (first in top_k) = label 1
            chosen_state, chosen_bay, chosen_half, chosen_col, chosen_tier, _ = top_k[0]

            # Extract features for ALL candidate (state, position) pairs
            seen: set = set()
            for state, bay, half, col, tier, _ in candidates:
                key = (id(state), bay, half, col, tier)
                if key in seen:
                    continue
                seen.add(key)

                pos = bay * 2 + (half if half is not None else 0)
                feat = _extract_features(
                    moment_z   = state.moment_z,
                    fp_w=state.fp_w, fs_w=state.fs_w,
                    ap_w=state.ap_w, as_w=state.as_w,
                    total_w    = state.total_w,
                    container  = container,
                    bay=pos, col=col, tier=tier,
                    ship_length = ship.length,
                    ship_width  = ship.width,
                    ship_height = ship.height,
                    max_weight  = weight_max,
                )
                label = 1.0 if (
                    state is chosen_state
                    and bay == chosen_bay
                    and half == chosen_half
                    and col == chosen_col
                    and tier == chosen_tier
                ) else 0.0

                X_list.append(feat)
                y_list.append(label)

            # Advance beams
            beams = [
                solver._expand(state, container, bay, half, col, tier, score)
                for state, bay, half, col, tier, score in top_k
            ]

    X = np.stack(X_list).astype(np.float32) if X_list else np.empty((0, N_FEATURES), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NeuralRankerSolver(BaseSolver):
    """Greedy solver using a trained MLP to score placement positions.

    Usage
    -----
    ::

        solver = NeuralRankerSolver(ship)
        solver.fit(n_episodes=200, beam_width=5, seed=42,
                   ship_params=PANAMAX_PARAMS)
        manifest = solver.load(containers)

        # Persist trained model
        solver.save("ranker.pkl")
        solver2 = NeuralRankerSolver.load_model(ship, "ranker.pkl")

    Parameters
    ----------
    ship : CargoShip
    hidden_layer_sizes : tuple
        MLP architecture (default: two layers of 128 units).
    max_weight : float
        Maximum possible container weight; used for feature normalisation.
    """

    def __init__(
        self,
        ship: CargoShip,
        hidden_layer_sizes: Tuple[int, ...] = (128, 128),
        max_weight: float = 28000.0,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for NeuralRankerSolver.  "
                "Install with: conda run -n personal pip install scikit-learn"
            )
        super().__init__(ship)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_weight         = max_weight
        self.manifest: List[Dict] = []
        self._fitted = False

        self._scaler = StandardScaler()
        self._mlp    = MLPClassifier(
            hidden_layer_sizes = hidden_layer_sizes,
            activation         = "relu",
            max_iter           = 500,
            random_state       = 42,
            early_stopping     = True,
            n_iter_no_change   = 20,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        n_episodes: int  = 200,
        beam_width: int  = 5,
        seed: int        = 42,
        ship_params: Optional[Dict[str, Any]] = None,
        n_20ft: int      = 60,
        n_40ft: int      = 25,
        n_20ft_max: int  = 0,
        n_40ft_max: int  = 0,
        weight_min: float = 2000.0,
    ) -> "NeuralRankerSolver":
        """Generate training data via Beam Search and fit the MLP."""
        import time as _time
        t0 = _time.perf_counter()

        if ship_params is None:
            ship_params = dict(
                length     = self.ship.length,
                base_width = self.ship.base_width,
                max_width  = self.ship.width,
                height     = self.ship.height,
                width_step = self.ship.width_step,
                max_weight = self.ship.max_weight,
            )

        X, y = generate_training_data(
            ship_params = ship_params,
            n_episodes  = n_episodes,
            beam_width  = beam_width,
            n_20ft      = n_20ft,
            n_40ft      = n_40ft,
            n_20ft_max  = n_20ft_max,
            n_40ft_max  = n_40ft_max,
            weight_min  = weight_min,
            weight_max  = self.max_weight,
            seed        = seed,
        )

        X_scaled = self._scaler.fit_transform(X)
        self._mlp.fit(X_scaled, y.astype(int))
        self._fitted = True

        elapsed = _time.perf_counter() - t0
        self.training_stats_: Dict[str, Any] = {
            "elapsed_s":        round(elapsed, 2),
            "n_episodes":       n_episodes,
            "n_samples":        int(X.shape[0]),
            "n_features":       int(X.shape[1]),
            "n_iter":           int(self._mlp.n_iter_),
            "loss_curve":       [round(v, 6) for v in self._mlp.loss_curve_],
            "val_score_curve":  [round(v, 6) for v in getattr(self._mlp, "validation_scores_", [])],
            "best_val_score":   round(float(getattr(self._mlp, "best_validation_score_", float("nan"))), 6),
        }
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted scaler + MLP to *path* (via joblib)."""
        joblib.dump({
            "scaler":         self._scaler,
            "mlp":            self._mlp,
            "training_stats": getattr(self, "training_stats_", {}),
        }, path)

    @classmethod
    def load_model(cls, ship: CargoShip, path: str, **kwargs) -> "NeuralRankerSolver":
        """Load a previously saved model from *path*."""
        data   = joblib.load(path)
        solver = cls(ship, **kwargs)
        solver._scaler        = data["scaler"]
        solver._mlp           = data["mlp"]
        solver._fitted        = True
        solver.training_stats_ = data.get("training_stats", {})
        return solver

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_position(
        self,
        loader: CargoLoader,
        container: ShippingContainer,
        bay: int, half, col: int, tier: int,
    ) -> float:
        pos = bay * 2 + (half if half is not None else 0)
        feat = _extract_features(
            moment_z    = loader._moment_z,
            fp_w=loader._fp_w, fs_w=loader._fs_w,
            ap_w=loader._ap_w, as_w=loader._as_w,
            total_w     = loader._total_w,
            container   = container,
            bay=pos, col=col, tier=tier,
            ship_length = self.ship.length,
            ship_width  = self.ship.width,
            ship_height = self.ship.height,
            max_weight  = self.max_weight,
        )
        X_scaled = self._scaler.transform(feat.reshape(1, -1))
        return float(self._mlp.predict_proba(X_scaled)[0, 1])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        if not self._fitted:
            raise RuntimeError(
                "NeuralRankerSolver must be fitted before calling load().  "
                "Call solver.fit(...) or NeuralRankerSolver.load_model(...)."
            )

        sorted_containers = sorted(containers, key=lambda c: (-c.weight, -c.size))
        # Use CargoLoader for running-sum tracking and position enumeration;
        # replace its scoring function with the MLP.
        loader = CargoLoader(self.ship)

        for container in sorted_containers:
            valid_positions = loader._enumerate_valid_positions(container)

            if not valid_positions:
                loader.manifest.append({
                    "container_id": container.container_id,
                    "size":   container.size,
                    "weight": container.weight,
                    "bay": None, "half": None, "col": None, "tier": None,
                    "placed": False,
                })
                continue

            best_bay, best_half, best_col, best_tier = max(
                valid_positions,
                key=lambda pos: self._score_position(loader, container, *pos),
            )

            best_pos = best_bay * 2 + (best_half if best_half is not None else 0)
            self.ship.place_container(container, best_pos, best_col, best_tier)
            loader._update_moments(container, best_bay, best_half, best_col, best_tier)
            loader.manifest.append({
                "container_id": container.container_id,
                "size":   container.size,
                "weight": container.weight,
                "bay":    best_bay,
                "half":   best_half,
                "col":    best_col,
                "tier":   best_tier,
                "placed": True,
            })

        self.manifest = loader.manifest
        return self.manifest
