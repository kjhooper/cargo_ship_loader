"""Deferred-placement solvers.

Key idea
--------
Standard greedy placement is order-dependent with no recovery: once a
late-stop container lands on top of an early-stop container the unloading
violation is locked in.

DeferSolver adds a *defer decision* at each step:

    for each container (heaviest / latest-stop / biggest first):
        find best available position (same scoring as CargoLoader)
        if _should_defer() → push to back of queue
        else               → place immediately

A container that has been deferred ``max_defers`` times is placed
unconditionally, guaranteeing termination.

Two implementations
-------------------
DeferSolver (rule-based)
    Defer iff placing in the best slot creates a non-zero unloading-order
    violation (_unload_penalty > 0).  No training required.

LearnedDeferSolver (MLP policy)
    An 8-feature MLP predicts P(defer) at each decision.  Trained via
    reward-weighted SIR: run stochastic episodes, weight (state, action)
    pairs by episode reward, resample with replacement, refit classifier.
    Falls back to the rule-based policy when unfitted.

Requires: scikit-learn, joblib  (LearnedDeferSolver only).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DEFER_FEATURES = 8   # length of feature vector per defer decision
_MAX_STOP_NORM   = 5   # denominator for facility normalisation


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


def _random_containers(
    n_20ft: int, n_40ft: int,
    weight_min: float, weight_max: float,
    seed: int,
    n_stops: int = 1,
) -> List[ShippingContainer]:
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    def fac() -> int:
        return rng.randint(1, max(n_stops, 1))
    conts = (
        [ShippingContainer(size=1,
                           weight=round(rng.uniform(weight_min, weight_max), 1),
                           facility=fac())
         for _ in range(n_20ft)]
        + [ShippingContainer(size=2,
                             weight=round(rng.uniform(weight_min, weight_max), 1),
                             facility=fac())
           for _ in range(n_40ft)]
    )
    rng.shuffle(conts)
    return conts


# ---------------------------------------------------------------------------
# Rule-based solver
# ---------------------------------------------------------------------------

class DeferSolver(CargoLoader):
    """Greedy placement with a rule-based defer policy.

    At each step the best position is computed with CargoLoader's scorer.
    If placing there would create an unloading-order violation the container
    is pushed to the back of the queue (up to ``max_defers`` times).

    Parameters
    ----------
    ship : CargoShip
    max_defers : int
        Maximum times any single container may be deferred before it is
        placed unconditionally.
    k_gz, k_trim, k_list, k_diag, k_stacking, k_unload : float
        CargoLoader scorer weights.
    """

    def __init__(
        self,
        ship: CargoShip,
        max_defers: int   = 2,
        k_gz: float       = 5.0,
        k_trim: float     = 4.0,
        k_list: float     = 4.0,
        k_diag: float     = 6.0,
        k_stacking: float = 0.5,
        k_unload: float   = 2.0,
    ):
        super().__init__(
            ship, k_gz=k_gz, k_trim=k_trim, k_list=k_list,
            k_diag=k_diag, k_stacking=k_stacking, k_unload=k_unload,
        )
        self.max_defers = max_defers

    # ------------------------------------------------------------------
    # Defer policy — override in subclass for a learned behaviour
    # ------------------------------------------------------------------

    def _defer_proba(
        self,
        container: ShippingContainer,
        bay: int, half, col: int, tier: int,
        queue_remaining: int,
        n_total: int,
        defer_count: int,
    ) -> float:
        """Return P(defer) for the current decision.  Rule: 1.0 if placing
        here creates a violation, 0.0 otherwise."""
        has_violation = self._unload_penalty(container, bay, half, col, tier) > 0.0
        return 1.0 if has_violation else 0.0

    def _should_defer(
        self,
        container: ShippingContainer,
        bay: int, half, col: int, tier: int,
        queue_remaining: int,
        n_total: int,
        defer_count: int,
    ) -> bool:
        return self._defer_proba(
            container, bay, half, col, tier,
            queue_remaining, n_total, defer_count,
        ) >= 0.5

    # ------------------------------------------------------------------
    # Main loading loop
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        n_total = len(containers)
        queue: List[ShippingContainer] = sorted(
            containers, key=lambda c: (-c.weight, -c.facility, -c.size)
        )
        defer_counts: Dict[int, int] = defaultdict(int)

        while queue:
            container = queue.pop(0)
            valid_positions = self._enumerate_valid_positions(container)

            if not valid_positions:
                self.manifest.append({
                    "container_id": container.container_id,
                    "size":     container.size,
                    "weight":   container.weight,
                    "facility": container.facility,
                    "bay": None, "half": None, "col": None, "tier": None,
                    "placed": False,
                })
                continue

            best_bay, best_half, best_col, best_tier = max(
                valid_positions,
                key=lambda p: self._score_position(container, *p),
            )

            d_count = defer_counts[container.container_id]
            if (
                d_count < self.max_defers
                and self._should_defer(
                    container, best_bay, best_half, best_col, best_tier,
                    len(queue), n_total, d_count,
                )
            ):
                defer_counts[container.container_id] += 1
                queue.append(container)
            else:
                pos = best_bay * 2 + (best_half if best_half is not None else 0)
                self.ship.place_container(container, pos, best_col, best_tier)
                self._update_moments(container, best_bay, best_half, best_col, best_tier)
                self.manifest.append({
                    "container_id": container.container_id,
                    "size":     container.size,
                    "weight":   container.weight,
                    "facility": container.facility,
                    "bay":  best_bay,
                    "half": best_half,
                    "col":  best_col,
                    "tier": best_tier,
                    "placed": True,
                })

        return self.manifest


# ---------------------------------------------------------------------------
# Learned policy solver
# ---------------------------------------------------------------------------

class LearnedDeferSolver(DeferSolver):
    """DeferSolver with an MLP defer policy trained via reward-weighted SIR.

    Feature vector (N_DEFER_FEATURES = 8) per defer decision
    ---------------------------------------------------------
    0  container.facility / _MAX_STOP_NORM   how late is this stop?
    1  container.weight   / ship.max_weight  how heavy?
    2  container.size / 2.0                  20 ft (0.5) vs 40 ft (1.0)
    3  unload_penalty_at_best_pos            violation fraction [0, 1]
    4  queue_remaining / n_total             fraction of queue left
    5  defer_count / max_defers              defers already used
    6  placed_weight / ship.max_weight       ship fill by weight
    7  tier_of_best_slot / (height - 1)      how high is the best slot?

    Training (SIR)
    --------------
    n_episodes stochastic rollouts are run.  At each defer decision the
    policy predicts P(defer); an epsilon-greedy action is sampled.  After all
    episodes the (feature, action) pairs are reward-weighted at the episode
    level (softmax with temperature beta), resampled with replacement (SIR),
    and the MLP is refit on the resulting dataset.

    Requires: scikit-learn, joblib.
    """

    def __init__(
        self,
        ship: CargoShip,
        max_defers: int               = 2,
        hidden_layer_sizes: Tuple[int, ...] = (32, 32),
        k_gz: float                   = 5.0,
        k_trim: float                 = 4.0,
        k_list: float                 = 4.0,
        k_diag: float                 = 6.0,
        k_stacking: float             = 0.5,
        k_unload: float               = 2.0,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for LearnedDeferSolver. "
                "Install with: conda run -n personal pip install scikit-learn"
            )
        super().__init__(
            ship, max_defers=max_defers,
            k_gz=k_gz, k_trim=k_trim, k_list=k_list,
            k_diag=k_diag, k_stacking=k_stacking, k_unload=k_unload,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self._fitted  = False
        self._scaler  = StandardScaler()
        self._mlp     = MLPClassifier(
            hidden_layer_sizes = hidden_layer_sizes,
            activation         = "relu",
            max_iter           = 500,
            random_state       = 42,
        )
        # Save ship geometry so fit() can restore state afterwards
        self._ship_params: dict = dict(
            length     = ship.length,
            base_width = ship.base_width,
            max_width  = ship.width,
            height     = ship.height,
            width_step = ship.width_step,
            max_weight = ship.max_weight,
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _defer_features(
        self,
        container: ShippingContainer,
        bay: int, half, col: int, tier: int,
        queue_remaining: int,
        n_total: int,
        defer_count: int,
    ) -> np.ndarray:
        max_w = max(self.ship.max_weight, 1.0)
        return np.array([
            container.facility / _MAX_STOP_NORM,
            container.weight   / max_w,
            container.size     / 2.0,
            self._unload_penalty(container, bay, half, col, tier),
            queue_remaining / max(n_total, 1),
            defer_count     / max(self.max_defers, 1),
            self._total_w   / max_w,
            tier            / max(self.ship.height - 1, 1),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Learned defer policy (overrides rule-based)
    # ------------------------------------------------------------------

    def _defer_proba(
        self,
        container: ShippingContainer,
        bay: int, half, col: int, tier: int,
        queue_remaining: int,
        n_total: int,
        defer_count: int,
    ) -> float:
        if not self._fitted:
            return super()._defer_proba(
                container, bay, half, col, tier,
                queue_remaining, n_total, defer_count,
            )
        feat   = self._defer_features(
            container, bay, half, col, tier,
            queue_remaining, n_total, defer_count,
        )
        feat_s = self._scaler.transform(feat.reshape(1, -1))
        classes = list(self._mlp.classes_)
        proba   = self._mlp.predict_proba(feat_s)[0]
        idx     = classes.index(1) if 1 in classes else -1
        return float(proba[idx]) if idx >= 0 else 0.0

    # ------------------------------------------------------------------
    # Stochastic rollout used during training
    # ------------------------------------------------------------------

    def _stochastic_load(
        self,
        containers: List[ShippingContainer],
        rng: random.Random,
        epsilon: float = 0.15,
    ) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
        """One training episode with epsilon-greedy defer decisions.

        self.ship must be a fresh (empty) ship before calling.

        Returns
        -------
        manifest    : placement results
        feat_list   : feature vectors, one per defer decision point
        action_list : corresponding binary actions (0=place, 1=defer)
        """
        # Reset running sums for this episode
        self.manifest  = []
        self._moment_z = self._fp_w = self._fs_w = self._ap_w = self._as_w = self._total_w = 0.0

        n_total = len(containers)
        queue: List[ShippingContainer] = sorted(
            containers, key=lambda c: (-c.weight, -c.facility, -c.size)
        )
        defer_counts: Dict[int, int] = defaultdict(int)
        feat_list:    List[np.ndarray] = []
        action_list:  List[int]        = []

        while queue:
            container = queue.pop(0)
            valid_positions = self._enumerate_valid_positions(container)

            if not valid_positions:
                self.manifest.append({
                    "container_id": container.container_id,
                    "size": container.size, "weight": container.weight,
                    "facility": container.facility,
                    "bay": None, "half": None, "col": None, "tier": None,
                    "placed": False,
                })
                continue

            best_bay, best_half, best_col, best_tier = max(
                valid_positions,
                key=lambda p: self._score_position(container, *p),
            )

            d_count = defer_counts[container.container_id]
            force   = d_count >= self.max_defers

            if force:
                action = 0
            else:
                feat  = self._defer_features(
                    container, best_bay, best_half, best_col, best_tier,
                    len(queue), n_total, d_count,
                )
                p_policy = self._defer_proba(
                    container, best_bay, best_half, best_col, best_tier,
                    len(queue), n_total, d_count,
                )
                # Epsilon-greedy: with probability epsilon, flip the decision
                p_actual = (1.0 - epsilon) * p_policy + epsilon * 0.5
                action   = 1 if rng.random() < p_actual else 0
                feat_list.append(feat)
                action_list.append(action)

            if action == 1:
                defer_counts[container.container_id] += 1
                queue.append(container)
            else:
                pos = best_bay * 2 + (best_half if best_half is not None else 0)
                self.ship.place_container(container, pos, best_col, best_tier)
                self._update_moments(container, best_bay, best_half, best_col, best_tier)
                self.manifest.append({
                    "container_id": container.container_id,
                    "size": container.size, "weight": container.weight,
                    "facility": container.facility,
                    "bay":  best_bay, "half": best_half,
                    "col":  best_col, "tier": best_tier,
                    "placed": True,
                })

        return self.manifest, feat_list, action_list

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        n_episodes:  int   = 200,
        n_20ft:      int   = 60,
        n_40ft:      int   = 25,
        weight_min:  float = 2_000.0,
        max_stops:   int   = 3,
        beta:        float = 5.0,
        epsilon:     float = 0.15,
        seed:        int   = 42,
        ship_params: Optional[dict] = None,
    ) -> "LearnedDeferSolver":
        """Train the defer policy via reward-weighted SIR.

        Parameters
        ----------
        n_episodes : int
            Number of stochastic rollouts.
        n_20ft, n_40ft : int
            Container counts per episode.
        weight_min : float
            Minimum container weight; max is taken from ship.max_weight.
        max_stops : int
            n_stops per episode is sampled uniformly from 1..max_stops.
        beta : float
            Softmax inverse-temperature for episode-level SIR weighting.
        epsilon : float
            Fraction of random (exploratory) defer decisions.
        seed : int
        ship_params : dict | None
            Override ship geometry for training manifests.  Useful when
            training on a reference ship and running inference on another.
        """
        ref_params = ship_params or self._ship_params
        weight_max = ref_params["max_weight"]
        rng = random.Random(seed)

        all_feats:   List[np.ndarray] = []
        all_actions: List[int]        = []
        rewards:     List[float]      = []
        ep_lengths:  List[int]        = []

        print(f"  [DeferRL] training {n_episodes} episodes…")
        for ep in range(n_episodes):
            if ep % 50 == 0:
                print(f"    episode {ep}/{n_episodes}", flush=True)
            ep_stops = rng.randint(1, max(max_stops, 1))
            ep_seed  = rng.randint(0, 2**31)
            conts    = _random_containers(
                n_20ft, n_40ft, weight_min, weight_max, ep_seed, n_stops=ep_stops,
            )
            self.ship = CargoShip(**ref_params)   # fresh ship for this episode

            manifest, feats, actions = self._stochastic_load(conts, rng, epsilon=epsilon)

            r = (BaseSolver.unloading_score(manifest) + self.final_score()) / 2.0
            rewards.append(r)
            ep_lengths.append(len(feats))
            all_feats.extend(feats)
            all_actions.extend(actions)

        # Restore solver state (ship + running sums) after training
        self.ship      = CargoShip(**self._ship_params)
        self.manifest  = []
        self._moment_z = self._fp_w = self._fs_w = self._ap_w = self._as_w = self._total_w = 0.0

        if not all_feats:
            return self  # no defer decisions were ever made; nothing to learn

        # ── SIR: reward-weight at episode level, resample per-decision pairs ──
        rewards_arr = np.array(rewards)
        log_alpha   = beta * (rewards_arr - rewards_arr.max())
        alpha       = np.exp(log_alpha)
        alpha      /= alpha.sum()

        X_raw = np.stack(all_feats)
        y_raw = np.array(all_actions)

        per_decision_w  = np.repeat(alpha, ep_lengths)
        per_decision_w /= per_decision_w.sum()

        sir_rng = np.random.default_rng(seed)
        indices = sir_rng.choice(len(X_raw), size=len(X_raw), replace=True,
                                 p=per_decision_w)
        X_sir = X_raw[indices]
        y_sir = y_raw[indices]

        # MLPClassifier requires both classes; ensure neither is absent
        for cls_val in (0, 1):
            if cls_val not in y_sir:
                mask = y_raw == cls_val
                if mask.any():
                    X_sir = np.vstack([X_sir, X_raw[mask][:1]])
                    y_sir = np.append(y_sir, cls_val)

        self._scaler.fit(X_sir)
        self._mlp.fit(self._scaler.transform(X_sir), y_sir)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted scaler and MLP to path (via joblib)."""
        joblib.dump({
            "scaler":             self._scaler,
            "mlp":                self._mlp,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_defers":         self.max_defers,
        }, path)

    @classmethod
    def load_model(
        cls, ship: CargoShip, path: str, **kwargs
    ) -> "LearnedDeferSolver":
        """Load a previously saved model from path."""
        data   = joblib.load(path)
        solver = cls(
            ship,
            max_defers         = data.get("max_defers",         2),
            hidden_layer_sizes = data.get("hidden_layer_sizes", (32, 32)),
            **kwargs,
        )
        solver._scaler  = data["scaler"]
        solver._mlp     = data["mlp"]
        solver._fitted  = True
        return solver
