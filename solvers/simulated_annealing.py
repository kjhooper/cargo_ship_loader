"""Simulated Annealing solver (H3).

Starting from the greedy solution, the algorithm repeatedly:
  1. Picks a randomly chosen "top" container — one with nothing stacked above it.
  2. Temporarily removes it from the ship.
  3. Finds all valid positions in the modified ship.
  4. Proposes a random new position.
  5. Accepts the move if it improves quality; accepts worse moves with
     probability exp(ΔQ / T) where T cools geometrically.

The best state seen during the search is always retained.

Complexity: O(I × m)   I = iterations, m = valid positions per move.
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Optional, Tuple

import numpy as np

from models import CargoShip, ShippingContainer
from algorithm import BaseSolver, CargoLoader


class SimulatedAnnealingSolver(BaseSolver):
    """Simulated Annealing solver.

    Parameters
    ----------
    ship : CargoShip
        Empty ship to load containers into.
    n_iterations : int
        Total SA iterations (move proposals).
    T_start : float
        Initial temperature.  ``final_score`` is in [0, 1]; a ΔQ of ~0.01
        represents a 1 % quality change, so T_start=0.05 is exploratory.
    cooling : float
        Geometric cooling factor applied after each iteration.
    seed : int | None
        Random seed for reproducibility.
    k_gz, k_trim, k_list, k_diag, k_stacking : float
        Scorer weights for the initial greedy solution.
    """

    def __init__(
        self,
        ship: CargoShip,
        n_iterations: int  = 2000,
        T_start: float     = 0.05,
        cooling: float     = 0.999,
        seed: Optional[int] = None,
        k_gz: float        = 5.0,
        k_trim: float      = 4.0,
        k_list: float      = 4.0,
        k_diag: float      = 6.0,
        k_stacking: float  = 0.5,
    ):
        super().__init__(ship)
        self.n_iterations = n_iterations
        self.T_start      = T_start
        self.cooling      = cooling
        self.rng          = random.Random(seed)
        self.k_gz         = k_gz
        self.k_trim       = k_trim
        self.k_list       = k_list
        self.k_diag       = k_diag
        self.k_stacking   = k_stacking
        self.manifest: List[Dict] = []

    # ------------------------------------------------------------------
    # Ship-mutation helpers (do NOT belong in the model layer)
    # ------------------------------------------------------------------

    def _place(self, bay: int, col: int, tier: int, size: int, weight: float) -> None:
        self.ship.cargo_hold[bay:bay + size, col, tier] = weight / size
        self.ship.occupied_mask[bay:bay + size, col, tier] = True
        self.ship.total_weight += weight

    def _remove(self, bay: int, col: int, tier: int, size: int, weight: float) -> None:
        self.ship.cargo_hold[bay:bay + size, col, tier] = 0.0
        self.ship.occupied_mask[bay:bay + size, col, tier] = False
        self.ship.total_weight -= weight

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _find_top_containers(self) -> List[Tuple[int, int, int, int, int, float]]:
        """Return (manifest_idx, bay, col, tier, size, weight) for top containers.

        A "top" container has no other container occupying any cell directly
        above it.
        """
        top = []
        for i, entry in enumerate(self.manifest):
            if not entry["placed"]:
                continue
            bay, col, tier = entry["bay"], entry["col"], entry["tier"]
            size, weight   = entry["size"], entry["weight"]
            if tier + 1 < self.ship.height:
                if np.any(self.ship.occupied_mask[bay:bay + size, col, tier + 1]):
                    continue
            top.append((i, bay, col, tier, size, weight))
        return top

    def _enumerate_valid_positions(
        self,
        size: int,
        weight: float,
        exclude: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        """Enumerate valid positions for a (size, weight) container,
        excluding *exclude* (the original position)."""
        # Reuse CargoLoader's enumeration by creating a thin wrapper
        c = _make_container(size, weight)
        loader = CargoLoader(self.ship)
        positions = loader._enumerate_valid_positions(c)
        return [p for p in positions if p != exclude]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        # --- Step 1: greedy warm start ---
        loader = CargoLoader(
            self.ship,
            k_gz=self.k_gz, k_trim=self.k_trim,
            k_list=self.k_list, k_diag=self.k_diag,
            k_stacking=self.k_stacking,
        )
        self.manifest = loader.load(containers)

        current_score = self.final_score()
        best_score    = current_score

        # Snapshot best state
        best_cargo  = self.ship.cargo_hold.copy()
        best_mask   = self.ship.occupied_mask.copy()
        best_weight = self.ship.total_weight
        best_manifest = [dict(e) for e in self.manifest]

        T = self.T_start

        # --- Step 2: SA search ---
        for _ in range(self.n_iterations):
            top = self._find_top_containers()
            if not top:
                break

            idx, bay, col, tier, size, weight = self.rng.choice(top)
            original_entry = dict(self.manifest[idx])

            # Temporarily remove the container
            self._remove(bay, col, tier, size, weight)

            valid = self._enumerate_valid_positions(size, weight, exclude=(bay, col, tier))
            if not valid:
                # No alternative positions — restore and skip
                self._place(bay, col, tier, size, weight)
                T *= self.cooling
                continue

            new_bay, new_col, new_tier = self.rng.choice(valid)
            self._place(new_bay, new_col, new_tier, size, weight)

            # Update manifest entry tentatively
            self.manifest[idx] = {
                "container_id": original_entry["container_id"],
                "size":   size,
                "weight": weight,
                "bay":    new_bay,
                "col":    new_col,
                "tier":   new_tier,
                "slot":   new_bay // 2,
                "placed": True,
            }

            new_score = self.final_score()
            delta     = new_score - current_score

            if delta > 0 or (T > 0 and self.rng.random() < math.exp(delta / T)):
                # Accept
                current_score = new_score
                if new_score > best_score:
                    best_score    = new_score
                    best_cargo    = self.ship.cargo_hold.copy()
                    best_mask     = self.ship.occupied_mask.copy()
                    best_weight   = self.ship.total_weight
                    best_manifest = [dict(e) for e in self.manifest]
            else:
                # Reject — undo move
                self._remove(new_bay, new_col, new_tier, size, weight)
                self._place(bay, col, tier, size, weight)
                self.manifest[idx] = original_entry

            T *= self.cooling

        # --- Step 3: restore best state ---
        self.ship.cargo_hold    = best_cargo
        self.ship.occupied_mask = best_mask
        self.ship.total_weight  = best_weight
        self.manifest           = best_manifest
        return self.manifest


# ---------------------------------------------------------------------------
# Module-level helper (not exported in __init__)
# ---------------------------------------------------------------------------

def _make_container(size: int, weight: float) -> ShippingContainer:
    """Create a ShippingContainer bypassing the global ID counter."""
    c = ShippingContainer.__new__(ShippingContainer)
    c.size         = size
    c.weight       = weight
    c.container_id = -1   # placeholder — not used for validity checks
    return c
