"""Beam Search solver (H2).

At each container-placement step the solver maintains the top-K partial
solutions ("beams").  For every beam, all valid positions for the next
container are enumerated and scored.  The K (beam, position) pairs with the
highest cumulative score are retained as the next generation of beams.

K = 1 is identical to the greedy CargoLoader.  Practical range: 3–20.
Complexity: O(n × K × m)  where n = containers, m = valid positions per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from models import CargoShip, ShippingContainer
from algorithm import BaseSolver, CargoLoader


@dataclass
class _BeamState:
    """Lightweight snapshot of one beam's state.

    Storing raw numpy arrays (instead of a full CargoShip) keeps memory
    usage modest and avoids the overhead of re-running hull-geometry setup
    for every deepcopy.
    """
    cargo_hold:    np.ndarray   # shape (length, width, height); -1=hull, 0=empty, >0=weight/size
    occupied_mask: np.ndarray   # shape (length, width, height); dtype bool
    total_weight:  float

    # Running sums — mirrors the ones maintained by CargoLoader
    moment_z: float
    fp_w:     float
    fs_w:     float
    ap_w:     float
    as_w:     float
    total_w:  float

    manifest:          List[Dict]
    cumulative_score:  float


class BeamSearchSolver(BaseSolver):
    """Beam Search solver.

    Parameters
    ----------
    ship : CargoShip
        Empty ship to load containers into.
    beam_width : int
        Number of beams (K) to maintain.  K=1 reproduces greedy behaviour.
    k_gz, k_trim, k_list, k_diag, k_stacking : float
        Scorer weights forwarded to the internal CargoLoader scorer.
        Defaults match the hand-tuned greedy weights.
    """

    def __init__(
        self,
        ship: CargoShip,
        beam_width: int   = 5,
        k_gz: float       = 5.0,
        k_trim: float     = 4.0,
        k_list: float     = 4.0,
        k_diag: float     = 6.0,
        k_stacking: float = 0.5,
    ):
        super().__init__(ship)
        self.beam_width  = beam_width
        self.k_gz        = k_gz
        self.k_trim      = k_trim
        self.k_list      = k_list
        self.k_diag      = k_diag
        self.k_stacking  = k_stacking
        self.manifest: List[Dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_state(self) -> _BeamState:
        return _BeamState(
            cargo_hold    = self.ship.cargo_hold.copy(),
            occupied_mask = self.ship.occupied_mask.copy(),
            total_weight  = float(self.ship.total_weight),
            moment_z=0.0, fp_w=0.0, fs_w=0.0, ap_w=0.0, as_w=0.0, total_w=0.0,
            manifest=[],
            cumulative_score=0.0,
        )

    def _tmp_ship(self, state: _BeamState) -> CargoShip:
        """Build a temporary CargoShip that mirrors *state* (no hull recompute)."""
        s = CargoShip(
            length     = self.ship.length,
            base_width = self.ship.base_width,
            max_width  = self.ship.width,
            height     = self.ship.height,
            width_step = self.ship.width_step,
            max_weight = self.ship.max_weight,
        )
        s.cargo_hold    = state.cargo_hold
        s.occupied_mask = state.occupied_mask
        s.total_weight  = state.total_weight
        return s

    def _loader_for_state(self, state: _BeamState) -> CargoLoader:
        """Thin CargoLoader wrapping *state* for enumeration + scoring only."""
        loader = CargoLoader(
            self._tmp_ship(state),
            k_gz=self.k_gz, k_trim=self.k_trim,
            k_list=self.k_list, k_diag=self.k_diag,
            k_stacking=self.k_stacking,
        )
        loader._moment_z = state.moment_z
        loader._fp_w     = state.fp_w
        loader._fs_w     = state.fs_w
        loader._ap_w     = state.ap_w
        loader._as_w     = state.as_w
        loader._total_w  = state.total_w
        loader.manifest  = state.manifest  # for _two_shorts_below stacking bonus
        return loader

    def _expand(
        self,
        state: _BeamState,
        container: ShippingContainer,
        bay: int,
        half,
        col: int,
        tier: int,
        step_score: float,
    ) -> _BeamState:
        """Return a new BeamState with *container* placed at (bay, half, col, tier)."""
        pos        = bay * 2 + (half if half is not None else 0)
        new_cargo  = state.cargo_hold.copy()
        new_mask   = state.occupied_mask.copy()
        w          = container.weight
        sz         = container.size

        new_cargo[pos:pos + sz, col, tier] = w / sz
        new_mask [pos:pos + sz, col, tier] = True

        bay_c   = pos + (sz - 1) / 2.0
        is_fore = bay_c < self.ship.length / 2.0
        is_port = col   < self.ship.width  / 2.0

        entry: Dict = {
            "container_id": container.container_id,
            "size":   sz,
            "weight": w,
            "bay":    bay,
            "half":   half,
            "col":    col,
            "tier":   tier,
            "placed": True,
        }

        return _BeamState(
            cargo_hold    = new_cargo,
            occupied_mask = new_mask,
            total_weight  = state.total_weight + w,
            moment_z = state.moment_z + w * tier,
            fp_w = state.fp_w + (w if is_fore and     is_port else 0.0),
            fs_w = state.fs_w + (w if is_fore and not is_port else 0.0),
            ap_w = state.ap_w + (w if not is_fore and     is_port else 0.0),
            as_w = state.as_w + (w if not is_fore and not is_port else 0.0),
            total_w = state.total_w + w,
            manifest          = state.manifest + [entry],
            cumulative_score  = state.cumulative_score + step_score,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        sorted_containers = sorted(containers, key=lambda c: (-c.weight, -c.size))
        beams: List[_BeamState] = [self._initial_state()]

        for container in sorted_containers:
            # Collect every (beam, position, step_score) tuple across all beams
            candidates: List[Tuple] = []
            for state in beams:
                loader = self._loader_for_state(state)
                for (bay, half, col, tier) in loader._enumerate_valid_positions(container):
                    score = loader._score_position(container, bay, half, col, tier)
                    candidates.append((state, bay, half, col, tier, score))

            if not candidates:
                # No beam can place this container — mark unplaced in all beams
                for state in beams:
                    state.manifest.append({
                        "container_id": container.container_id,
                        "size":   container.size,
                        "weight": container.weight,
                        "bay": None, "half": None, "col": None, "tier": None,
                        "placed": False,
                    })
                continue

            # Keep top-K by (cumulative_score + step_score)
            candidates.sort(
                key=lambda c: c[0].cumulative_score + c[5], reverse=True
            )
            top_k = candidates[: self.beam_width]

            beams = [
                self._expand(state, container, bay, half, col, tier, score)
                for state, bay, half, col, tier, score in top_k
            ]

        # Pick the beam with the highest cumulative score
        best = max(beams, key=lambda b: b.cumulative_score)

        # Propagate best beam's ship state into self.ship so final_score() works
        self.ship.cargo_hold    = best.cargo_hold
        self.ship.occupied_mask = best.occupied_mask
        self.ship.total_weight  = best.total_weight

        self.manifest = best.manifest
        return self.manifest
