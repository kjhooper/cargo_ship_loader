from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

from models import CargoShip, ShippingContainer


class BaseSolver(ABC):
    """Common interface for all cargo placement solvers."""

    def __init__(self, ship: CargoShip):
        self.ship = ship

    @abstractmethod
    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        """Place all containers and return the placement manifest."""
        ...

    def final_score(self) -> float:
        """Scalar quality metric in [0, 1]; higher is better.

        Returns the mean of port/stbd ratio, fore/aft ratio, and diagonal
        balance ratio.  A perfectly balanced load scores 1.0.
        """
        port_w, stbd_w = self.ship.port_starboard_balance()
        fore_w, aft_w  = self.ship.fore_aft_balance()
        fp, fs, ap, as_ = self.ship.quadrant_balance()

        max_ps = max(port_w, stbd_w)
        max_fa = max(fore_w, aft_w)
        d1, d2 = fp + as_, fs + ap
        max_d  = max(d1, d2)

        ps_ratio   = min(port_w, stbd_w) / max_ps if max_ps > 0 else 1.0
        fa_ratio   = min(fore_w, aft_w)  / max_fa if max_fa > 0 else 1.0
        diag_ratio = min(d1, d2)         / max_d  if max_d  > 0 else 1.0

        return (ps_ratio + fa_ratio + diag_ratio) / 3.0


class CargoLoader(BaseSolver):
    """Greedy single-pass solver.

    Scorer weights are now constructor parameters — identical defaults to the
    original hard-coded values, so existing code is fully backward-compatible.
    """

    def __init__(
        self,
        ship: CargoShip,
        k_gz: float      = 5.0,
        k_trim: float    = 4.0,
        k_list: float    = 4.0,
        k_diag: float    = 6.0,
        k_stacking: float = 0.5,
    ):
        super().__init__(ship)
        self.k_gz       = k_gz
        self.k_trim     = k_trim
        self.k_list     = k_list
        self.k_diag     = k_diag
        self.k_stacking = k_stacking

        self.manifest: List[Dict] = []
        # Running sums — all O(1) per placement
        self._moment_z = 0.0  # Σ wᵢ · tierᵢ  (for G_z / metacentric stability)
        self._fp_w     = 0.0  # fore-port weight
        self._fs_w     = 0.0  # fore-stbd weight
        self._ap_w     = 0.0  # aft-port weight
        self._as_w     = 0.0  # aft-stbd weight
        self._total_w  = 0.0

    # ------------------------------------------------------------------
    # Position enumeration
    # ------------------------------------------------------------------

    def _enumerate_valid_positions(
        self, container: ShippingContainer
    ) -> List[Tuple[int, int, int]]:
        """Return all (bay, col, tier) positions where this container fits.

        Iterates tier-major (lower tiers first), then by distance from the
        ship's centroidal axis (central positions first).  This ensures that
        when multiple positions score identically — as they all do on an empty
        ship — the most central position wins the tie, producing a natural
        centre-outward loading pattern.
        """
        center_bay = self.ship.length / 2.0
        center_col = self.ship.width  / 2.0
        bay_c_offset = (container.size - 1) / 2.0
        bays = sorted(range(self.ship.length),
                      key=lambda b: abs(b + bay_c_offset - center_bay))
        cols = sorted(range(self.ship.width),
                      key=lambda c: abs(c - center_col))
        valid = []
        for tier in range(self.ship.height):
            for col in cols:
                for bay in bays:
                    if self.ship.is_position_valid(container, bay, col, tier):
                        valid.append((bay, col, tier))
        return valid

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _update_moments(
        self, container: ShippingContainer, bay: int, col: int, tier: int
    ) -> None:
        bay_c = bay + (container.size - 1) / 2.0
        w = container.weight
        self._moment_z += w * tier
        self._total_w  += w
        is_fore = bay_c < self.ship.length / 2.0
        is_port = col   < self.ship.width  / 2.0
        if   is_fore and     is_port: self._fp_w += w
        elif is_fore and not is_port: self._fs_w += w
        elif is_port:                 self._ap_w += w
        else:                         self._as_w += w

    def _two_shorts_below(self, bay: int, col: int, tier: int) -> bool:
        """True if two distinct 20ft containers sit directly below (bay, bay+1)."""
        if tier == 0:
            return False
        below = tier - 1
        at_bay: Optional[int] = None
        at_bay_plus1: Optional[int] = None
        for entry in self.manifest:
            if not entry["placed"]:
                continue
            if entry["col"] == col and entry["tier"] == below and entry["size"] == 1:
                if entry["bay"] == bay:
                    at_bay = entry["container_id"]
                elif entry["bay"] == bay + 1:
                    at_bay_plus1 = entry["container_id"]
        return (
            at_bay is not None
            and at_bay_plus1 is not None
            and at_bay != at_bay_plus1
        )

    def _score_position(
        self, container: ShippingContainer, bay: int, col: int, tier: int
    ) -> float:
        w = container.weight
        bay_c = bay + (container.size - 1) / 2.0
        total_new = self._total_w + w

        # Projected quadrant weights after hypothetical placement
        is_fore = bay_c < self.ship.length / 2.0
        is_port = col   < self.ship.width  / 2.0
        new_fp = self._fp_w + w if (    is_fore and     is_port) else self._fp_w
        new_fs = self._fs_w + w if (    is_fore and not is_port) else self._fs_w
        new_ap = self._ap_w + w if (not is_fore and     is_port) else self._ap_w
        new_as = self._as_w + w if (not is_fore and not is_port) else self._as_w

        # Metacentric stability: minimise centre-of-gravity height
        G_z = (self._moment_z + w * tier) / total_new
        gz_norm = G_z / (self.ship.height - 1)

        # Fore-aft weight balance
        trim_norm = abs((new_fp + new_fs) - (new_ap + new_as)) / total_new

        # Port-starboard weight balance
        list_norm = abs((new_fp + new_ap) - (new_fs + new_as)) / total_new

        # Diagonal balance: penalise fore-port+aft-stbd or fore-stbd+aft-port bias
        # K_diag > K_list guarantees aft-port beats aft-stbd when fore-port is heavy
        diag_norm = abs((new_fp + new_as) - (new_fs + new_ap)) / total_new

        score  = -self.k_gz       * gz_norm
        score -= self.k_trim      * trim_norm
        score -= self.k_list      * list_norm
        score -= self.k_diag      * diag_norm

        if container.size == 2 and self._two_shorts_below(bay, col, tier):
            score += self.k_stacking
        return score

    # ------------------------------------------------------------------
    # Main loading loop
    # ------------------------------------------------------------------

    def load(self, containers: List[ShippingContainer]) -> List[Dict]:
        """Place all containers and return the placement manifest."""
        # Heaviest first; 40ft breaks ties (harder to place, reduce dead-ends)
        sorted_containers = sorted(containers, key=lambda c: (-c.weight, -c.size))

        for container in sorted_containers:
            valid_positions = self._enumerate_valid_positions(container)

            if not valid_positions:
                self.manifest.append(
                    {
                        "container_id": container.container_id,
                        "size": container.size,
                        "weight": container.weight,
                        "bay": None,
                        "col": None,
                        "tier": None,
                        "slot": None,
                        "placed": False,
                    }
                )
                continue

            best_bay, best_col, best_tier = max(
                valid_positions,
                key=lambda pos: self._score_position(container, *pos),
            )

            self.ship.place_container(container, best_bay, best_col, best_tier)
            self._update_moments(container, best_bay, best_col, best_tier)
            self.manifest.append(
                {
                    "container_id": container.container_id,
                    "size": container.size,
                    "weight": container.weight,
                    "bay": best_bay,
                    "col": best_col,
                    "tier": best_tier,
                    "slot": best_bay // 2,
                    "placed": True,
                }
            )

        return self.manifest
