import numpy as np
from typing import Tuple
from abc import abstractmethod


class Container:

    def __init__(self, length: int, max_width: int, height: int, max_weight: float):
        self.length = length
        self.width = max_width
        self.height = height
        self.max_weight = max_weight

        self.cargo_hold = self.create_cargo_hold()
        self.max_volume = self.compute_max_volume()

    def compute_max_volume(self):
        return self.length * self.width * self.height

    def create_cargo_hold(self):
        return np.zeros((self.length, self.width, self.height))

    @abstractmethod
    def additional_masking(self, package, position: Tuple[int, int, int], placement: np.ndarray):
        pass


class CargoShip(Container):

    def __init__(
        self,
        length: int,
        base_width: int,
        max_width: int,
        height: int,
        width_step: int = 1,
        max_weight: float = 50000.0,
    ):
        """
        Args:
            length:     Number of 20ft positions — must be even (bays are 40ft slots).
            base_width: Valid columns at tier 0 (keel level).
            max_width:  Valid columns at full expansion (beam).
            height:     Total tiers.
            width_step: Tiers elapsed before each +1 column expansion on each side.
            max_weight: Maximum total cargo weight (kg).
        """
        if length % 2 != 0:
            raise ValueError(
                f"length must be even (bays are 40ft slots = 2 × 20ft positions); got {length}"
            )
        # Store hull-shape params before super().__init__() calls create_cargo_hold()
        self.base_width = base_width
        self.width_step = width_step
        super().__init__(length, max_width, height, max_weight)
        self.occupied_mask = np.zeros((self.length, self.width, self.height), dtype=bool)
        self.total_weight: float = 0.0

    @property
    def n_bays(self) -> int:
        """Number of physical 40ft bays (= length // 2)."""
        return self.length // 2

    def compute_max_volume(self):
        # -1 hull cells reduce the total, giving the true valid capacity
        return self.length * self.width * self.height + int(np.sum(self.cargo_hold))

    def create_cargo_hold(self):
        """
        Create a 3D float array representing the cargo hold.
        -1.0 marks hull (invalid) cells, 0.0 marks empty valid cells.

        The hull expands outward from base_width at the keel (tier 0) to
        self.width (max_width) at full beam, widening by 1 column on each
        side every width_step tiers.  Above the full-beam tier the width
        stays constant — matching real cargo-hold geometry.
        """
        prism = np.zeros((self.length, self.width, self.height), dtype=float)

        max_expansions = (self.width - self.base_width) // 2

        for tier in range(self.height):
            expansions = min(tier // self.width_step, max_expansions)
            current_width = self.base_width + 2 * expansions
            left_col = (self.width - current_width) // 2
            right_col = left_col + current_width  # exclusive

            if left_col > 0:
                prism[:, :left_col, tier] = -1.0
            if right_col < self.width:
                prism[:, right_col:, tier] = -1.0

        return prism

    def additional_masking(
        self, package, position: Tuple[int, int, int], placement: np.ndarray
    ) -> bool:
        # Reject if any cell in the placement slice is a hull cell
        if np.any(placement == -1.0):
            return False
        # 40ft containers must start at even bays (real-world slot alignment)
        if package.size == 2 and position[0] % 2 != 0:
            return False
        return True

    def is_position_valid(self, container, bay: int, col: int, tier: int) -> bool:
        # 0. Weight constraint — reject if adding this container would sink the ship
        if self.total_weight + container.weight > self.max_weight:
            return False

        # 1. Boundary check
        if bay + container.size > self.length:
            return False
        if col >= self.width:
            return False
        if tier >= self.height:
            return False

        # 2. Extract the placement slice and run hull/alignment checks
        placement = self.cargo_hold[bay : bay + container.size, col, tier]
        if not self.additional_masking(container, (bay, col, tier), placement):
            return False

        # 3. Not already occupied
        if np.any(self.occupied_mask[bay : bay + container.size, col, tier]):
            return False

        # 4. Stacking support: every bay below must be occupied
        if tier > 0:
            if not np.all(self.occupied_mask[bay : bay + container.size, col, tier - 1]):
                return False

        return True

    def place_container(self, container, bay: int, col: int, tier: int) -> None:
        # Store weight per bay-cell so that summing over any region gives the
        # correct distributed weight (avoids double-counting 40ft containers).
        self.cargo_hold[bay : bay + container.size, col, tier] = (
            container.weight / container.size
        )
        self.occupied_mask[bay : bay + container.size, col, tier] = True
        self.total_weight += container.weight

    def port_starboard_balance(self) -> Tuple[float, float]:
        # Use ceiling division so the split matches center_col = width / 2.0
        # used in the scorer and visualizer. For odd widths (e.g. 13) this
        # puts the centre column (6) on the port side (cols 0-6) rather than
        # starboard, consistent with the float threshold col < 6.5.
        half = (self.width + 1) // 2
        port = float(np.sum(
            self.cargo_hold[:, :half, :] * self.occupied_mask[:, :half, :]
        ))
        starboard = float(np.sum(
            self.cargo_hold[:, half:, :] * self.occupied_mask[:, half:, :]
        ))
        return port, starboard

    def fore_aft_balance(self) -> Tuple[float, float]:
        half = self.length // 2
        fore = float(np.sum(
            self.cargo_hold[:half, :, :] * self.occupied_mask[:half, :, :]
        ))
        aft = float(np.sum(
            self.cargo_hold[half:, :, :] * self.occupied_mask[half:, :, :]
        ))
        return fore, aft

    def quadrant_balance(self) -> Tuple[float, float, float, float]:
        """Return (fore_port, fore_stbd, aft_port, aft_stbd) weight.

        Uses the same split thresholds as port_starboard_balance() and
        fore_aft_balance() so all three methods are mutually consistent.
        """
        half_l = self.length // 2
        half_w = (self.width + 1) // 2
        w = self.cargo_hold * self.occupied_mask
        fore_port = float(np.sum(w[:half_l, :half_w, :]))
        fore_stbd = float(np.sum(w[:half_l, half_w:, :]))
        aft_port  = float(np.sum(w[half_l:, :half_w, :]))
        aft_stbd  = float(np.sum(w[half_l:, half_w:, :]))
        return fore_port, fore_stbd, aft_port, aft_stbd


class ShippingContainer:
    """A cargo unit placed inside a CargoShip. Not a hold — standalone class."""

    _id_counter = 0

    def __init__(self, size: int, weight: float):
        # size: 1 = 20ft (1 bay), 2 = 40ft (2 bays)
        ShippingContainer._id_counter += 1
        self.container_id = ShippingContainer._id_counter
        self.size = size
        self.weight = weight

    def __repr__(self) -> str:
        ft = "20ft" if self.size == 1 else "40ft"
        return f"ShippingContainer(id={self.container_id}, {ft}, {self.weight:.1f}kg)"

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0
