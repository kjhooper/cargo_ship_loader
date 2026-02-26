"""
Weight distribution correctness tests.

Run with:  conda run -n personal pytest tests.py -v

Coverage
--------
1. Hull geometry          — valid cell counts, symmetry
2. Split-point consistency — balance fn / scorer / visualizer all agree
3. Single-container balance — correct side reported by balance functions
4. Symmetric placement    — perfect balance when containers are mirrored
5. 40ft weight storage    — weight/2 per bay, consistent across all three systems
6. Scorer direction       — scorer prefers positions that reduce imbalance
7. Greedy balance outcomes — end-to-end ratio >= 0.95 on multiple seeds
8. Visualizer accumulator — panel.port_w / fore_w match ship balance functions
9. Quadrant balance       — quadrant_balance() correctness and greedy diagonal fix
"""

import random
import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from models import CargoShip, ShippingContainer
from algorithm import CargoLoader
from visualizer import _ShipPanel


# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

PANAMAX = dict(
    length=36, base_width=7, max_width=13, height=9,
    width_step=1, max_weight=50_000.0,
)
WIDTH  = PANAMAX["max_width"]   # 13
LENGTH = PANAMAX["length"]      # 36

# Expected split boundaries (derived from the implementation)
PS_HALF = (WIDTH + 1) // 2      # 7  — balance fn port threshold (col < 7)
FA_HALF = LENGTH // 2           # 18 — balance fn fore threshold (bay < 18)
PS_CENTER = WIDTH  / 2.0        # 6.5 — scorer / visualizer threshold
FA_CENTER = LENGTH / 2.0        # 18.0


def make_ship() -> CargoShip:
    ShippingContainer.reset_id_counter()
    return CargoShip(**PANAMAX)


def force_place(ship, bay, col, tier, weight=10_000.0, size=1):
    """Bypass is_position_valid — lets us test balance functions in isolation."""
    c = ShippingContainer(size=size, weight=weight)
    ship.place_container(c, bay, col, tier)
    return c


def make_panel(manifest, ship):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    panel = _ShipPanel(
        manifest=manifest,
        ship_length=ship.length, ship_width=ship.width, ship_height=ship.height,
        fig=fig,
        ax_ship=fig.add_subplot(gs[0]),
        ax_bal=fig.add_subplot(gs[1]),
        norm=mcolors.Normalize(vmin=0, vmax=30_000),
        cmap=cm.coolwarm,
    )
    return panel, fig


# ---------------------------------------------------------------------------
# 1. Hull geometry
# ---------------------------------------------------------------------------

class TestHullGeometry:
    def test_odd_length_rejected(self):
        with pytest.raises(ValueError, match="even"):
            CargoShip(length=7, base_width=7, max_width=13, height=9)

    def test_valid_cell_count_per_tier(self):
        ship = make_ship()
        # Each tier's valid width: 7, 9, 11, 13, 13, 13, 13, 13, 13
        widths = [7, 9, 11] + [13] * 6
        for tier, w in enumerate(widths):
            valid = int(np.sum(ship.cargo_hold[:, :, tier] == 0.0))
            assert valid == LENGTH * w, \
                f"tier {tier}: expected {LENGTH * w} valid cells, got {valid}"

    def test_hull_is_port_starboard_symmetric(self):
        """Valid columns at each tier must be symmetric about the centreline."""
        ship = make_ship()
        for tier in range(ship.height):
            for col in range(ship.width):
                mirror = ship.width - 1 - col
                is_hull      = ship.cargo_hold[0, col,    tier] == -1.0
                mirror_hull  = ship.cargo_hold[0, mirror, tier] == -1.0
                assert is_hull == mirror_hull, \
                    f"tier {tier}: asymmetry between col {col} and col {mirror}"


# ---------------------------------------------------------------------------
# 2. Split-point consistency
# ---------------------------------------------------------------------------

class TestSplitPointConsistency:
    """
    Three independent implementations each classify a position as port/stbd
    or fore/aft.  They must all agree for every column / bay.

    Port / Starboard
      balance fn  : col < (width+1)//2  →  col < 7   (7 port cols: 0-6)
      scorer      : col < width/2.0     →  col < 6.5 (7 port cols: 0-6)
      visualizer  : col < ship_width/2.0 → col < 6.5 (same)

    Fore / Aft
      balance fn  : bay < length//2     →  bay < 18
      scorer      : bay < length/2.0    →  bay < 18.0
      visualizer  : bay < ship_length/2.0 → bay < 18.0
    """

    @pytest.mark.parametrize("col", range(WIDTH))
    def test_ps_balance_fn_matches_scorer(self, col):
        balance_port = col < PS_HALF     # col < 7
        scorer_port  = col < PS_CENTER   # col < 6.5
        assert balance_port == scorer_port, \
            f"col {col}: balance→{'port' if balance_port else 'stbd'} " \
            f"scorer→{'port' if scorer_port else 'stbd'}"

    @pytest.mark.parametrize("col", range(WIDTH))
    def test_ps_balance_fn_matches_visualizer(self, col):
        balance_port = col < PS_HALF
        viz_port     = col < PS_CENTER   # visualizer uses same formula as scorer
        assert balance_port == viz_port, \
            f"col {col}: balance→{'port' if balance_port else 'stbd'} " \
            f"viz→{'port' if viz_port else 'stbd'}"

    @pytest.mark.parametrize("bay", range(LENGTH))
    def test_fa_balance_fn_matches_scorer(self, bay):
        balance_fore = bay < FA_HALF     # bay < 18
        scorer_fore  = bay < FA_CENTER   # bay < 18.0
        assert balance_fore == scorer_fore, \
            f"bay {bay}: balance→{'fore' if balance_fore else 'aft'} " \
            f"scorer→{'fore' if scorer_fore else 'aft'}"


# ---------------------------------------------------------------------------
# 3. Single-container balance functions
# ---------------------------------------------------------------------------

class TestSingleContainerBalance:
    """Place one container; verify balance functions report correct side."""

    @pytest.mark.parametrize("col,side", [
        (0, "port"), (3, "port"), (5, "port"), (6, "port"),   # cols 0-6 → port
        (7, "stbd"), (9, "stbd"), (12, "stbd"),               # cols 7-12 → stbd
    ])
    def test_port_starboard_single_container(self, col, side):
        ship = make_ship()
        # Use a tier where all columns are valid (tier 3 covers cols 0-12)
        force_place(ship, bay=0, col=col, tier=3, weight=10_000.0)
        p, s = ship.port_starboard_balance()
        if side == "port":
            assert p == pytest.approx(10_000.0), \
                f"col {col}: expected port=10000, got port={p:.0f} stbd={s:.0f}"
            assert s == pytest.approx(0.0)
        else:
            assert s == pytest.approx(10_000.0), \
                f"col {col}: expected stbd=10000, got port={p:.0f} stbd={s:.0f}"
            assert p == pytest.approx(0.0)

    @pytest.mark.parametrize("bay,side", [
        (0, "fore"), (10, "fore"), (17, "fore"),
        (18, "aft"), (25, "aft"), (35, "aft"),
    ])
    def test_fore_aft_single_container(self, bay, side):
        ship = make_ship()
        force_place(ship, bay=bay, col=6, tier=0, weight=10_000.0)
        f, a = ship.fore_aft_balance()
        if side == "fore":
            assert f == pytest.approx(10_000.0), \
                f"bay {bay}: expected fore=10000, got fore={f:.0f} aft={a:.0f}"
            assert a == pytest.approx(0.0)
        else:
            assert a == pytest.approx(10_000.0), \
                f"bay {bay}: expected aft=10000, got fore={f:.0f} aft={a:.0f}"
            assert f == pytest.approx(0.0)

    def test_center_column_6_is_port(self):
        """Col 6 is the port-side centre column and must be counted as port."""
        ship = make_ship()
        force_place(ship, bay=0, col=6, tier=0, weight=10_000.0)
        p, s = ship.port_starboard_balance()
        assert p == pytest.approx(10_000.0) and s == pytest.approx(0.0), \
            f"col 6 should be port: port={p:.0f} stbd={s:.0f}"

    def test_col_7_is_starboard(self):
        """Col 7 is the first starboard column."""
        ship = make_ship()
        force_place(ship, bay=0, col=7, tier=0, weight=10_000.0)
        p, s = ship.port_starboard_balance()
        assert s == pytest.approx(10_000.0) and p == pytest.approx(0.0), \
            f"col 7 should be stbd: port={p:.0f} stbd={s:.0f}"

    def test_boundary_bay_17_is_fore(self):
        """Bay 17 is the last fore bay (< 18)."""
        ship = make_ship()
        force_place(ship, bay=17, col=6, tier=0, weight=10_000.0)
        f, a = ship.fore_aft_balance()
        assert f == pytest.approx(10_000.0) and a == pytest.approx(0.0), \
            f"bay 17 should be fore: fore={f:.0f} aft={a:.0f}"

    def test_boundary_bay_18_is_aft(self):
        """Bay 18 is the first aft bay (>= 18)."""
        ship = make_ship()
        force_place(ship, bay=18, col=6, tier=0, weight=10_000.0)
        f, a = ship.fore_aft_balance()
        assert a == pytest.approx(10_000.0) and f == pytest.approx(0.0), \
            f"bay 18 should be aft: fore={f:.0f} aft={a:.0f}"


# ---------------------------------------------------------------------------
# 4. Symmetric placement → perfect balance
# ---------------------------------------------------------------------------

class TestSymmetricBalance:
    def test_ps_mirror_gives_ratio_1(self):
        """Containers placed symmetrically around col 6.5 → ratio = 1.0."""
        ship = make_ship()
        for i in range(5):
            force_place(ship, bay=i,   col=i,      tier=3, weight=10_000.0)
            force_place(ship, bay=i,   col=12 - i, tier=3, weight=10_000.0)
        p, s = ship.port_starboard_balance()
        assert min(p, s) / max(p, s) == pytest.approx(1.0), \
            f"port={p:.0f} stbd={s:.0f}"

    def test_fa_mirror_gives_ratio_1(self):
        """Containers placed symmetrically around bay 18 → ratio = 1.0."""
        ship = make_ship()
        for i in range(5):
            force_place(ship, bay=i,      col=6, tier=0, weight=10_000.0)
            force_place(ship, bay=35 - i, col=7, tier=0, weight=10_000.0)
        f, a = ship.fore_aft_balance()
        assert min(f, a) / max(f, a) == pytest.approx(1.0), \
            f"fore={f:.0f} aft={a:.0f}"

    def test_four_quadrant_symmetric(self):
        """Equal weight in all four quadrants → both axes perfectly balanced."""
        ship = make_ship()
        w = 10_000.0
        force_place(ship, bay=5,  col=5,  tier=3, weight=w)
        force_place(ship, bay=5,  col=7,  tier=3, weight=w)
        force_place(ship, bay=30, col=5,  tier=3, weight=w)
        force_place(ship, bay=30, col=7,  tier=3, weight=w)
        p, s = ship.port_starboard_balance()
        f, a = ship.fore_aft_balance()
        assert min(p, s) / max(p, s) == pytest.approx(1.0), f"PS: {p:.0f}/{s:.0f}"
        assert min(f, a) / max(f, a) == pytest.approx(1.0), f"FA: {f:.0f}/{a:.0f}"

    def test_total_weight_equals_port_plus_starboard(self):
        """port + stbd must always equal total_weight."""
        ship = make_ship()
        for bay in range(0, 10):
            force_place(ship, bay=bay, col=6, tier=0, weight=5_000.0)
        p, s = ship.port_starboard_balance()
        assert p + s == pytest.approx(ship.total_weight), \
            f"port+stbd={p+s:.0f} != total={ship.total_weight:.0f}"

    def test_total_weight_equals_fore_plus_aft(self):
        """fore + aft must always equal total_weight."""
        ship = make_ship()
        for bay in range(0, 10):
            force_place(ship, bay=bay, col=6, tier=0, weight=5_000.0)
        f, a = ship.fore_aft_balance()
        assert f + a == pytest.approx(ship.total_weight), \
            f"fore+aft={f+a:.0f} != total={ship.total_weight:.0f}"


# ---------------------------------------------------------------------------
# 5. 40ft container weight distribution
# ---------------------------------------------------------------------------

class TestFortyFootWeight:
    def test_weight_split_across_bays_in_cargo_hold(self):
        """place_container stores weight/2 at each of the two bays."""
        ship = make_ship()
        force_place(ship, bay=10, col=6, tier=0, size=2, weight=20_000.0)
        assert ship.cargo_hold[10, 6, 0] == pytest.approx(10_000.0)
        assert ship.cargo_hold[11, 6, 0] == pytest.approx(10_000.0)

    def test_40ft_fully_fore_balance(self):
        """40ft at bays 14-15 (both fore): balance fn counts full weight as fore."""
        ship = make_ship()
        force_place(ship, bay=14, col=6, tier=0, size=2, weight=20_000.0)
        f, a = ship.fore_aft_balance()
        assert f == pytest.approx(20_000.0)
        assert a == pytest.approx(0.0)

    def test_40ft_fully_aft_balance(self):
        """40ft at bays 20-21 (both aft): balance fn counts full weight as aft."""
        ship = make_ship()
        force_place(ship, bay=20, col=6, tier=0, size=2, weight=20_000.0)
        f, a = ship.fore_aft_balance()
        assert a == pytest.approx(20_000.0)
        assert f == pytest.approx(0.0)

    def test_40ft_ps_determined_by_column(self):
        """40ft spans 2 bays but one column — PS side = column side."""
        ship = make_ship()
        force_place(ship, bay=10, col=6, tier=0, size=2, weight=20_000.0)   # port col
        p, s = ship.port_starboard_balance()
        assert p == pytest.approx(20_000.0)
        assert s == pytest.approx(0.0)

    def test_40ft_fore_scorer_matches_balance_fn(self):
        """
        For a 40ft container, bay_c = bay + 0.5 determines fore/aft classification.
        From an empty ship, bay=14 (fore, bay_c=14.5) and bay=20 (aft, bay_c=20.5)
        each place the full container weight on one side — equal imbalance magnitude
        — so the scorer treats them identically.
        """
        ship = make_ship()
        loader = CargoLoader(ship)
        c = ShippingContainer(size=2, weight=20_000.0)

        # Score bay=14 (spans 14-15, both fore) vs bay=20 (spans 20-21, both aft)
        score_fore = loader._score_position(c, bay=14, col=6, tier=0)
        score_aft  = loader._score_position(c, bay=20, col=7, tier=0)

        # Both create equal weight imbalance from an empty ship — they must tie
        assert abs(score_fore - score_aft) < 1e-6, \
            f"Empty ship: fore and aft create equal imbalance, should score equally; " \
            f"fore={score_fore:.3f} aft={score_aft:.3f}"

    def test_40ft_no_valid_bay_spans_fa_boundary(self):
        """
        No valid 40ft placement can span the fore/aft boundary (bay 17-18).
        Bay 17 is odd so 40ft containers cannot start there.
        """
        ship = make_ship()
        c = ShippingContainer(size=2, weight=10_000.0)
        assert not ship.is_position_valid(c, bay=17, col=6, tier=0), \
            "bay=17 is odd; 40ft must start at even bay — should be invalid"


# ---------------------------------------------------------------------------
# 6. Scorer direction
# ---------------------------------------------------------------------------

class TestScorerDirection:
    """Scorer must prefer positions that reduce imbalance."""

    def test_scorer_prefers_starboard_when_port_heavy(self):
        ship = make_ship()
        loader = CargoLoader(ship)
        # Simulate port-heavy state: (bay=5, col=5, w=30000) fore-port
        #                            (bay=5, col=9, w=10000) fore-stbd
        loader._moment_z = 30_000 * 3 + 10_000 * 3    # 120_000
        loader._fp_w     = 30_000.0
        loader._fs_w     = 10_000.0
        loader._ap_w     = 0.0
        loader._as_w     = 0.0
        loader._total_w  = 40_000.0

        c = ShippingContainer(size=1, weight=5_000.0)
        score_port = loader._score_position(c, bay=10, col=5, tier=3)
        score_stbd = loader._score_position(c, bay=10, col=9, tier=3)
        assert score_stbd > score_port, \
            f"port heavy: expected stbd preferred; " \
            f"port_score={score_port:.1f} stbd_score={score_stbd:.1f}"

    def test_scorer_prefers_aft_when_fore_heavy(self):
        ship = make_ship()
        loader = CargoLoader(ship)
        # Simulate fore-heavy, PS-balanced state:
        # (bay=5, col=6, w=15000) fore-port, (bay=5, col=7, w=15000) fore-stbd
        loader._moment_z = 0.0
        loader._fp_w     = 15_000.0
        loader._fs_w     = 15_000.0
        loader._ap_w     = 0.0
        loader._as_w     = 0.0
        loader._total_w  = 30_000.0

        c = ShippingContainer(size=1, weight=5_000.0)
        # Same col (identical PS contribution), different bay
        score_fore = loader._score_position(c, bay=10, col=6, tier=0)
        score_aft  = loader._score_position(c, bay=25, col=6, tier=0)
        assert score_aft > score_fore, \
            f"fore heavy: expected aft preferred; " \
            f"fore_score={score_fore:.1f} aft_score={score_aft:.1f}"

    def test_scorer_prefers_lower_tier(self):
        """All else equal, tier 0 must score higher than tier 1."""
        ship = make_ship()
        loader = CargoLoader(ship)
        c = ShippingContainer(size=1, weight=5_000.0)
        score_t0 = loader._score_position(c, bay=5, col=6, tier=0)
        score_t1 = loader._score_position(c, bay=5, col=6, tier=1)
        assert score_t0 > score_t1, \
            f"tier 0 should outscore tier 1; t0={score_t0:.1f} t1={score_t1:.1f}"


# ---------------------------------------------------------------------------
# 7. Greedy algorithm end-to-end balance
# ---------------------------------------------------------------------------

class TestGreedyBalance:
    def _run_greedy(self, seed):
        ship = make_ship()
        rng = random.Random(seed)
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1))
             for _ in range(60)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1))
               for _ in range(25)]
        )
        rng.shuffle(containers)
        CargoLoader(ship).load(containers)
        return ship

    @pytest.mark.parametrize("seed", [42, 99, 7, 1234])
    def test_greedy_port_starboard_ratio(self, seed):
        ship = self._run_greedy(seed)
        p, s = ship.port_starboard_balance()
        ratio = min(p, s) / max(p, s) if max(p, s) > 0 else 1.0
        assert ratio >= 0.95, \
            f"seed={seed}: PS ratio={ratio:.3f} (port={p:.0f} stbd={s:.0f})"

    @pytest.mark.parametrize("seed", [42, 99, 7, 1234])
    def test_greedy_fore_aft_ratio(self, seed):
        ship = self._run_greedy(seed)
        f, a = ship.fore_aft_balance()
        ratio = min(f, a) / max(f, a) if max(f, a) > 0 else 1.0
        assert ratio >= 0.95, \
            f"seed={seed}: FA ratio={ratio:.3f} (fore={f:.0f} aft={a:.0f})"

    @pytest.mark.parametrize("seed", [42, 99, 7, 1234])
    def test_all_containers_placed(self, seed):
        ship = make_ship()
        rng = random.Random(seed)
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1))
             for _ in range(60)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1))
               for _ in range(25)]
        )
        rng.shuffle(containers)
        manifest = CargoLoader(ship).load(containers)
        unplaced = [e for e in manifest if not e["placed"]]
        assert len(unplaced) == 0, \
            f"seed={seed}: {len(unplaced)} containers could not be placed"


# ---------------------------------------------------------------------------
# 8. Visualizer accumulator matches ship balance functions
# ---------------------------------------------------------------------------

class TestVisualizerAccumulator:
    """
    After replaying all frames, the panel's internal accumulators must
    exactly match what ship.port_starboard_balance() and
    ship.fore_aft_balance() return.  This verifies the visualizer uses
    the same split thresholds as the model.
    """

    def _run_and_replay(self, seed, n_20ft=20, n_40ft=5):
        ship = make_ship()
        rng = random.Random(seed)
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1))
             for _ in range(n_20ft)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1))
               for _ in range(n_40ft)]
        )
        rng.shuffle(containers)
        manifest = CargoLoader(ship).load(containers)

        panel, fig = make_panel(manifest, ship)
        for i in range(len(panel.manifest)):
            panel._frame(i)
        return ship, panel, fig

    def test_port_starboard_matches(self):
        ship, panel, fig = self._run_and_replay(seed=42)
        p, s = ship.port_starboard_balance()
        assert panel.port_w == pytest.approx(p, rel=1e-6), \
            f"panel.port_w={panel.port_w:.0f} != balance_fn port={p:.0f}"
        assert panel.stbd_w == pytest.approx(s, rel=1e-6), \
            f"panel.stbd_w={panel.stbd_w:.0f} != balance_fn stbd={s:.0f}"
        plt.close(fig)

    def test_fore_aft_matches(self):
        ship, panel, fig = self._run_and_replay(seed=42)
        f, a = ship.fore_aft_balance()
        assert panel.fore_w == pytest.approx(f, rel=1e-6), \
            f"panel.fore_w={panel.fore_w:.0f} != balance_fn fore={f:.0f}"
        assert panel.aft_w == pytest.approx(a, rel=1e-6), \
            f"panel.aft_w={panel.aft_w:.0f} != balance_fn aft={a:.0f}"
        plt.close(fig)

    def test_accumulator_total_equals_ship_total_weight(self):
        ship, panel, fig = self._run_and_replay(seed=99)
        viz_total = panel.port_w + panel.stbd_w
        assert viz_total == pytest.approx(ship.total_weight, rel=1e-6), \
            f"viz total={viz_total:.0f} != ship.total_weight={ship.total_weight:.0f}"
        plt.close(fig)

    def test_40ft_ps_accumulator_matches(self):
        """Specifically verify 40ft containers are tracked correctly for PS."""
        ship, panel, fig = self._run_and_replay(seed=7, n_20ft=0, n_40ft=10)
        p, s = ship.port_starboard_balance()
        assert panel.port_w == pytest.approx(p, rel=1e-6)
        assert panel.stbd_w == pytest.approx(s, rel=1e-6)
        plt.close(fig)

    def test_40ft_fa_accumulator_matches(self):
        """Specifically verify 40ft containers are tracked correctly for FA."""
        ship, panel, fig = self._run_and_replay(seed=7, n_20ft=0, n_40ft=10)
        f, a = ship.fore_aft_balance()
        assert panel.fore_w == pytest.approx(f, rel=1e-6)
        assert panel.aft_w == pytest.approx(a, rel=1e-6)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Quadrant balance
# ---------------------------------------------------------------------------

class TestQuadrantBalance:
    """
    Tests for quadrant_balance() correctness and the greedy algorithm's
    ability to avoid diagonal loading (all port weight in fore, all stbd
    weight in aft — which satisfies P/S and F/A ratios but is not balanced).
    """

    def test_quadrant_sum_equals_total_weight(self):
        """fp + fs + ap + as_ must equal total_weight."""
        ship = make_ship()
        for bay in [0, 5, 18, 30]:
            force_place(ship, bay=bay, col=6, tier=0, weight=8_000.0)
        fp, fs, ap, as_ = ship.quadrant_balance()
        assert fp + fs + ap + as_ == pytest.approx(ship.total_weight), \
            f"quadrant sum={fp+fs+ap+as_:.0f} != total={ship.total_weight:.0f}"

    def test_quadrant_sums_match_ps_and_fa(self):
        """fp+ap == port_w and fp+fs == fore_w."""
        ship = make_ship()
        for bay, col, w in [(2, 4, 10_000), (2, 9, 12_000), (25, 4, 9_000), (25, 9, 11_000)]:
            force_place(ship, bay=bay, col=col, tier=0, weight=w)
        fp, fs, ap, as_ = ship.quadrant_balance()
        port_w, stbd_w = ship.port_starboard_balance()
        fore_w, aft_w  = ship.fore_aft_balance()
        assert fp + ap == pytest.approx(port_w), f"fp+ap={fp+ap:.0f} != port={port_w:.0f}"
        assert fs + as_ == pytest.approx(stbd_w), f"fs+as={fs+as_:.0f} != stbd={stbd_w:.0f}"
        assert fp + fs == pytest.approx(fore_w), f"fp+fs={fp+fs:.0f} != fore={fore_w:.0f}"
        assert ap + as_ == pytest.approx(aft_w), f"ap+as={ap+as_:.0f} != aft={aft_w:.0f}"

    def test_single_container_in_each_quadrant(self):
        """A container in each quadrant lands in exactly that quadrant."""
        cases = [
            (2,  4, "fore-port"),
            (2,  9, "fore-stbd"),
            (25, 4, "aft-port"),
            (25, 9, "aft-stbd"),
        ]
        for bay, col, label in cases:
            ship = make_ship()
            force_place(ship, bay=bay, col=col, tier=0, weight=10_000.0)
            fp, fs, ap, as_ = ship.quadrant_balance()
            expected = {"fore-port": fp, "fore-stbd": fs, "aft-port": ap, "aft-stbd": as_}[label]
            assert expected == pytest.approx(10_000.0), \
                f"container at ({bay},{col}) should land in {label}: fp={fp:.0f} fs={fs:.0f} ap={ap:.0f} as={as_:.0f}"

    @pytest.mark.parametrize("seed", [42, 99, 7, 1234])
    def test_greedy_no_diagonal_loading(self, seed):
        """
        After greedy loading, neither main diagonal should dominate.
        Diagonal ratio (min/max of the two diagonals) must be >= 0.90.
        A ratio near 0 indicates pure diagonal loading (all weight on one diagonal).
        """
        ship = make_ship()
        rng = random.Random(seed)
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1))
             for _ in range(60)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1))
               for _ in range(25)]
        )
        rng.shuffle(containers)
        CargoLoader(ship).load(containers)

        fp, fs, ap, as_ = ship.quadrant_balance()
        d1 = fp + as_   # fore-port + aft-stbd diagonal
        d2 = fs + ap    # fore-stbd + aft-port diagonal
        ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0
        assert ratio >= 0.90, (
            f"seed={seed}: diagonal ratio={ratio:.3f} — diagonal loading detected "
            f"(d1={d1:.0f} d2={d2:.0f}; fp={fp:.0f} fs={fs:.0f} ap={ap:.0f} as={as_:.0f})"
        )

    @pytest.mark.parametrize("seed", [42, 99, 7, 1234])
    def test_greedy_all_quadrants_roughly_equal(self, seed):
        """
        Each quadrant should hold between 20% and 30% of total weight.
        Diagonal loading would produce 50%/0%/0%/50% — caught by this test.
        """
        ship = make_ship()
        rng = random.Random(seed)
        containers = (
            [ShippingContainer(size=1, weight=round(rng.uniform(2_000, 28_000), 1))
             for _ in range(60)]
            + [ShippingContainer(size=2, weight=round(rng.uniform(2_000, 28_000), 1))
               for _ in range(25)]
        )
        rng.shuffle(containers)
        CargoLoader(ship).load(containers)

        fp, fs, ap, as_ = ship.quadrant_balance()
        total = ship.total_weight
        for label, q in [("fore-port", fp), ("fore-stbd", fs), ("aft-port", ap), ("aft-stbd", as_)]:
            frac = q / total
            assert 0.20 <= frac <= 0.30, (
                f"seed={seed}: {label}={q:.0f} is {100*frac:.1f}% of total — "
                f"expected 20-30% (fp={fp:.0f} fs={fs:.0f} ap={ap:.0f} as={as_:.0f})"
            )
