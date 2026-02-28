"""
Visualizer — matplotlib-only module (no imports from models or algorithm).

Produces a FuncAnimation showing containers placed one-by-one in a bird's-eye
top-down view, colour-coded from blue (light) to red (heavy) via coolwarm.
A separate panel shows live Port/Starboard and Fore/Aft balance bars.

Classes
-------
_ShipPanel          Internal helper: owns one ship-grid axes + one balance axes.
Visualizer          Single-ship animation (unchanged public API).
ComparisonVisualizer Side-by-side animation of two solver results.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


class _ShipPanel:
    """Owns one ship-grid axes and one balance axes for a single solver result."""

    def __init__(
        self,
        manifest: List[Dict],
        ship_length: int,
        ship_width: int,
        ship_height: int,
        fig,
        ax_ship,
        ax_bal,
        norm: mcolors.Normalize,
        cmap,
        label: str = "",
        hull: Optional[np.ndarray] = None,
    ):
        self.manifest = [e for e in manifest if e["placed"]]
        self.ship_length = ship_length
        self.ship_width = ship_width
        self.ship_height = ship_height
        self.fig = fig
        self.ax_ship = ax_ship
        self.ax_bal = ax_bal
        self.norm = norm
        self.cmap = cmap
        self.label = label
        self.hull = hull  # cargo_hold array; -1.0 = hull cell

        self._cell_patch: Dict[Tuple[int, int], mpatches.FancyBboxPatch] = {}
        self._cell_text: Dict[Tuple[int, int], plt.Text] = {}

        self.port_w = 0.0
        self.stbd_w = 0.0
        self.fore_w = 0.0
        self.aft_w = 0.0

        self._setup_ship_axes()
        self._setup_balance_axes()
        self._init_balance_bars()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ship_axes(self) -> None:
        ax = self.ax_ship
        ax.set_xlim(0, self.ship_length)
        ax.set_ylim(0, self.ship_width)
        ax.set_xlabel("Bay")
        ax.set_ylabel("Column (port → starboard)")
        title = "Cargo Ship — Bird's Eye View (topmost tier shown)"
        if self.label:
            title = f"{self.label}\n{title}"
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks(range(self.ship_length + 1))
        ax.set_yticks(range(self.ship_width + 1))

        # Hull overlay: for each column, shade cells that are hull at some tiers.
        # Opacity is proportional to the fraction of tiers that are hull so
        # the gradient visually communicates the hull expansion from keel up.
        if self.hull is not None:
            for col in range(self.ship_width):
                hull_frac = float((self.hull[0, col, :] == -1.0).sum()) / self.ship_height
                if hull_frac > 0:
                    ax.add_patch(mpatches.Rectangle(
                        (0, col), self.ship_length, 1,
                        facecolor="dimgray",
                        alpha=min(hull_frac * 2.0, 0.72),
                        edgecolor="none",
                        zorder=1,
                    ))

        sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        self.fig.colorbar(
            sm, ax=ax, label="Container Weight (kg)",
            orientation="horizontal", location="bottom",
            shrink=0.8, pad=0.12,
        )

    def _setup_balance_axes(self) -> None:
        ax = self.ax_bal
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title("Weight Balance", pad=8)

    def _init_balance_bars(self) -> None:
        ax = self.ax_bal
        bar_h = 0.55
        x0 = 0.05

        # --- Port / Starboard row ---
        ax.text(0.5, 3.55, "Port / Starboard", ha="center", va="center",
                fontsize=9, fontweight="bold")

        bg_ps = mpatches.Rectangle((x0, 2.85), 0.9, bar_h, color="lightgray", zorder=1)
        ax.add_patch(bg_ps)

        self._ps_port = mpatches.Rectangle((x0, 2.85), 0.0, bar_h,
                                           color="steelblue", zorder=2)
        self._ps_stbd = mpatches.Rectangle((x0, 2.85), 0.0, bar_h,
                                           color="coral", zorder=2)
        ax.add_patch(self._ps_port)
        ax.add_patch(self._ps_stbd)

        self._ps_label = ax.text(0.5, 2.75, "Port: 0  |  Stbd: 0",
                                  ha="center", va="top", fontsize=8)

        # --- Fore / Aft row ---
        ax.text(0.5, 2.1, "Fore / Aft", ha="center", va="center",
                fontsize=9, fontweight="bold")

        bg_fa = mpatches.Rectangle((x0, 1.15), 0.9, bar_h, color="lightgray", zorder=1)
        ax.add_patch(bg_fa)

        self._fa_fore = mpatches.Rectangle((x0, 1.15), 0.0, bar_h,
                                           color="steelblue", zorder=2)
        self._fa_aft = mpatches.Rectangle((x0, 1.15), 0.0, bar_h,
                                          color="coral", zorder=2)
        ax.add_patch(self._fa_fore)
        ax.add_patch(self._fa_aft)

        self._fa_label = ax.text(0.5, 1.05, "Fore: 0  |  Aft: 0",
                                  ha="center", va="top", fontsize=8)

    # ------------------------------------------------------------------
    # Balance bar update
    # ------------------------------------------------------------------

    def _update_balance_bars(self) -> None:
        max_bar = 0.9
        x0 = 0.05

        total_ps = (self.port_w + self.stbd_w) or 1.0
        total_fa = (self.fore_w + self.aft_w) or 1.0

        pf = self.port_w / total_ps
        sf = self.stbd_w / total_ps
        ff = self.fore_w / total_fa
        af = self.aft_w / total_fa

        ps_ok = abs(pf - sf) <= 0.10
        fa_ok = abs(ff - af) <= 0.10

        port_color = "steelblue" if ps_ok else "red"
        stbd_color = "coral" if ps_ok else "orange"
        fore_color = "steelblue" if fa_ok else "red"
        aft_color = "coral" if fa_ok else "orange"

        self._ps_port.set_x(x0)
        self._ps_port.set_width(pf * max_bar)
        self._ps_port.set_color(port_color)
        self._ps_stbd.set_x(x0 + pf * max_bar)
        self._ps_stbd.set_width(sf * max_bar)
        self._ps_stbd.set_color(stbd_color)
        self._ps_label.set_text(
            f"Port: {self.port_w:.0f}  |  Stbd: {self.stbd_w:.0f}"
        )

        self._fa_fore.set_x(x0)
        self._fa_fore.set_width(ff * max_bar)
        self._fa_fore.set_color(fore_color)
        self._fa_aft.set_x(x0 + ff * max_bar)
        self._fa_aft.set_width(af * max_bar)
        self._fa_aft.set_color(aft_color)
        self._fa_label.set_text(
            f"Fore: {self.fore_w:.0f}  |  Aft: {self.aft_w:.0f}"
        )

    # ------------------------------------------------------------------
    # Animation frame
    # ------------------------------------------------------------------

    def _frame(self, idx: int) -> list:
        """Draw frame idx.  Returns [] if idx is beyond this panel's manifest."""
        if idx >= len(self.manifest):
            return []

        entry = self.manifest[idx]
        bay: int   = entry["bay"]
        half       = entry.get("half")
        pos: int   = bay * 2 + (half if half is not None else 0)
        col: int   = entry["col"]
        tier: int  = entry["tier"]
        size: int  = entry["size"]
        weight: float = entry["weight"]

        # Remove existing patches for every cell this container covers.
        removed_ids: set = set()
        for b in range(pos, pos + size):
            key = (b, col)
            p = self._cell_patch.get(key)
            t = self._cell_text.get(key)
            if p is not None and id(p) not in removed_ids:
                try:
                    p.remove()
                except (ValueError, AttributeError):
                    pass
                removed_ids.add(id(p))
            if t is not None and id(t) not in removed_ids:
                try:
                    t.remove()
                except (ValueError, AttributeError):
                    pass
                removed_ids.add(id(t))
            self._cell_patch.pop(key, None)
            self._cell_text.pop(key, None)

        # Draw the container rectangle
        color = self.cmap(self.norm(weight))
        rect = mpatches.FancyBboxPatch(
            (pos + 0.05, col + 0.05),
            size - 0.10,
            0.90,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        self.ax_ship.add_patch(rect)

        # Text label — white on dark colours, black on light
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "white" if brightness < 0.55 else "black"
        txt = self.ax_ship.text(
            pos + size / 2,
            col + 0.5,
            f"T{tier}",
            ha="center",
            va="center",
            fontsize=7,
            color=txt_color,
            zorder=4,
        )

        # Store under each covered cell
        for b in range(pos, pos + size):
            self._cell_patch[(b, col)] = rect
            self._cell_text[(b, col)] = txt

        # Update balance accumulators
        center_col = self.ship_width / 2.0
        center_bay = self.ship_length / 2.0
        if col < center_col:
            self.port_w += weight
        else:
            self.stbd_w += weight
        if pos < center_bay:
            self.fore_w += weight
        else:
            self.aft_w += weight

        self._update_balance_bars()

        return (
            list(self._cell_patch.values())
            + list(self._cell_text.values())
            + [
                self._ps_port,
                self._ps_stbd,
                self._fa_fore,
                self._fa_aft,
                self._ps_label,
                self._fa_label,
            ]
        )


class Visualizer:
    """Single-ship animation.  Public API is unchanged from the original."""

    def __init__(
        self,
        manifest: List[Dict],
        ship_length: int,
        ship_width: int,
        ship_height: int,
        hull: Optional[np.ndarray] = None,
    ):
        weights = [e["weight"] for e in manifest if e["placed"]]
        max_weight = max(weights) if weights else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_weight)
        cmap = cm.coolwarm

        self.fig = plt.figure(figsize=(14, 7))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[7, 3], wspace=0.35)
        ax_ship = self.fig.add_subplot(gs[0])
        ax_bal = self.fig.add_subplot(gs[1])

        self._panel = _ShipPanel(
            manifest=manifest,
            ship_length=ship_length,
            ship_width=ship_width,
            ship_height=ship_height,
            fig=self.fig,
            ax_ship=ax_ship,
            ax_bal=ax_bal,
            norm=norm,
            cmap=cmap,
            hull=hull,
        )

    def animate(
        self, interval_ms: int = 400, save_path: Optional[str] = None
    ) -> None:
        """Run the animation. Pass save_path to write a GIF instead of displaying."""
        if not self._panel.manifest:
            print("No placed containers to animate.")
            return

        anim = FuncAnimation(
            self.fig,
            self._panel._frame,
            frames=len(self._panel.manifest),
            interval=interval_ms,
            blit=False,
            repeat=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow")
            print(f"Animation saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()


class ComparisonVisualizer:
    """Side-by-side animation of two solver results on the same ship dimensions.

    Both panels share the same colour scale so weights are directly comparable.
    Frames are synchronized: frame i places container i on each side.  If one
    manifest is shorter it simply stops updating after its last container.

    Parameters
    ----------
    left:   (manifest, label) for the left panel.
    right:  (manifest, label) for the right panel.
    ship_length, ship_width, ship_height: ship dimensions (must match both runs).
    """

    def __init__(
        self,
        left: Tuple[List[Dict], str],
        right: Tuple[List[Dict], str],
        ship_length: int,
        ship_width: int,
        ship_height: int,
        hull: Optional[np.ndarray] = None,
    ):
        left_manifest, left_label = left
        right_manifest, right_label = right

        # Shared colour scale across both panels
        all_weights = (
            [e["weight"] for e in left_manifest if e["placed"]]
            + [e["weight"] for e in right_manifest if e["placed"]]
        )
        max_weight = max(all_weights) if all_weights else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_weight)
        cmap = cm.coolwarm

        self.fig = plt.figure(figsize=(24, 8))
        gs = self.fig.add_gridspec(1, 4, width_ratios=[7, 3, 7, 3], wspace=0.35)
        ax_ship_l = self.fig.add_subplot(gs[0])
        ax_bal_l = self.fig.add_subplot(gs[1])
        ax_ship_r = self.fig.add_subplot(gs[2])
        ax_bal_r = self.fig.add_subplot(gs[3])

        self._left = _ShipPanel(
            manifest=left_manifest,
            ship_length=ship_length,
            ship_width=ship_width,
            ship_height=ship_height,
            fig=self.fig,
            ax_ship=ax_ship_l,
            ax_bal=ax_bal_l,
            norm=norm,
            cmap=cmap,
            label=left_label,
            hull=hull,
        )
        self._right = _ShipPanel(
            manifest=right_manifest,
            ship_length=ship_length,
            ship_width=ship_width,
            ship_height=ship_height,
            fig=self.fig,
            ax_ship=ax_ship_r,
            ax_bal=ax_bal_r,
            norm=norm,
            cmap=cmap,
            label=right_label,
            hull=hull,
        )

        self._total_frames = max(
            len(self._left.manifest), len(self._right.manifest)
        )

    def _frame(self, idx: int) -> list:
        artists = []
        artists.extend(self._left._frame(idx))
        artists.extend(self._right._frame(idx))
        return artists

    def animate(
        self, interval_ms: int = 400, save_path: Optional[str] = None
    ) -> None:
        """Run the side-by-side animation."""
        if self._total_frames == 0:
            print("No placed containers to animate.")
            return

        anim = FuncAnimation(
            self.fig,
            self._frame,
            frames=self._total_frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow")
            print(f"Animation saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()


def visualize_hull_3d(
    cargo_hold: np.ndarray,
    ship_length: int,
    ship_width: int,
    ship_height: int,
    title: str = "Hull — Empty Cargo Hold (3D)",
) -> None:
    """Render the empty hull geometry as a 3-D voxel plot.

    Valid cells are coloured by tier (dark blue = keel, light yellow = top)
    so the outward expansion of the hull is immediately visible.
    Hull cells (-1.0) are transparent.

    Parameters
    ----------
    cargo_hold:  The ship's cargo_hold array (length × width × height).
                 -1.0 = hull cell, 0.0 = valid empty cell.
    """
    filled = cargo_hold == 0.0  # shape (length, width, height)

    # Per-voxel RGBA, coloured by tier
    tier_cmap = cm._colormaps["YlGnBu"]
    facecolors = np.zeros((*filled.shape, 4), dtype=float)
    for tier in range(ship_height):
        rgba = tier_cmap(tier / max(ship_height - 1, 1))
        facecolors[:, :, tier] = rgba
    facecolors[~filled, 3] = 0.0  # fully transparent for hull cells

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.1)

    ax.set_xlabel("Bay (fore → aft)")
    ax.set_ylabel("Column (port → starboard)")
    ax.set_zlabel("Tier (keel → deck)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=225)

    sm = cm.ScalarMappable(
        cmap=tier_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=ship_height - 1),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Tier", shrink=0.55, pad=0.1)

    plt.tight_layout()
    plt.show()
