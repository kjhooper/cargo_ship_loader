import random

from algorithm import CargoLoader
from models import CargoShip, ShippingContainer
from visualizer import ComparisonVisualizer, visualize_hull_3d


def generate_containers(
    n_20ft: int,
    n_40ft: int,
    weight_min: float,
    weight_max: float,
    seed: int = 42,
) -> list:
    rng = random.Random(seed)
    ShippingContainer.reset_id_counter()
    containers = []
    for _ in range(n_20ft):
        w = round(rng.uniform(weight_min, weight_max), 1)
        containers.append(ShippingContainer(size=1, weight=w))
    for _ in range(n_40ft):
        w = round(rng.uniform(weight_min, weight_max), 1)
        containers.append(ShippingContainer(size=2, weight=w))
    rng.shuffle(containers)
    return containers


def print_manifest(manifest: list, ship: CargoShip, label: str = "") -> None:
    placed = [e for e in manifest if e["placed"]]
    unplaced = [e for e in manifest if not e["placed"]]

    col_w = 70
    heading = f"CARGO MANIFEST{f'  [{label}]' if label else ''}"
    print(f"\n{'=' * col_w}")
    print(f"{heading:^{col_w}}")
    print(f"{'=' * col_w}")
    header = f"{'ID':>4}  {'Size':>4}  {'Weight':>8}  {'Bay':>4}  {'Col':>4}  {'Tier':>4}  {'Slot':>4}"
    print(header)
    print("-" * col_w)
    for e in placed:
        print(
            f"{e['container_id']:>4}  {e['size']:>4}  {e['weight']:>8.1f}  "
            f"{e['bay']:>4}  {e['col']:>4}  {e['tier']:>4}  {e['slot']:>4}"
        )

    if unplaced:
        print(f"\n  Unplaced containers ({len(unplaced)}):")
        for e in unplaced:
            ft = "20ft" if e["size"] == 1 else "40ft"
            print(f"    ID={e['container_id']}  {ft}  weight={e['weight']:.1f}")

    port_w, stbd_w = ship.port_starboard_balance()
    fore_w, aft_w = ship.fore_aft_balance()
    total = ship.total_weight or 1.0

    ps_ratio = (
        min(port_w, stbd_w) / max(port_w, stbd_w)
        if max(port_w, stbd_w) > 0
        else 1.0
    )
    fa_ratio = (
        min(fore_w, aft_w) / max(fore_w, aft_w)
        if max(fore_w, aft_w) > 0
        else 1.0
    )

    print(f"\n{'=' * col_w}")
    print(f"  Containers placed   : {len(placed)} / {len(manifest)}")
    print(f"  Total weight loaded : {ship.total_weight:.1f} kg")
    print(f"  Port weight         : {port_w:.1f}  ({100 * port_w / total:.1f}%)")
    print(f"  Starboard weight    : {stbd_w:.1f}  ({100 * stbd_w / total:.1f}%)")
    print(f"  Fore weight         : {fore_w:.1f}  ({100 * fore_w / total:.1f}%)")
    print(f"  Aft weight          : {aft_w:.1f}  ({100 * aft_w / total:.1f}%)")
    print(f"  Port/Stbd ratio     : {ps_ratio:.3f}  (1.000 = perfect)")
    print(f"  Fore/Aft ratio      : {fa_ratio:.3f}  (1.000 = perfect)")
    print(f"{'=' * col_w}\n")


def make_panamax_ship() -> CargoShip:
    """Panamax-class ship with realistic hull geometry.

    length=36  : 18 real 40ft bays × 2 = 36 20ft positions
    base_width=7: ~17m beam at keel ÷ 2.44m per TEU ≈ 7 columns
    max_width=13: Panamax beam ~32m ÷ 2.44m ≈ 13 columns
    height=9    : 4 tiers below deck + 5 above
    width_step=1: hull widens every tier (smooth expansion)
    """
    return CargoShip(
        length=36,
        base_width=7,
        max_width=13,
        height=9,
        width_step=1,
        max_weight=50000.0,
    )


if __name__ == "__main__":
    # --- 3D hull geometry ---
    ship_hull = make_panamax_ship()
    visualize_hull_3d(
        ship_hull.cargo_hold,
        ship_hull.length,
        ship_hull.width,
        ship_hull.height,
        title="Panamax Hull — Empty Cargo Hold (3D)",
    )

    # --- Run A: Greedy, seed=42 ---
    ship_a = make_panamax_ship()
    containers_a = generate_containers(
        n_20ft=60,
        n_40ft=25,
        weight_min=2000.0,
        weight_max=28000.0,
        seed=42,
    )
    manifest_a = CargoLoader(ship_a).load(containers_a)
    print_manifest(manifest_a, ship_a, label="Greedy — seed 42")

    # --- Run B: Greedy, seed=99 ---
    ship_b = make_panamax_ship()
    containers_b = generate_containers(
        n_20ft=60,
        n_40ft=25,
        weight_min=2000.0,
        weight_max=28000.0,
        seed=99,
    )
    manifest_b = CargoLoader(ship_b).load(containers_b)
    print_manifest(manifest_b, ship_b, label="Greedy — seed 99")

    # --- Side-by-side comparison ---
    viz = ComparisonVisualizer(
        left=(manifest_a, "Greedy — seed 42"),
        right=(manifest_b, "Greedy — seed 99"),
        ship_length=ship_a.length,
        ship_width=ship_a.width,
        ship_height=ship_a.height,
        hull=ship_a.cargo_hold,
    )
    viz.animate(interval_ms=400)
