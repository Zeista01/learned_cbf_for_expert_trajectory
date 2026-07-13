"""
generate_obstacle_stl.py — build 3D CAD (STL) models of the critical obstacles
DIRECTLY from the exact SDF definitions in config.py.

Because the task is planar (fixed Z), every obstacle is a constant cross-section:
we extract its true 2D boundary (the sdf = 0 contour, including concavities and
any interior holes) and EXTRUDE it vertically to a uniform height -> a prism whose
top face and every slice equal the 2D profile used in the project.

No Fusion / no GUI needed. Pure Python (skimage + shapely + trimesh).

Output: assets/obstacle_stl/<label>_<type>.stl  (+ a top-view preview PNG)

Usage:
    python src/generate_obstacle_stl.py                 # default: mm units, 20 mm tall
    python src/generate_obstacle_stl.py --height 15 --units mm
    python src/generate_obstacle_stl.py --units m       # keep real meters
    python src/generate_obstacle_stl.py --scene         # also export one combined
                                                        #   scene STL at world XY
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import CRITICAL_SHAPES, sdf_critical_shape_2d

from skimage import measure
import shapely.geometry as sg
from shapely.geometry import Polygon, MultiPolygon
import trimesh

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(ROOT, "assets", "obstacle_stl")
os.makedirs(OUT, exist_ok=True)

GRID  = 600          # marching-squares resolution per axis
HALF  = 0.055        # half-extent (m) of the sampling box around each centre
SIMPLIFY = 2e-5      # boundary simplification tolerance (m); keeps curvature
MIN_FRAC = 0.05      # drop disconnected solids smaller than this fraction of the
                     # largest solid (removes spurious SDF slivers)


def shape_boundary_polygon(shape):
    """Return a shapely (Multi)Polygon of the obstacle's true 2D cross-section,
    reconstructed from the exact analytic SDF (concavities + holes preserved)."""
    c2 = np.asarray(shape['center'][:2], dtype=np.float64)
    xs = np.linspace(c2[0] - HALF, c2[0] + HALF, GRID)
    ys = np.linspace(c2[1] - HALF, c2[1] + HALF, GRID)
    XX, YY = np.meshgrid(xs, ys)                       # [row=y, col=x]
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    sdf = sdf_critical_shape_2d(pts, shape).reshape(GRID, GRID)

    # marching squares on the sdf=0 level set -> boundary contours (row, col)
    contours = measure.find_contours(sdf, 0.0)
    if not contours:
        raise RuntimeError(f"no boundary found for {shape['type']} "
                           f"(shape may be empty at this resolution)")

    def to_world(c):
        col, row = c[:, 1], c[:, 0]
        x = c2[0] - HALF + col / (GRID - 1) * (2 * HALF)
        y = c2[1] - HALF + row / (GRID - 1) * (2 * HALF)
        return np.stack([x, y], axis=1)

    rings = []
    for c in contours:
        w = to_world(c)
        if len(w) < 4:
            continue
        p = Polygon(w)
        if not p.is_valid:
            p = p.buffer(0)
        if p.area > 1e-8:
            rings.append(p)
    if not rings:
        raise RuntimeError(f"no valid ring for {shape['type']}")

    # classify rings by containment depth: even depth = solid, odd = hole
    def depth(i):
        return sum(1 for j, q in enumerate(rings)
                   if j != i and q.contains(rings[i].representative_point()))
    solids, holes = [], []
    for i, r in enumerate(rings):
        (holes if depth(i) % 2 else solids).append(r)

    # drop negligible disconnected fragments (e.g. the kidney SDF's micro sliver)
    # so each part is a single clean solid; keep anything >= MIN_FRAC of the largest.
    if solids:
        a_max = max(s.area for s in solids)
        solids = [s for s in solids if s.area >= MIN_FRAC * a_max]

    polys = []
    for s in solids:
        s_holes = [h.exterior.coords for h in holes if s.contains(h.representative_point())]
        polys.append(Polygon(s.exterior.coords, s_holes))
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    geom = geom.simplify(SIMPLIFY, preserve_topology=True)
    return geom


def extrude(geom, height):
    """Extrude a shapely (Multi)Polygon to a watertight prism of given height."""
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    meshes = [trimesh.creation.extrude_polygon(p, height=height) for p in parts]
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=20.0,
                    help="extrusion height in OUTPUT units (default 20)")
    ap.add_argument("--units", choices=["m", "mm", "cm"], default="mm",
                    help="output units (default mm; STL is unitless, mm is print-friendly)")
    ap.add_argument("--center", action="store_true", default=True,
                    help="center each part at XY origin (default: on)")
    ap.add_argument("--world", dest="center", action="store_false",
                    help="keep real world XY position instead of centering")
    ap.add_argument("--scene", action="store_true",
                    help="also export one combined scene STL at world XY")
    args = ap.parse_args()

    U = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[args.units]   # meters -> output
    height_m = args.height / U                              # back to meters for extrude
    print(f"[STL] units={args.units}  height={args.height}{args.units}  "
          f"-> {OUT}")

    scene_parts = []
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(CRITICAL_SHAPES), figsize=(3.2*len(CRITICAL_SHAPES), 3.4))

    for ax, shape in zip(axes, CRITICAL_SHAPES):
        geom = shape_boundary_polygon(shape)
        mesh = extrude(geom, height_m)              # meters

        # optionally recentre the part at the XY origin (clean standalone model)
        if args.center:
            mesh.apply_translation([-mesh.centroid[0], -mesh.centroid[1], 0])

        mesh.apply_scale(U)                          # meters -> output units

        name = f"{shape['label']}_{shape['type']}.stl"
        path = os.path.join(OUT, name)
        mesh.export(path)
        bb = mesh.bounds
        print(f"  {shape['type']:9s} [{shape['label']:9s}] -> {name:22s} "
              f"watertight={mesh.is_watertight}  "
              f"XY={ (bb[1,0]-bb[0,0]):.1f}x{(bb[1,1]-bb[0,1]):.1f}{args.units}  "
              f"tris={len(mesh.faces)}")

        # collect a WORLD-placed copy for the optional combined scene
        if args.scene:
            g = shape_boundary_polygon(shape)
            m = extrude(g, height_m); m.apply_scale(U)
            scene_parts.append(m)

        # top-view preview (matches the project's 2D profile)
        for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]):
            xe, ye = poly.exterior.xy
            ax.fill(xe, ye, facecolor="#e06666", edgecolor="k", lw=1.2, alpha=0.85)
            for ring in poly.interiors:
                xi, yi = ring.xy
                ax.fill(xi, yi, facecolor="white", edgecolor="k", lw=1.0)
        ax.set_aspect("equal"); ax.set_title(f"{shape['type']}\n({shape['label']})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Obstacle top-view cross-sections (extruded profile)", fontsize=12)
    fig.tight_layout()
    prev = os.path.join(OUT, "_preview_top_view.png")
    fig.savefig(prev, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  preview -> {prev}")

    if args.scene and scene_parts:
        scene = trimesh.util.concatenate(scene_parts)
        sp = os.path.join(OUT, "all_obstacles_scene.stl")
        scene.export(sp)
        print(f"  scene   -> {sp}  (world XY, {len(scene.faces)} tris)")

    print("[STL] done.")


if __name__ == "__main__":
    main()
