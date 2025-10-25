#!/usr/bin/env python3
"""
precompute_routing_artifacts.py

Precompute simplified flood layer and routing graph for Streamlit deployment.

Outputs (written to `precomputed/` by default):
- flood_simplified.geojson      (WGS84)   -- simplified/dissolved flood polygons for fast map rendering
- node_positions.npy            (projected EPSG:32651) -- Nx2 array of node coordinates (meters)
- node_positions_wgs84.npy      (WGS84 lat/lon) -- Nx2 array (lat, lon) for mapping convenience
- var_per_node.npy              (int8)    -- risk value per node (aligned with node_positions)
- graph.pkl                     (pickled networkx Graph) -- routing network with edge attrs 'weight' & 'distance'

Usage:
    python precompute_routing_artifacts.py --shapefile path/to/ph137404000_fh5yr_30m_10m.shp

Requirements (run locally):
    geopandas, numpy, scipy, shapely, networkx
"""
import argparse
from pathlib import Path
import pickle
import math

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import networkx as nx

# -----------------------
# Parameters (tweakable)
# -----------------------
DEFAULT_OUT_DIR = Path("precomputed")
# Projected working CRS (meters) - keep same as your router code
WORK_EPSG = 32651

# Grid and simplification defaults
DEFAULT_GRID_RES = 50         # meters between grid nodes
SIMPLIFY_TOL_M = 200         # meters; larger -> fewer vertices
BATCH_SIZE = 10000           # for batched spatial join of points -> polygons

# Risk cost mapping used for weights (same as router)
RISK_COSTS = np.array([1.0, 1.0, 1.5, 1.0, 2.0, 5.0, 20.0])  # index by Var + 2


# -----------------------
# Helpers
# -----------------------
def ensure_projected(gdf, epsg=WORK_EPSG):
    """Ensure gdf has a CRS; if in lat/lon assume EPSG:4326, then project to epsg."""
    if gdf.crs is None:
        # try to infer from bounds: if within lon/lat range assume 4326
        minx, miny, maxx, maxy = gdf.total_bounds
        if -180 <= minx <= 180 and -90 <= miny <= 90:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        else:
            # fallback to provided working EPSG
            gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    if gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf


def create_smart_grid(flood_gdf_proj, grid_resolution=DEFAULT_GRID_RES):
    """
    Create a rectangular grid (projected CRS) and restrict to flood bounds (with small buffer).
    Returns Nx2 numpy array of coordinates (x, y).
    """
    minx, miny, maxx, maxy = flood_gdf_proj.total_bounds
    # Add tiny epsilon to include last edge
    x_coords = np.arange(minx, maxx + 1e-6, grid_resolution)
    y_coords = np.arange(miny, maxy + 1e-6, grid_resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    # Use a buffered bounding box around the flood unary_union to keep relevant nodes
    total_bounds = flood_gdf_proj.unary_union.bounds
    buffer_dist = grid_resolution * 2.0
    bounds_poly = box(
        total_bounds[0] - buffer_dist,
        total_bounds[1] - buffer_dist,
        total_bounds[2] + buffer_dist,
        total_bounds[3] + buffer_dist,
    )

    mask = (
        (pts[:, 0] >= bounds_poly.bounds[0])
        & (pts[:, 0] <= bounds_poly.bounds[2])
        & (pts[:, 1] >= bounds_poly.bounds[1])
        & (pts[:, 1] <= bounds_poly.bounds[3])
    )
    return pts[mask]


def batch_assign_var_to_points(grid_points, flood_gdf_proj, batch_size=BATCH_SIZE):
    """
    Assign Var value to each point by batched spatial join.
    Returns an integer numpy array length = len(grid_points).
    Logic: create points GeoDataFrame with explicit index matching grid_points order,
    then for batches perform spatial join and aggregate (max Var if multiple polygons match).
    """
    n = len(grid_points)
    risk_levels = np.zeros(n, dtype=np.int8)

    # Build GeoDataFrame of points in projected CRS
    pts_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in grid_points]},
        index=np.arange(n),
        crs=f"epsg:{WORK_EPSG}",
    )

    # Use batched sjoin to manage memory and avoid broadcasting problems
    for i in range(0, n, batch_size):
        batch_idx_end = min(i + batch_size, n)
        batch = pts_gdf.iloc[i:batch_idx_end]

        # spatial join with flood polygons (left so all points preserved)
        joined = gpd.sjoin(batch, flood_gdf_proj[["geometry", "Var"]], how="left", predicate="within")

        if not joined.empty:
            # joined.index corresponds to original point index (we set it explicitly)
            # There may be multiple rows per point if many polygons overlap; choose max Var (worst-case)
            var_per_point = joined["Var"].groupby(joined.index).max()
        else:
            var_per_point = gpd.GeoSeries(dtype=float)

        # Align with batch index and fill unmatched with 0
        aligned = var_per_point.reindex(batch.index).fillna(0).astype(np.int8).values
        risk_levels[i:batch_idx_end] = aligned

    return risk_levels


def build_graph_from_grid(grid_points, var_levels, grid_resolution=DEFAULT_GRID_RES):
    """
    Build an undirected networkx Graph connecting nodes within max_distance.
    Node ids are integer indices into grid_points array.
    Edge attributes:
      - distance: Euclidean distance in meters
      - weight: distance * average_risk_cost
    """
    G = nx.Graph()
    for idx, (pt, risk) in enumerate(zip(grid_points, var_levels)):
        G.add_node(int(idx), pos=(float(pt[0]), float(pt[1])), risk=int(risk))

    tree = cKDTree(grid_points)
    max_distance = grid_resolution * 1.5
    neighbor_lists = tree.query_ball_tree(tree, max_distance)

    edge_count = 0
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if j > i:
                dist = float(np.linalg.norm(grid_points[i] - grid_points[j]))
                if dist < max_distance:
                    ri = int(var_levels[i]); rj = int(var_levels[j])
                    idx_i = np.clip(ri + 2, 0, len(RISK_COSTS) - 1)
                    idx_j = np.clip(rj + 2, 0, len(RISK_COSTS) - 1)
                    avg_cost = float((RISK_COSTS[idx_i] + RISK_COSTS[idx_j]) / 2.0)
                    weight = float(dist * avg_cost)
                    G.add_edge(int(i), int(j), weight=weight, distance=dist)
                    edge_count += 1

    return G, edge_count


# -----------------------
# Main precompute logic
# -----------------------
def precompute(
    shapefile_path,
    out_dir=DEFAULT_OUT_DIR,
    grid_resolution=DEFAULT_GRID_RES,
    simplify_tol_m=SIMPLIFY_TOL_M,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading flood shapefile:", shapefile_path)
    flood_gdf = gpd.read_file(shapefile_path)

    # Ensure 'Var' exists
    if "Var" not in flood_gdf.columns:
        print("Warning: 'Var' column not found in shapefile. Filling with zeros.")
        flood_gdf["Var"] = 0

    # Project to working CRS (meters)
    flood_proj = ensure_projected(flood_gdf, epsg=WORK_EPSG)

    # Simplify + dissolve by Var (reduces number of features drastically)
    print("Dissolving & simplifying flood polygons by Var (this may take a moment)...")
    simplified_rows = []
    for var_val in sorted(flood_proj["Var"].unique()):
        subset = flood_proj[flood_proj["Var"] == var_val]
        if subset.empty:
            continue
        union_geom = subset.geometry.union_all()
        try:
            simp = union_geom.simplify(simplify_tol_m)
        except Exception:
            simp = union_geom
        simplified_rows.append({"Var": int(var_val), "geometry": simp})

    simp_gdf = gpd.GeoDataFrame(simplified_rows, crs=flood_proj.crs)

    # Save simplified flood to GeoJSON (in WGS84 / EPSG:4326) for fast client-side rendering
    out_geojson = out_dir / "flood_simplified.geojson"
    print("Saving simplified flood GeoJSON:", out_geojson)
    try:
        simp_gdf.to_crs(epsg=4326).to_file(out_geojson, driver="GeoJSON")
    except Exception as e:
        print("Warning: failed to write GeoJSON via to_file(); trying alternative write.")
        with open(out_geojson, "w", encoding="utf-8") as f:
            f.write(simp_gdf.to_crs(epsg=4326).to_json())

    # Create smart grid and filter to flood bbox
    print("Creating smart grid with resolution (m):", grid_resolution)
    grid_points = create_smart_grid(flood_proj, grid_resolution=grid_resolution)
    print(f"  grid points created: {len(grid_points):,}")

    # Assign Var (risk) to grid points in batches
    print("Assigning risk levels to grid points via batched spatial join...")
    var_per_node = batch_assign_var_to_points(grid_points, flood_proj, batch_size=BATCH_SIZE)
    print("  assigned risk levels to nodes")

    # Build routing graph
    print("Building routing graph (NetworkX)...")
    G, edge_count = build_graph_from_grid(grid_points, var_per_node, grid_resolution=grid_resolution)
    print(f"  Nodes: {G.number_of_nodes():,} Edges: {G.number_of_edges():,} (computed {edge_count} edges)")

    # Save artifacts
    node_positions_path = out_dir / "node_positions.npy"
    np.save(node_positions_path, grid_points)
    var_path = out_dir / "var_per_node.npy"
    np.save(var_path, var_per_node.astype(np.int8))
    graph_path = out_dir / "graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save node positions converted to WGS84 for convenience (lat, lon)
    try:
        print("Saving node positions in WGS84 for mapping convenience...")
        nodes_gdf = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in grid_points]}, crs=f"epsg:{WORK_EPSG}")
        nodes_wgs = nodes_gdf.to_crs(epsg=4326)
        coords_wgs = np.vstack([[geom.y, geom.x] for geom in nodes_wgs.geometry])  # lat, lon
        np.save(out_dir / "node_positions_wgs84.npy", coords_wgs)
    except Exception as e:
        print("Warning: failed to compute/save node_positions_wgs84.npy:", e)

    print("Precompute finished. Artifacts written to:", out_dir)
    return {
        "flood_geojson": str(out_geojson),
        "node_positions": str(node_positions_path),
        "var_per_node": str(var_path),
        "graph": str(graph_path),
    }


# -----------------------
# CLI
# -----------------------
def cli():
    parser = argparse.ArgumentParser(description="Precompute routing artifacts (simplified flood + routing graph).")
    parser.add_argument("--shapefile", "-s", required=True, help="Path to flood shapefile (.shp)")
    parser.add_argument("--outdir", "-o", default=str(DEFAULT_OUT_DIR), help="Output directory for artifacts")
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID_RES, help="Grid resolution in meters")
    parser.add_argument("--simplify", type=float, default=SIMPLIFY_TOL_M, help="Simplify tolerance in meters")
    args = parser.parse_args()

    shp = Path(args.shapefile)
    if not shp.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp}")

    precompute(
        shapefile_path=shp,
        out_dir=Path(args.outdir),
        grid_resolution=args.grid,
        simplify_tol_m=args.simplify,
    )


if __name__ == "__main__":
    cli()