# router.py
"""
OptimizedFloodEvacuationRouter module
Auto-ready: tries to load default shapefile ph137404000_fh5yr_30m_10m.shp if no input provided.
"""

import os
import pickle
import warnings
from pathlib import Path
import tempfile
import subprocess

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import folium

warnings.filterwarnings("ignore")

# ---- Defaults ----
DEFAULT_PROJ_EPSG = 32651
DEFAULT_SHAPEFILE_NAME = "ph137404000_fh5yr_30m_10m.shp"

# ---- helper loader with repair attempts ----
def load_flood_gdf(path_or_gdf=None):
    """
    Robust loader for flood shapefile or GeoDataFrame.
    - If path_or_gdf is None, attempts to load DEFAULT_SHAPEFILE_NAME from cwd.
    - Attempts to repair missing .shx by setting SHAPE_RESTORE_SHX or using ogr2ogr if available.
    Returns GeoDataFrame projected to DEFAULT_PROJ_EPSG.
    """
    # If caller passed None, try default shapefile in cwd
    if path_or_gdf is None:
        candidate = Path(DEFAULT_SHAPEFILE_NAME)
        if candidate.exists():
            path_or_gdf = candidate
        else:
            # Return empty GeoDataFrame in projected CRS
            empty = gpd.GeoDataFrame(columns=["geometry", "Var"], crs="EPSG:4326").to_crs(epsg=DEFAULT_PROJ_EPSG)
            return empty

    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        gdf = path_or_gdf.copy()
    else:
        path = Path(path_or_gdf)
        if not path.exists():
            raise FileNotFoundError(f"Flood file not found: {path}")
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            # Try GDAL repair env
            os.environ['SHAPE_RESTORE_SHX'] = 'YES'
            try:
                gdf = gpd.read_file(path)
            except Exception:
                # Try ogr2ogr rewrite
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        repaired_path = Path(tmpdir) / path.name
                        cmd = ["ogr2ogr", "-f", "ESRI Shapefile", str(repaired_path), str(path)]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        gdf = gpd.read_file(str(repaired_path))
                except FileNotFoundError:
                    raise RuntimeError(
                        f"Failed to open shapefile and 'ogr2ogr' not available to repair it. "
                        "Install GDAL/ogr2ogr or recreate the shapefile with its .shx index."
                    )
                except subprocess.CalledProcessError as cp:
                    raise RuntimeError(
                        "ogr2ogr failed to rewrite the shapefile (it may be corrupt). "
                        f"ogr2ogr stderr: {cp.stderr.decode('utf-8', errors='ignore')}"
                    ) from cp
                except Exception as e2:
                    raise RuntimeError("Failed to repair shapefile. Try regenerating it or set SHAPE_RESTORE_SHX=YES.") from e2

    # Ensure Var exists
    if "Var" not in gdf.columns:
        gdf["Var"] = 0

    # Project to working CRS if not already
    try:
        gdf = gdf.to_crs(epsg=DEFAULT_PROJ_EPSG)
    except Exception:
        # If to_crs fails (likely missing CRS), try to infer
        minx, miny, maxx, maxy = gdf.total_bounds
        if (-180 <= minx <= 180) and (-90 <= miny <= 90):
            warnings.warn("Input flood file has no CRS metadata — assuming EPSG:4326 (lat/lon).")
            gdf = gdf.set_crs(epsg=4326, allow_override=True).to_crs(epsg=DEFAULT_PROJ_EPSG)
        else:
            warnings.warn(f"Input flood file has no CRS metadata — assuming EPSG:{DEFAULT_PROJ_EPSG}.")
            gdf = gdf.set_crs(epsg=DEFAULT_PROJ_EPSG, allow_override=True)

    return gdf

# ---- small filesystem util ----
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---- Router class ----
class OptimizedFloodEvacuationRouter:
    def __init__(self, flood_gdf_or_path=None, grid_resolution: int=50, cache_dir: str="./cache", flood_simplify_tol_m: float=200.0, flood_dissolve: bool=True):
        self.flood_gdf = load_flood_gdf(flood_gdf_or_path)
        self.grid_resolution = float(grid_resolution)
        self.graph = None
        self.node_positions = None
        self.evacuation_centers = []
        self.evacuation_names = []
        self.cache_dir = Path(cache_dir)
        _safe_mkdir(self.cache_dir)
        self.minx, self.miny, self.maxx, self.maxy = self.flood_gdf.total_bounds
        self.flood_simplify_tol_m = float(flood_simplify_tol_m)
        self.flood_dissolve = bool(flood_dissolve)
        self._simplified_flood_gdf = None

    def _get_cache_path(self):
        bounds_str = f"{int(self.minx)}_{int(self.maxx)}_{int(self.miny)}_{int(self.maxy)}"
        return self.cache_dir / f"router_grid_{int(self.grid_resolution)}_{bounds_str}.pkl"

    def load_from_cache(self):
        p = self._get_cache_path()
        if not p.exists():
            return False
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self.node_positions = data["node_positions"]
            return True
        except Exception:
            return False

    def save_to_cache(self):
        p = self._get_cache_path()
        with open(p, "wb") as f:
            pickle.dump({"graph": self.graph, "node_positions": self.node_positions}, f)

    def set_evacuation_centers(self, centers_data):
        if not centers_data:
            self.evacuation_centers = []
            self.evacuation_names = []
            return
        if isinstance(centers_data[0], dict):
            self.evacuation_names = [c.get("name", f"Center {i+1}") for i, c in enumerate(centers_data)]
            points = [Point(c["lon"], c["lat"]) for c in centers_data]
        else:
            self.evacuation_names = [f"Center {i+1}" for i in range(len(centers_data))]
            points = [Point(lon, lat) for lat, lon in centers_data]
        self.evacuation_centers = gpd.GeoSeries(points, crs="epsg:4326").to_crs(epsg=DEFAULT_PROJ_EPSG)

    def _create_smart_grid(self):
        x_coords = np.arange(self.minx, self.maxx + 1e-6, self.grid_resolution)
        y_coords = np.arange(self.miny, self.maxy + 1e-6, self.grid_resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        total_bounds = self.flood_gdf.unary_union.bounds
        buffer_dist = self.grid_resolution * 2.0
        bounds_poly = box(total_bounds[0]-buffer_dist, total_bounds[1]-buffer_dist, total_bounds[2]+buffer_dist, total_bounds[3]+buffer_dist)
        mask = (
            (grid_points[:, 0] >= bounds_poly.bounds[0])
            & (grid_points[:, 0] <= bounds_poly.bounds[2])
            & (grid_points[:, 1] >= bounds_poly.bounds[1])
            & (grid_points[:, 1] <= bounds_poly.bounds[3])
        )
        return grid_points[mask]

    def _batch_spatial_join(self, points: np.ndarray, batch_size: int = 10000):
        risk_levels = np.zeros(len(points), dtype=np.int8)
        points_gdf = gpd.GeoDataFrame({"geometry":[Point(x,y) for x,y in points]}, index=np.arange(len(points)), crs=f"epsg:{DEFAULT_PROJ_EPSG}")
        for i in range(0, len(points_gdf), batch_size):
            batch = points_gdf.iloc[i:i+batch_size]
            joined = gpd.sjoin(batch, self.flood_gdf[["geometry","Var"]], how="left", predicate="within")
            if not joined.empty:
                var_per_point = joined["Var"].groupby(joined.index).max()
            else:
                var_per_point = gpd.GeoSeries(dtype=float)
            aligned = var_per_point.reindex(batch.index).fillna(0).astype(np.int8).values
            risk_levels[i:i+len(batch)] = aligned
        return risk_levels

    def prepare_simplified_flood(self, simplify_tol_m=None, dissolve=None, force_refresh: bool=False):
        if simplify_tol_m is None:
            simplify_tol_m = self.flood_simplify_tol_m
        if dissolve is None:
            dissolve = self.flood_dissolve
        if self._simplified_flood_gdf is not None and not force_refresh:
            return self._simplified_flood_gdf
        if self.flood_gdf.empty:
            self._simplified_flood_gdf = gpd.GeoDataFrame(columns=["geometry","Var"], crs=self.flood_gdf.crs)
            return self._simplified_flood_gdf
        if dissolve:
            rows=[]
            for var_val in sorted(self.flood_gdf["Var"].unique()):
                subset = self.flood_gdf[self.flood_gdf["Var"]==var_val]
                if subset.empty:
                    continue
                union_geom = subset.unary_union
                try:
                    simp = union_geom.simplify(simplify_tol_m)
                except Exception:
                    simp = union_geom
                rows.append({"Var": var_val, "geometry": simp})
            simp_gdf = gpd.GeoDataFrame(rows, crs=self.flood_gdf.crs)
        else:
            simp_gdf = self.flood_gdf.copy()
            simp_gdf["geometry"] = simp_gdf.geometry.simplify(simplify_tol_m)
        self._simplified_flood_gdf = simp_gdf
        return simp_gdf

    def build_routing_network(self, verbose: bool=True, use_cache: bool=True):
        if use_cache and self.load_from_cache():
            if verbose:
                print("Loaded from cache")
            return
        grid_points = self._create_smart_grid()
        risk_levels = self._batch_spatial_join(grid_points)
        G = nx.Graph()
        for idx, (pt, risk) in enumerate(zip(grid_points, risk_levels)):
            G.add_node(int(idx), pos=(float(pt[0]), float(pt[1])), risk=int(risk))
        tree = cKDTree(grid_points)
        max_dist = self.grid_resolution * 1.5
        neighbor_lists = tree.query_ball_tree(tree, max_dist)
        risk_costs = np.array([1.0,1.0,1.5,1.0,2.0,5.0,20.0])
        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if j <= i:
                    continue
                dist = np.linalg.norm(grid_points[i]-grid_points[j])
                if dist >= max_dist:
                    continue
                ri = int(risk_levels[i]); rj=int(risk_levels[j])
                idx_i = np.clip(ri+2,0,len(risk_costs)-1)
                idx_j = np.clip(rj+2,0,len(risk_costs)-1)
                avg_cost = (risk_costs[idx_i]+risk_costs[idx_j])/2.0
                weight = float(dist*avg_cost)
                G.add_edge(int(i), int(j), weight=weight, distance=float(dist))
        self.graph = G
        self.node_positions = grid_points
        if use_cache:
            try:
                self.save_to_cache()
            except Exception:
                pass

    def find_nearest_node(self, lat: float, lon: float):
        if self.node_positions is None:
            raise ValueError("Network not built")
        pt = gpd.GeoSeries([Point(lon, lat)], crs="epsg:4326").to_crs(epsg=DEFAULT_PROJ_EPSG).iloc[0]
        tree = cKDTree(self.node_positions)
        _, idx = tree.query([pt.x, pt.y])
        return int(idx)

    def find_routes_to_all_centers(self, start_lat: float, start_lon: float, max_routes: int=3):
        if self.graph is None:
            raise ValueError("Network not built")
        start_node = self.find_nearest_node(start_lat, start_lon)
        evac_nodes = []
        tree = cKDTree(self.node_positions)
        for center in self.evacuation_centers:
            _, idx = tree.query([center.x, center.y])
            evac_nodes.append(int(idx))
        def heuristic(n1,n2):
            return np.linalg.norm(self.node_positions[n1]-self.node_positions[n2])
        routes=[]
        for i, end_node in enumerate(evac_nodes):
            try:
                path = nx.astar_path(self.graph, start_node, end_node, heuristic=heuristic, weight="weight")
            except nx.NetworkXNoPath:
                continue
            total_distance=0.0; total_weighted_cost=0.0
            risk_breakdown={k:0.0 for k in range(-2,4)}
            for u,v in zip(path[:-1], path[1:]):
                ed = self.graph.get_edge_data(u,v)
                total_distance+=float(ed["distance"])
                total_weighted_cost+=float(ed["weight"])
                risk=self.graph.nodes[u]["risk"]
                risk_breakdown[risk]+=float(ed["distance"])
            coords=[]
            for n in path:
                x,y = self.node_positions[n]
                p = gpd.GeoSeries([Point(x,y)], crs=f"epsg:{DEFAULT_PROJ_EPSG}").to_crs(epsg=4326).iloc[0]
                coords.append((p.y, p.x))
            routes.append({
                "center_name": self.evacuation_names[i] if i < len(self.evacuation_names) else f"Center {i+1}",
                "path": coords,
                "distance_m": total_distance,
                "distance_km": total_distance/1000.0,
                "weighted_cost": total_weighted_cost,
                "risk_breakdown": risk_breakdown,
                "high_risk_distance_m": risk_breakdown.get(3,0.0),
                "medium_risk_distance_m": risk_breakdown.get(2,0.0),
                "low_risk_distance_m": risk_breakdown.get(1,0.0),
                "safe_distance_m": sum(risk_breakdown.get(k,0.0) for k in (-2,-1,0)),
                "estimated_time_min": (total_distance/1000.0)/5.0*60.0
            })
        routes.sort(key=lambda r: r["weighted_cost"])
        return routes[:max_routes]

    def visualize_routes(self, routes, start_coords, show_flood_layer: bool=False, simplify_tol_m=None, dissolve_flood=None, light_mode: bool=False):
        m = folium.Map(location=start_coords, zoom_start=13)
        if light_mode:
            if simplify_tol_m is None:
                simplify_tol_m = max(self.flood_simplify_tol_m, 500)
            else:
                simplify_tol_m = max(simplify_tol_m, 200)
        elif simplify_tol_m is None:
            simplify_tol_m = self.flood_simplify_tol_m
        if dissolve_flood is None:
            dissolve_flood = self.flood_dissolve
        if show_flood_layer and not self.flood_gdf.empty:
            simp = self.prepare_simplified_flood(simplify_tol_m=simplify_tol_m, dissolve=dissolve_flood)
            if not simp.empty:
                flood_layer = simp.to_crs(epsg=4326).copy()
                try:
                    tol_deg = simplify_tol_m/111000.0
                    flood_layer["geometry"] = flood_layer.geometry.simplify(tol_deg)
                except Exception:
                    pass
                def style_function(feature):
                    risk = feature["properties"].get("Var", 0)
                    colors = {-2:"#CCCCCC", -1:"#FFFFCC", 0:"#FFFFFF", 1:"#FFEDA0", 2:"#FD8D3C", 3:"#E31A1C"}
                    return {"fillColor": colors.get(risk, "#FFFFFF"), "color":"black","weight":0.5,"fillOpacity":0.3}
                folium.GeoJson(flood_layer, style_function=style_function, name="Flood Risk (simplified)").add_to(m)
        route_colors=["#0000FF","#00AA00","#FF8C00"]
        for idx, route in enumerate(routes):
            color = route_colors[idx % len(route_colors)]
            folium.PolyLine(route["path"], color=color, weight=4, opacity=0.8, popup=f"{route['center_name']}").add_to(m)
            folium.Marker(route["path"][-1], popup=route["center_name"], icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
        folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="green", icon="user", prefix="fa")).add_to(m)
        folium.LayerControl().add_to(m)
        return m
