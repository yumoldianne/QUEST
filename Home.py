# app.py
import os
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")  # help GDAL rebuild missing .shx if needed

import io
import tempfile
import zipfile
import types
from pathlib import Path
import hashlib
import pickle

import nbformat
import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import streamlit.components.v1 as components

st.set_page_config(page_title="Flood Evacuation Router", layout="wide")
st.title("🚨 Flood Evacuation Router — router.py backend")

# -------------------------
# Small helper functions
# -------------------------
def recommend_departure(distance_km):
    """Very simple recommended departure message based on walking time (5 km/h)."""
    if distance_km is None:
        return ""
    mins = (distance_km / 5.0) * 60.0
    if mins <= 10:
        return "Leave immediately if safe to do so."
    elif mins <= 30:
        return "Consider leaving within 10–15 minutes."
    else:
        return "Start moving now — this is a long walk."

# -------------------------
# Notebook loader (fallback)
# -------------------------
def load_module_from_notebook(nb_path: Path, module_name: str = "routing_nb"):
    """
    Execute code cells of a notebook into a fresh module and return it.
    Provides dummy `display` and `get_ipython` so notebook visualization calls won't break.
    """
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    code_strings = [cell.get("source", "") for cell in code_cells if cell.get("source", "")]
    combined = "\n\n".join(code_strings)

    module = types.ModuleType(module_name)
    module.__file__ = str(nb_path)

    # Provide dummy notebook helpers so display/get_ipython calls in the nb won't crash.
    def _dummy_display(*args, **kwargs):
        return None
    module.display = _dummy_display
    module.get_ipython = lambda: None

    # Execute notebook code in module namespace
    exec(compile(combined, str(nb_path), "exec"), module.__dict__)
    return module


def get_router_class(local_notebook_path: Path = Path("routing.ipynb"), uploaded_nb_bytes: bytes = None):
    """
    Return OptimizedFloodEvacuationRouter class.
    Priority:
      1) import router (router.py)
      2) uploaded notebook (if provided by user)
      3) local routing.ipynb file (if present)
    """
    # 1) Try import router.py module
    try:
        import router as router_module  # router.py should define OptimizedFloodEvacuationRouter
        if hasattr(router_module, "OptimizedFloodEvacuationRouter"):
            return router_module.OptimizedFloodEvacuationRouter
    except Exception:
        # ignore and try notebook fallback
        pass

    # 2) Use uploaded notebook bytes (if provided)
    if uploaded_nb_bytes is not None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "uploaded_routing.ipynb"
            tmp_path.write_bytes(uploaded_nb_bytes)
            mod = load_module_from_notebook(tmp_path, module_name="routing_uploaded")
            if hasattr(mod, "OptimizedFloodEvacuationRouter"):
                return getattr(mod, "OptimizedFloodEvacuationRouter")
            else:
                raise ImportError("Uploaded notebook executed but OptimizedFloodEvacuationRouter not found in it.")

    # 3) Try local routing.ipynb
    if local_notebook_path.exists():
        mod = load_module_from_notebook(local_notebook_path, module_name="routing_local")
        if hasattr(mod, "OptimizedFloodEvacuationRouter"):
            return getattr(mod, "OptimizedFloodEvacuationRouter")
        else:
            raise ImportError(f"Local notebook {local_notebook_path} executed but OptimizedFloodEvacuationRouter not found.")

    raise ImportError("Could not find OptimizedFloodEvacuationRouter in router.py or routing.ipynb (or uploaded notebook).")


# -------------------------
# Sidebar: inputs & settings
# -------------------------
st.sidebar.header("Inputs & Settings")

use_uploaded_nb = st.sidebar.checkbox("Upload routing.ipynb instead of using router.py", value=False)
uploaded_nb_file = None
if use_uploaded_nb:
    uploaded_nb_file = st.sidebar.file_uploader("Upload routing.ipynb", type=["ipynb"])

# Flood Geo data uploader (optional)
flood_file = st.sidebar.file_uploader(
    "Upload flood GeoJSON / GeoPackage / Shapefile (.geojson .gpkg .zip .shp) — optional",
    type=["geojson", "gpkg", "zip", "shp"],
)

# Evacuation centers uploader (CSV with 'name','lat','lon') optional
evac_file = st.sidebar.file_uploader("Upload evacuation centers CSV (name,lat,lon) — optional", type=["csv"])

# Grid resolution and simplification settings
grid_resolution = st.sidebar.slider("Grid resolution (meters)", min_value=20, max_value=200, value=50, step=10)
simplify_tol_m = st.sidebar.slider("Flood simplify tolerance (m)", min_value=50, max_value=2000, value=300, step=50)
show_flood_layer = st.sidebar.checkbox("Show flood layer on map", value=False)
light_mode = st.sidebar.checkbox("Light map mode (more simplification)", value=True)

# RISK penalty slider for soft avoidance (1.0 = no avoidance)
risk_penalty = st.sidebar.slider("Risk penalty factor (soft avoidance)", min_value=1.0, max_value=6.0, value=2.0, step=0.5)

# Dropdown sample locations
SAMPLE_LOCATIONS = [
    {"name": "Near UP Diliman", "lat": 14.6510, "lon": 121.0723},
    {"name": "QC Hall area", "lat": 14.6760, "lon": 121.0437},
    {"name": "Katipunan", "lat": 14.6488, "lon": 121.0786},
    {"name": "BGC (Taguig)", "lat": 14.5547, "lon": 121.0456},
    {"name": "Custom coordinates", "lat": None, "lon": None},
]

uploaded_locs = st.sidebar.file_uploader("Upload locations CSV (name,lat,lon) for dropdown", type=["csv"])
if uploaded_locs:
    try:
        locs_df = pd.read_csv(uploaded_locs)
        SAMPLE_LOCATIONS = locs_df.to_dict(orient="records")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded locations CSV: {e}")

# -------------------------
# Load evac centers: prefer uploaded sidebar, else local evac-centers.csv, else defaults
# -------------------------
evac_centers = None
if evac_file:
    try:
        evac_df = pd.read_csv(evac_file)
        evac_centers = [{"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"])} for _, row in evac_df.iterrows()]
    except Exception as e:
        st.sidebar.error(f"Failed to read evacuation centers CSV: {e}")
        evac_centers = None

local_evac_path = Path("evac-centers.csv")
if evac_centers is None and local_evac_path.exists():
    try:
        df_local = pd.read_csv(local_evac_path)
        new_centers = []
        # try common column names
        for _, r in df_local.iterrows():
            name = r.get("Center Name") or r.get("Name") or r.get("center") or r.get("center_name")
            lat = r.get("Latitude") or r.get("lat")
            lon = r.get("Longitude") or r.get("lon")
            if pd.notna(name) and pd.notna(lat) and pd.notna(lon):
                try:
                    new_centers.append({"name": str(name), "lat": float(lat), "lon": float(lon)})
                except Exception:
                    continue
        if new_centers:
            evac_centers = new_centers
            st.sidebar.success(f"Loaded {len(new_centers)} evacuation centers from evac-centers.csv")
    except Exception as e:
        st.sidebar.warning(f"Could not read local evac-centers.csv: {e}")

if evac_centers is None:
    evac_centers = [
        {"name": "QC Hall", "lat": 14.6760, "lon": 121.0437},
        {"name": "UP Diliman", "lat": 14.6507, "lon": 121.0494},
        {"name": "Sample Barangay Center", "lat": 14.6600, "lon": 121.0500},
    ]

# -------------------------
# Helpers: flood loader for uploaded file
# -------------------------
@st.cache_data(ttl=3600)
def load_flood_gdf_from_bytes(uploaded_file):
    """Load an uploaded flood vector into a GeoDataFrame (or return None)."""
    if uploaded_file is None:
        # if the local default shapefile exists, try to load it
        local = Path("ph137404000_fh5yr_30m_10m.shp")
        if local.exists():
            try:
                gdf = gpd.read_file(local)
                # ensure CRS set to EPSG:32651 if not present
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=32651, allow_override=True)
                return gdf
            except Exception as e:
                st.sidebar.warning(f"Failed to read local shapefile: {e}")
                return None
        return None

    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".geojson") or name.endswith(".json") or name.endswith(".gpkg"):
            gdf = gpd.read_file(io.BytesIO(file_bytes))
        elif name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                z = zipfile.ZipFile(io.BytesIO(file_bytes))
                z.extractall(tmp)
                shp = list(Path(tmp).rglob("*.shp"))
                if not shp:
                    raise RuntimeError("No .shp found inside zip")
                gdf = gpd.read_file(shp[0].as_posix())
        elif name.endswith(".shp"):
            # Shapefile without companions is brittle; but attempt reading
            gdf = gpd.read_file(io.BytesIO(file_bytes))
        else:
            gdf = gpd.read_file(io.BytesIO(file_bytes))

        # set CRS to EPSG:32651 if missing (your dataset uses 32651)
        if gdf is not None and gdf.crs is None:
            gdf = gdf.set_crs(epsg=32651, allow_override=True)

        return gdf
    except Exception as e:
        st.error(f"Failed to read uploaded flood file: {e}")
        return None


# -------------------------
# Build router (cached resource) - avoid hashing GeoDataFrame by leading "_" prefix
# -------------------------
@st.cache_resource
def build_router_class_and_instance(_flood_gdf_obj, grid_res, simplify_tol_m, evac_centers_list, uploaded_nb_bytes=None):
    """
    Load router class from router.py or notebook and return a built router instance (with network built).
    Leading underscore in the GeoDataFrame argument name prevents Streamlit from trying to hash it.
    """
    flood_gdf_obj = _flood_gdf_obj  # local name for readability

    # Load class
    try:
        RouterClass = get_router_class(local_notebook_path=Path("routing.ipynb"), uploaded_nb_bytes=uploaded_nb_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to load routing class from backend: {e}") from e

    # Instantiate router. Try common constructor signatures:
    try:
        router = RouterClass(flood_gdf_or_path=flood_gdf_obj, grid_resolution=grid_res)
    except TypeError:
        # fallback to alternate constructor patterns
        try:
            router = RouterClass(flood_gdf_obj, grid_resolution=grid_res)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate router class: {e}") from e

    # Set evac centers
    if hasattr(router, "set_evacuation_centers"):
        try:
            router.set_evacuation_centers(evac_centers_list)
        except Exception:
            # ignore if something odd; still proceed
            pass
    else:
        st.warning("Router class does not expose `set_evacuation_centers`. Make sure the class in router.py has that method.")

    # Build network (heavy)
    try:
        if hasattr(router, "flood_simplify_tol_m"):
            router.flood_simplify_tol_m = simplify_tol_m
    except Exception:
        pass

    router.build_routing_network(verbose=False, use_cache=True)
    return router


# -------------------------
# Utility: create a stable hash of inputs so we know when inputs changed
# -------------------------
def compute_input_hash(start_lat, start_lon, grid_res, simplify_tol_m, evac_centers, flood_file_name, risk_penalty):
    m = hashlib.sha256()
    m.update(f"{start_lat}_{start_lon}_{grid_res}_{simplify_tol_m}_{flood_file_name}_{risk_penalty}".encode())
    m.update(pickle.dumps(evac_centers))
    return m.hexdigest()


# -------------------------
# Helper: build and cache map HTML
# -------------------------
def _render_and_cache_map_html(router, routes, start_coords, simplify_tol_m, show_flood_layer, light_mode, input_hash_key):
    """
    Build folium map (via router.visualize_routes when possible), render to HTML,
    cache HTML string in st.session_state['map_html_<input_hash_key>'].
    Returns HTML string.
    """
    cache_key = f"map_html_{input_hash_key}"
    # Reuse if available and inputs unchanged
    if st.session_state.get("last_input_hash") == input_hash_key and cache_key in st.session_state:
        return st.session_state[cache_key]

    # Build the map using router if possible
    try:
        if router is not None and hasattr(router, "visualize_routes"):
            folium_map = router.visualize_routes(
                routes,
                start_coords,
                show_flood_layer=show_flood_layer,
                simplify_tol_m=(simplify_tol_m if not light_mode else max(simplify_tol_m, 500)),
                dissolve_flood=True,
                light_mode=light_mode,
            )
        else:
            # fallback: minimal folium map
            import folium as _folium
            folium_map = _folium.Map(location=start_coords, zoom_start=13)
            colors = ["#0000FF", "#00AA00", "#FF8C00"]
            for idx, route in enumerate(routes):
                _folium.PolyLine(route["path"], color=colors[idx % len(colors)], weight=4, opacity=0.8).add_to(folium_map)
                _folium.Marker(route["path"][-1], popup=route["center_name"]).add_to(folium_map)
            _folium.Marker(start_coords, popup="Start").add_to(folium_map)

        html = folium_map.get_root().render()
    except Exception as e:
        # fallback minimal map HTML
        try:
            import folium as _folium
            folium_map = _folium.Map(location=start_coords, zoom_start=13)
            html = folium_map.get_root().render()
        except Exception:
            html = f"<p>Could not render map: {e}</p>"

    # Cache HTML
    st.session_state[cache_key] = html
    st.session_state["last_map_html"] = html
    return html


# -------------------------
# Utility: render stored results (table + cached HTML)
# -------------------------
def render_results_from_state():
    routes = st.session_state.get("routes")
    start_coords = st.session_state.get("start_coords")
    if not routes or not start_coords:
        return

    # Summary table
    rows = []
    for i, r in enumerate(routes):
        rows.append(
            {
                "rank": i + 1,
                "center": r["center_name"],
                "distance_km": round(r["distance_km"], 3),
                "weighted_cost": round(r["weighted_cost"], 3),
                "time_min": int(r["estimated_time_min"]),
                "high_risk_m": int(r.get("high_risk_distance_m", 0)),
            }
        )
    st.subheader("Route options (top results)")
    st.table(pd.DataFrame(rows))

    # Map
    st.subheader("Map — selected route(s)")
    cache_key = f"map_html_{st.session_state.get('last_input_hash')}"
    html = st.session_state.get(cache_key) or st.session_state.get("last_map_html")
    if html:
        components.html(html, height=600, scrolling=True)
    else:
        # try to rebuild html from stored router + routes
        try:
            router = st.session_state.get("router")
            html = _render_and_cache_map_html(router, routes, start_coords, simplify_tol_m, show_flood_layer, light_mode, input_hash_key=st.session_state.get("last_input_hash"))
            components.html(html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Could not rebuild map from session: {e}")


# -------------------------
# Main UI (in a form so app doesn't rerun on every widget change)
# -------------------------
st.subheader("Choose your location")

with st.form(key="route_form"):
    names = [loc["name"] for loc in SAMPLE_LOCATIONS]
    selected_name = st.selectbox("Select a location (dropdown)", names)

    selected_loc = next(filter(lambda d: d["name"] == selected_name, SAMPLE_LOCATIONS), None)

    if selected_loc and selected_loc["lat"] is None:
        c1, c2 = st.columns(2)
        with c1:
            custom_lat = st.number_input("Latitude", value=14.6500, format="%.6f")
        with c2:
            custom_lon = st.number_input("Longitude", value=121.0300, format="%.6f")
        start_lat = custom_lat
        start_lon = custom_lon
    else:
        start_lat = selected_loc["lat"]
        start_lon = selected_loc["lon"]

    max_routes = st.slider("Maximum routes to show", min_value=1, max_value=3, value=3)

    submitted = st.form_submit_button("🔍 Find routes")

# Notebook upload bytes if user chose that option
uploaded_nb_bytes = None
if uploaded_nb_file is not None:
    uploaded_nb_bytes = uploaded_nb_file.read()

# Load flood gdf if uploaded
flood_gdf = load_flood_gdf_from_bytes(flood_file)
flood_file_name = flood_file.name if flood_file is not None else "default"

# Compute deterministic input hash
current_input_hash = compute_input_hash(start_lat, start_lon, grid_resolution, simplify_tol_m, evac_centers, flood_file_name, risk_penalty)
if "last_input_hash" not in st.session_state:
    st.session_state["last_input_hash"] = None

# If stored results exist and inputs didn't change, render them
if st.session_state.get("routes") and st.session_state.get("last_input_hash") == current_input_hash:
    render_results_from_state()

# -------------------------
# TABS: Landing, Evacuation Route, Relief Centers
# -------------------------
tabs = st.tabs(["Landing", "Evacuation Route", "Relief Centers"])

# ----- Landing: overview of flood-prone zones -----
with tabs[0]:
    st.header("Overview — Flood-prone Zones (Quezon City)")
    if flood_gdf is None:
        st.info("No flood layer loaded. Upload a flood shapefile/GeoJSON in the sidebar or place 'ph137404000_fh5yr_30m_10m.shp' in the app folder.")
    else:
        try:
            # simplify for web map
            flood_for_map = flood_gdf.to_crs(epsg=4326).copy()
            try:
                flood_for_map["geometry"] = flood_for_map.geometry.simplify(tolerance=0.0001)
            except Exception:
                pass
            import folium
            center = [flood_for_map.geometry.unary_union.centroid.y, flood_for_map.geometry.unary_union.centroid.x]
            m = folium.Map(location=center, zoom_start=12)
            def style_fn(feature):
                risk = feature["properties"].get("Var", 0)
                colors = {-2: "#CCCCCC", -1: "#FFFFCC", 0: "#FFFFFF", 1: "#FFEDA0", 2: "#FD8D3C", 3: "#E31A1C"}
                return {"fillColor": colors.get(risk, "#FFFFFF"), "color": "black", "weight": 0.3, "fillOpacity": 0.4}
            folium.GeoJson(flood_for_map.__geo_interface__, name="Flood Risk", style_function=style_fn).add_to(m)
            folium.LayerControl().add_to(m)
            components.html(m.get_root().render(), height=600, scrolling=True)
        except Exception as e:
            st.error(f"Failed to render landing map: {e}")

# ----- Evacuation Route: simpler working panel -----
with tabs[1]:
    st.header("Evacuation Route Panel")

    # If there's a local evac-centers.csv file, show note
    local_evac_path = Path("evac-centers.csv")
    if local_evac_path.exists():
        st.info("Using evac-centers.csv from app folder for evacuation centers.")

    # Location input
    col_a, col_b = st.columns([2, 1])
    with col_a:
        use_barangay = st.checkbox("Use barangay centroid as start", value=False)
        if use_barangay and 'barangay_gdf' in globals() and barangay_gdf is not None:
            try:
                chosen = st.selectbox("Choose Barangay", options=barangay_gdf['NAME'].tolist())
                sel = barangay_gdf[barangay_gdf['NAME'] == chosen]
                # compute union/centroid robustly (use union_all when available)
                try:
                    union_geom = sel.geometry.union_all() if hasattr(sel.geometry, "union_all") else sel.geometry.unary_union
                except Exception:
                    union_geom = sel.geometry.unary_union
                # get centroid in lat/lon for inputs
                try:
                    start_pt = gpd.GeoSeries([union_geom], crs=sel.crs).to_crs(epsg=4326).iloc[0].centroid
                    start_lat, start_lon = start_pt.y, start_pt.x
                except Exception:
                    start_lat, start_lon = 14.6500, 121.0300
            except Exception:
                start_lat, start_lon = 14.6500, 121.0300
        else:
            c1, c2 = st.columns(2)
            with c1:
                start_lat = st.number_input("Start latitude", value=14.6500, format="%.6f")
            with c2:
                start_lon = st.number_input("Start longitude", value=121.0300, format="%.6f")

        max_routes = st.number_input("Max routes to show", min_value=1, max_value=3, value=3)
        run_button = st.button("Compute routes")

    with col_b:
        st.subheader("Real-time road status (derived)")
        st.markdown("Road segments overlapping flood polygons are red; clear segments are green. (Heuristic)")

    # When user presses Compute
    if run_button:
        if flood_gdf is None:
            st.error("Flood layer required. Upload or place the shapefile in the app folder.")
        else:
            with st.spinner("Loading router class and building network (cached) — this may take a few seconds..."):
                # read uploaded notebook bytes if provided
                uploaded_nb_bytes = uploaded_nb_file.read() if uploaded_nb_file is not None else None

                # Build router instance (uses caching)
                try:
                    router = build_router_class_and_instance(
                        flood_gdf_obj=flood_gdf,
                        grid_res=grid_resolution,
                        simplify_tol_m=simplify_tol_m,
                        evac_centers_list=evac_centers,
                        uploaded_nb_bytes=uploaded_nb_bytes,
                    )
                except Exception as e:
                    st.error(f"Failed to prepare router: {e}")
                    st.stop()

            # Compute routes (try to pass risk_penalty; fallback if router doesn't accept it)
            try:
                try:
                    routes = router.find_routes_to_all_centers(start_lat, start_lon, max_routes=max_routes, risk_penalty_factor=float(risk_penalty))
                except TypeError:
                    routes = router.find_routes_to_all_centers(start_lat, start_lon, max_routes=max_routes)
            except Exception as e:
                st.error(f"Failed to compute routes: {e}")
                routes = []

            if not routes:
                st.info("No routes found. Try a different starting location or adjust grid resolution.")
                # clear previous session results so UI updates cleanly
                for k in ("routes", "router", "start_coords", "last_input_hash", "last_map_html"):
                    st.session_state.pop(k, None)
            else:
                # Save into session_state so results persist across reruns and avoid flicker
                st.session_state["router"] = router
                st.session_state["routes"] = routes
                st.session_state["start_coords"] = (start_lat, start_lon)

                # Build and store map HTML (use the existing helper which caches HTML by input hash)
                try:
                    current_input_hash = compute_input_hash(start_lat, start_lon, grid_resolution, simplify_tol_m, evac_centers, (flood_file.name if flood_file is not None else "default"), risk_penalty)
                    html = _render_and_cache_map_html(
                        router, routes, (start_lat, start_lon),
                        simplify_tol_m, show_flood_layer, light_mode,
                        input_hash_key=current_input_hash
                    )
                    # display the HTML
                    components.html(html, height=600, scrolling=True)
                    # store values for future re-render
                    st.session_state["last_input_hash"] = current_input_hash
                    st.session_state["start_coords"] = (start_lat, start_lon)
                except Exception as e:
                    st.error(f"Failed to render map: {e}")
                    # fallback: try st_folium direct rendering
                    try:
                        st_folium(router.visualize_routes(routes, (start_lat, start_lon)), width=1000, height=600)
                    except Exception:
                        st.error("Fallback rendering also failed.")

                # summary for best route
                try:
                    best = routes[0]
                    st.subheader("Route summary (best option)")
                    st.write(f"Destination: **{best['center_name']}**")
                    st.write(f"Distance: **{best['distance_km']:.2f} km**")
                    st.write(f"Weighted cost: **{best['weighted_cost']:.2f}**")
                    st.write(f"Estimated time: **{best['estimated_time_min']:.0f} min**")
                    st.write(recommend_departure(best['distance_km']))
                except Exception:
                    pass

# ----- Relief Centers: list and simple accessibility heuristic -----
with tabs[2]:
    st.header("Relief Centers")
    st.write("Relief centers loaded from evac-centers.csv (if present) or from uploaded CSV / defaults.")

    try:
        rc_df = pd.DataFrame(evac_centers)
        if not rc_df.empty:
            # compute simple accessibility heuristic if flood layer present
            if flood_gdf is not None and 'Var' in flood_gdf.columns:
                acc = []
                for r in evac_centers:
                    try:
                        pt = gpd.GeoSeries([gpd.points_from_xy([r['lon']],[r['lat']])[0]], crs="epsg:4326").to_crs(flood_gdf.crs)
                        buf = pt.buffer(200).iloc[0]
                        high = flood_gdf[flood_gdf['Var']==3]
                        frac = 0.0
                        if not high.empty:
                            inter = high.geometry.intersection(buf)
                            try:
                                inter_union = inter.unary_union
                                frac = (inter_union.area / buf.area) if buf.area>0 else 0.0
                            except Exception:
                                frac = 0.0
                        acc.append(max(0, 1.0 - frac))
                    except Exception:
                        acc.append(None)
                rc_df['accessibility'] = [round(float(x),2) if x is not None else None for x in acc]
            else:
                rc_df['accessibility'] = [None]*len(rc_df)

            st.dataframe(rc_df)
            # small map of relief centers
            try:
                import folium
                center = [rc_df['lat'].mean(), rc_df['lon'].mean()]
                mrc = folium.Map(location=center, zoom_start=12)
                for _, row in rc_df.iterrows():
                    folium.Marker([row['lat'], row['lon']], popup=row['name'], icon=folium.Icon(color='green', icon='home', prefix='fa')).add_to(mrc)
                components.html(mrc.get_root().render(), height=500, scrolling=True)
            except Exception:
                pass
        else:
            st.info("No relief center data available.")
    except Exception as e:
        st.error(f"Could not display relief centers: {e}")

# Footer tips
st.markdown("---")
st.markdown(
    """
**Tips**
- If building the network takes too long, increase grid resolution (larger = fewer nodes) or upload a smaller flood polygon file.
- For fastest map rendering, uncheck "Show flood layer".
- To use your own evacuation centers, upload a CSV with `name,lat,lon` in the sidebar or place evac-centers.csv in the app folder.
"""
)
