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

if evac_file:
    try:
        evac_df = pd.read_csv(evac_file)
        evac_centers = [{"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"])} for _, row in evac_df.iterrows()]
    except Exception as e:
        st.sidebar.error(f"Failed to read evacuation centers CSV: {e}")
        evac_centers = None
else:
    evac_centers = None

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
        return None
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".geojson") or name.endswith(".json") or name.endswith(".gpkg"):
            return gpd.read_file(io.BytesIO(file_bytes))
        elif name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                z = zipfile.ZipFile(io.BytesIO(file_bytes))
                z.extractall(tmp)
                shp = list(Path(tmp).rglob("*.shp"))
                if not shp:
                    raise RuntimeError("No .shp found inside zip")
                return gpd.read_file(shp[0].as_posix())
        elif name.endswith(".shp"):
            # Shapefile without companions is brittle; but attempt reading
            return gpd.read_file(io.BytesIO(file_bytes))
        else:
            return gpd.read_file(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Failed to read uploaded flood file: {e}")
        return None


# -------------------------
# Build router (cached resource)
# -------------------------
@st.cache_resource
def build_router_class_and_instance(flood_gdf_obj, grid_res, simplify_tol_m, evac_centers_list, uploaded_nb_bytes=None):
    """
    Load router class from router.py or notebook and return a built router instance (with network built).
    This is cached by Streamlit to avoid rebuilding on every interaction.
    """
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
        router.set_evacuation_centers(evac_centers_list)
    else:
        st.warning("Router class does not expose `set_evacuation_centers`. Make sure the class in router.py has that method.")

    # Build network (heavy)
    router.flood_simplify_tol_m = simplify_tol_m if hasattr(router, "flood_simplify_tol_m") else simplify_tol_m
    router.build_routing_network(verbose=False, use_cache=True)
    return router


# -------------------------
# Utility: create a stable hash of inputs so we know when inputs changed
# -------------------------
def compute_input_hash(start_lat, start_lon, grid_res, simplify_tol_m, evac_centers, flood_file_name):
    m = hashlib.sha256()
    m.update(f"{start_lat}_{start_lon}_{grid_res}_{simplify_tol_m}_{flood_file_name}".encode())
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
                "high_risk_m": int(r["high_risk_distance_m"]),
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
current_input_hash = compute_input_hash(start_lat, start_lon, grid_resolution, simplify_tol_m, evac_centers, flood_file_name)
if "last_input_hash" not in st.session_state:
    st.session_state["last_input_hash"] = None

# If stored results exist and inputs didn't change, render them
if st.session_state.get("routes") and st.session_state.get("last_input_hash") == current_input_hash:
    render_results_from_state()

# If user submitted the form, compute routes (explicit trigger)
if submitted:
    if start_lat is None or start_lon is None:
        st.warning("Please provide coordinates (choose a location or enter custom coordinates).")
    else:
        with st.spinner("Loading router class and building network (cached) — this may take a few seconds..."):
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

        # Compute routes
        try:
            routes = router.find_routes_to_all_centers(start_lat, start_lon, max_routes=max_routes)
        except Exception as e:
            st.error(f"Failed to compute routes: {e}")
            routes = []

        if not routes:
            st.info("No routes found. Try different starting location or adjust grid resolution.")
            # clear previous session results
            for k in ("routes", "router", "start_coords", "last_input_hash", "last_map_html"):
                st.session_state.pop(k, None)
        else:
            # Save into session_state so results persist across reruns
            st.session_state["router"] = router
            st.session_state["routes"] = routes
            st.session_state["start_coords"] = (start_lat, start_lon)
            st.session_state["last_input_hash"] = current_input_hash

            # build and store map HTML for persistence (use helper)
            try:
                html = _render_and_cache_map_html(
                    router, routes, (start_lat, start_lon),
                    simplify_tol_m, show_flood_layer, light_mode,
                    input_hash_key=current_input_hash
                )
                # display the HTML
                components.html(html, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Failed to render map: {e}")
                # fallback: try st_folium
                try:
                    st_folium(router.visualize_routes(routes, (start_lat, start_lon)), width=1000, height=600)
                except Exception:
                    st.error("Fallback rendering also failed.")

            # Render summary table below map
            render_results_from_state()

st.markdown("---")
st.markdown(
    """
**Tips**
- If building the network takes too long, increase grid resolution (larger = fewer nodes) or upload a smaller flood polygon file.
- For fastest map rendering, uncheck "Show flood layer".
- To use your own evacuation centers, upload a CSV with `name,lat,lon` in the sidebar.
"""
)
