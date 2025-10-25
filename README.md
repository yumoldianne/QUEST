# QUEST: Quezon City Evacuation Support Tool

**Fast, safety-first routing for flood evacuations.**

Compute safe, practical evacuation routes from a user location to nearby relief centers while minimizing exposure to flood-prone areas.

## Why does it matter?

During floods, the *safest* route is not always the shortest. QUEST helps residents and local responders find realistic evacuation routes that **balance travel time with flood exposure**, using an interpretable soft-avoidance model and a compact, interactive dashboard.

## Highlights

* Interactive Streamlit dashboard with three tabs: **Landing**, **Evacuation Route**, and **Relief Centers**.
* **Smart grid graph** built from flood polygons to approximate walkable space.
* **Soft-avoidance** risk model — user-controlled slider penalizes risky areas (1.0 = no avoidance, higher = safer routing).
* Up to **three ranked route options** with distance, risk exposure breakdown, and estimated time.
* Relief center list and accessibility heuristic.


## Technical summary

**Flood-routing algorithm:** Create a geo-referenced grid labeled by flood risk, connect neighbouring cells into a graph with edge weights = distance × risk cost × user-controlled penalty, run A* to find low-risk shortest routes, and display the results on an interactive map.

**Accesibility hueristic**: Measure accessibility around each relief center by buffering the center (default 200 m), seeing how much of that buffer overlaps high-risk flood polygons (Var == 3), and converting that overlap fraction into an accessibility score: accessibility = max(0, 1 − (area_of_high_risk_within_buffer / buffer_area)), with 1.0 = fully accessible (no high-risk area in the buffer), 0.0 = fully surrounded by high risk.

## How does the flood routing algorithm work?

1. **Preprocess**

   * Read flood polygons, ensure CRS = **EPSG:32651** (meters).
   * Optionally simplify/dissolve by risk level for fast rendering.

2. **Smart grid creation**

   * Generate a regular grid over flood extent at `grid_resolution` meters.
   * Keep only grid nodes inside a buffered flood bounding region to reduce nodes.

3. **Assign risk**

   * Use batched spatial joins to assign `Var` risk to nodes (vectorized for speed).

4. **Graph construction**

   * Connect each node to nearby neighbors (KD-tree) and compute:

     * `distance` (m)
     * `weight` = `distance × avg_risk_cost` (base), stored on edges.

5. **Soft avoidance**

   * Adjust `weight` by a **risk multiplier** derived from node risks and a user-provided `risk_penalty_factor` (≥1).
   * This penalizes routes that go through risky cells rather than forbidding them.

6. **Pathfinding**

   * A* with Euclidean heuristic on projected coordinates to each evacuation center node.
   * Return up to 3 ranked routes (lowest weighted cost).

## Authors & acknowledgements

* Team: Juliana Ambrosio (BS MIS '26), Caitlyn Lee (BS AMDSc '25, M DSc '26), Jan Manzano (BS MIS '26) , Andrea Senson (BS AMDSc '25), and Dianne Yumol (BS AMDSc '25, M DSc '26)