# QUEST: Quezon City Evacuation Support Tool

**Fast, safety-first routing for flood evacuations (Quezon City demo).**
Compute safe, practical evacuation routes from a user location to nearby relief centers while minimizing exposure to flood-prone areas.

## Why this matters

During floods, the *safest* route is not always the shortest. SafeRouteQC helps residents and local responders find realistic evacuation routes that **balance travel time with flood exposure**, using an interpretable soft-avoidance model and a compact, interactive dashboard.

## Highlights / Quick features

* Interactive Streamlit dashboard with three tabs: **Landing**, **Evacuation Route**, and **Relief Centers**.
* **Smart grid graph** built from flood polygons to approximate walkable space.
* **Soft-avoidance** risk model — user-controlled slider penalizes risky areas (1.0 = no avoidance, higher = safer routing).
* Up to **three ranked route options** with distance, risk exposure breakdown, and estimated time.
* Relief center list and accessibility heuristic.


## One-line technical summary

Build a risk-labeled spatial grid → connect neighbors (graph) → assign edge weights (distance × risk cost × soft penalty) → run A* to compute safe routes → render on an interactive map.

## How does the algorithm work?

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

* Team: Juliana Ambrosio (5 BS MIS), Caitlyn Lee (M DSc), Jan Manzano (4 BS MIS), Andrea Senson (BS AMDSc '25), and Dianne Yumol (M DSc)