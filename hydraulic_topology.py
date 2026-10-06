import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from config import (
    K_NEIGHBORS,
    MAX_EDGE_DISTANCE
)


# =========================
# SAFE NUMERIC CONVERSION
# =========================
def safe_numeric(series):

    return pd.to_numeric(

        series.astype(str)

        .str.strip()

        .str.replace(" ", "", regex=False)

        .str.replace("\u00A0", "", regex=False)

        .str.replace(",", ".", regex=False),

        errors="coerce"
    )


# =========================
# REMOVE DUPLICATE EDGES
# =========================
def remove_duplicate_edges(edges):

    unique_edges = set()

    for src, tgt in edges:

        if src == tgt:
            continue

        unique_edges.add((
            int(src),
            int(tgt)
        ))

    return list(unique_edges)


# =========================
# HYDRAULIC TOPOLOGY
# =========================
def reconstruct_hydraulic_edges(
    df_nodes,
    x_col="coordx",
    y_col="coordy",
    elevation_col="altitude",
    k=K_NEIGHBORS,
    max_distance=MAX_EDGE_DISTANCE
):

    print(
        "   → Hydraulic-directed reconstruction..."
    )

    # =========================
    # COPY DATA
    # =========================
    df = df_nodes.copy()

    # =========================
    # VALIDATE COLUMNS
    # =========================
    required_cols = [

        x_col,
        y_col,
        elevation_col
    ]

    for col in required_cols:

        if col not in df.columns:

            raise ValueError(
                f"Missing column: {col}"
            )

    # =========================
    # SAFE CONVERSION
    # =========================
    df[x_col] = safe_numeric(
        df[x_col]
    )

    df[y_col] = safe_numeric(
        df[y_col]
    )

    df[elevation_col] = safe_numeric(
        df[elevation_col]
    )

    # =========================
    # REMOVE INVALID ROWS
    # =========================
    valid_mask = (

        df[x_col].notna()

        & df[y_col].notna()

        & df[elevation_col].notna()
    )

    df = df.loc[
        valid_mask
    ].copy()

    # =========================
    # RESET INDEX
    # =========================
    df = df.reset_index(
        drop=True
    )

    # =========================
    # GRAPH INDEX
    # =========================
    df["graph_idx"] = np.arange(
        len(df)
    )

    print(
        f"   → Valid hydraulic nodes: "
        f"{len(df)}"
    )

    # =========================
    # COORDINATES
    # =========================
    coords = df[
        [x_col, y_col]
    ].values.astype(float)

    elevations = df[
        elevation_col
    ].values.astype(float)

    # =========================
    # FINITE CHECK
    # =========================
    finite_mask = (

        np.isfinite(coords).all(axis=1)

        & np.isfinite(elevations)
    )

    coords = coords[
        finite_mask
    ]

    elevations = elevations[
        finite_mask
    ]

    df = df.loc[
        finite_mask
    ].reset_index(drop=True)

    df["graph_idx"] = np.arange(
        len(df)
    )

    print(
        f"   → Finite nodes: "
        f"{len(coords)}"
    )

    # =========================
    # EMPTY SAFETY
    # =========================
    if len(coords) == 0:

        raise ValueError(
            "No valid coordinates available."
        )

    # =========================
    # BUILD KD TREE
    # =========================
    tree = cKDTree(coords)

    edges = []

    downhill_count = 0

    equal_altitude_count = 0

    total_connections = 0

    isolated_nodes = 0

    # =========================
    # LOCAL HYDRAULIC LINKS
    # =========================
    for i in range(len(coords)):

        point = coords[i]

        distances, neighbors = tree.query(

            point,

            k=min(
                k + 1,
                len(coords)
            )
        )

        # =====================
        # SINGLE NEIGHBOR CASE
        # =====================
        if np.isscalar(distances):

            distances = [distances]

            neighbors = [neighbors]

        node_has_connection = False

        for dist, j in zip(

            distances[1:],
            neighbors[1:]
        ):

            j = int(j)

            # =====================
            # SELF LOOP
            # =====================
            if i == j:
                continue

            # =====================
            # MAX DISTANCE
            # =====================
            if dist > max_distance:
                continue

            elev_i = elevations[i]

            elev_j = elevations[j]

            # =====================
            # INVALID ELEVATION
            # =====================
            if (

                not np.isfinite(elev_i)

                or not np.isfinite(elev_j)
            ):

                continue

            total_connections += 1

            node_has_connection = True

            # =====================
            # STRICT DOWNHILL FLOW
            # =====================
            if elev_i > elev_j:

                edges.append((
                    int(i),
                    int(j)
                ))

                downhill_count += 1

            elif elev_j > elev_i:

                edges.append((
                    int(j),
                    int(i)
                ))

                downhill_count += 1

            # =====================
            # SAME ELEVATION
            # =====================
            else:

                # VERY CLOSE ONLY
                if dist <= 2.0:

                    smaller = min(i, j)

                    larger = max(i, j)

                    edges.append((
                        int(smaller),
                        int(larger)
                    ))

                    equal_altitude_count += 1

        # =====================
        # ISOLATED NODE
        # =====================
        if not node_has_connection:

            isolated_nodes += 1

    # =========================
    # REMOVE DUPLICATES
    # =========================
    edges = remove_duplicate_edges(
        edges
    )

    # =========================
    # SAFETY
    # =========================
    if len(edges) == 0:

        raise ValueError(
            "No hydraulic edges reconstructed."
        )

    # =========================
    # EDGE INDEX
    # =========================
    edge_index = np.array(
        edges,
        dtype=int
    ).T

    # =========================
    # GRAPH ANALYSIS
    # =========================
    connected_nodes = np.unique(
        edge_index
    )

    connectivity_ratio = (

        len(connected_nodes)

        / len(coords)
    )

    # =========================
    # LOGGING
    # =========================
    print(
        f"   → Hydraulic edges: "
        f"{edge_index.shape[1]}"
    )

    print(
        f"   → Connected nodes: "
        f"{len(connected_nodes)}"
    )

    print(
        f"   → Connectivity ratio: "
        f"{connectivity_ratio:.3f}"
    )

    print(
        f"   → Isolated nodes: "
        f"{isolated_nodes}"
    )

    if total_connections > 0:

        downhill_ratio = (

            downhill_count

            / total_connections
        )

        print(
            f"   → Downhill ratio: "
            f"{downhill_ratio:.3f}"
        )

    print(
        f"   → Equal elevation links: "
        f"{equal_altitude_count}"
    )

    print(
        f"   → Max edge distance: "
        f"{max_distance} m"
    )

    print(
        f"   → K neighbors: "
        f"{k}"
    )

    # =========================
    # RETURN
    # =========================
    return edge_index, df