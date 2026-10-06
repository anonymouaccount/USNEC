import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data

from config import (
    NODE_ID_COL,
    SOURCE_COL,
    TARGET_COL,
    NODE_FEATURES,
    EDGE_FEATURES,
    TEST_EDGE_RATIO,
    ENABLE_SPATIAL_RECONSTRUCTION,
    K_NEIGHBORS,
    MAX_EDGE_DISTANCE
)

from utils import (
    print_step
)


# =========================================================
# BUILD GRAPH
# =========================================================
def build_graph(
    df_nodes,
    df_arcs,
    df_basins
):

    # =====================================================
    # NODE FEATURES
    # =====================================================
    print_step(
        "Preparing node features..."
    )

    feature_columns = []

    for col in NODE_FEATURES:

        if col in df_nodes.columns:

            feature_columns.append(col)

    print(
        f"   → Node feature columns: "
        f"{feature_columns}"
    )

    # =====================================================
    # KEEP RAW GIS VALUES
    # IMPORTANT:
    # USED FOR HYDRAULIC VALIDATION
    # =====================================================
    raw_coordx = (
        pd.to_numeric(
            df_nodes["coordx"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    raw_coordy = (
        pd.to_numeric(
            df_nodes["coordy"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    raw_altitude = (
        pd.to_numeric(
            df_nodes["altitude"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    # =====================================================
    # NORMALIZED FEATURES
    # =====================================================
    X = (
        df_nodes[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    print(
        f"   → Node feature shape: "
        f"{X.shape}"
    )

    # =====================================================
    # NODE ID MAPPING
    # =====================================================
    node_ids = (
        df_nodes[NODE_ID_COL]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .tolist()
    )

    node_mapping = {

        node_id: idx

        for idx, node_id

        in enumerate(node_ids)
    }

    # =====================================================
    # EXISTING TOPOLOGY
    # =====================================================
    print_step(
        "Preparing existing topology..."
    )

    edges = []

    skipped_arcs = 0

    for _, row in df_arcs.iterrows():

        source_id = (
            str(row[SOURCE_COL])
            .strip()
            .replace(".0", "")
        )

        target_id = (
            str(row[TARGET_COL])
            .strip()
            .replace(".0", "")
        )

        if (
            source_id not in node_mapping
            or
            target_id not in node_mapping
        ):

            skipped_arcs += 1

            continue

        source_idx = node_mapping[source_id]

        target_idx = node_mapping[target_id]

        # ================================================
        # REMOVE SELF LOOPS
        # ================================================
        if source_idx == target_idx:

            continue

        edges.append(
            [source_idx, target_idx]
        )

    print(
        f"   → Existing topology edges: "
        f"{len(edges)}"
    )

    print(
        f"   → Skipped invalid arcs: "
        f"{skipped_arcs}"
    )

    # =====================================================
    # SPATIAL RECOVERY
    # IMPORTANT:
    # ONLY FOR ISOLATED NODES
    # =====================================================
    if (
        ENABLE_SPATIAL_RECONSTRUCTION
        and
        len(edges) > 0
    ):

        print_step(
            "Running isolated-node reconstruction..."
        )

        # ================================================
        # CONNECTED NODES
        # ================================================
        connected_nodes = set()

        for s, t in edges:

            connected_nodes.add(s)
            connected_nodes.add(t)

        # ================================================
        # ISOLATED NODES
        # ================================================
        isolated_nodes = [

            idx

            for idx in range(len(node_ids))

            if idx not in connected_nodes
        ]

        print(
            f"   → Isolated nodes detected: "
            f"{len(isolated_nodes)}"
        )

        # ================================================
        # SPATIAL COORDINATES
        # ================================================
        coordinates = np.column_stack([

            raw_coordx,
            raw_coordy
        ])

        nbrs = NearestNeighbors(

            n_neighbors=K_NEIGHBORS + 1,
            algorithm="ball_tree"

        ).fit(coordinates)

        distances, indices = nbrs.kneighbors(
            coordinates
        )

        reconstructed_edges = 0

        edge_set = set()

        for s, t in edges:

            edge_set.add((s, t))
            edge_set.add((t, s))

        # ================================================
        # ONLY ISOLATED NODE RECOVERY
        # ================================================
        for node_idx in isolated_nodes:

            neighbors = indices[node_idx][1:]
            neighbor_distances = distances[node_idx][1:]

            added = False

            for neighbor_idx, dist in zip(
                neighbors,
                neighbor_distances
            ):

                # ========================================
                # DISTANCE CONSTRAINT
                # ========================================
                if dist > MAX_EDGE_DISTANCE:

                    continue

                # ========================================
                # HYDRAULIC CONSTRAINT
                # ========================================
                elev1 = raw_altitude[node_idx]
                elev2 = raw_altitude[neighbor_idx]

                slope = abs(
                    elev1 - elev2
                ) / (dist + 1e-8)

                # realistic drainage slope
                if slope > 0.10:

                    continue

                # ========================================
                # DOWNHILL FLOW
                # ========================================
                if elev1 >= elev2:

                    edge = (
                        node_idx,
                        neighbor_idx
                    )

                else:

                    edge = (
                        neighbor_idx,
                        node_idx
                    )

                # ========================================
                # AVOID DUPLICATES
                # ========================================
                if edge in edge_set:

                    continue

                edges.append(
                    [edge[0], edge[1]]
                )

                edge_set.add(edge)

                reconstructed_edges += 1

                added = True

                break

            if not added:

                continue

        print(
            f"   → Reconstructed edges: "
            f"{reconstructed_edges}"
        )

    # =====================================================
    # SAFETY
    # =====================================================
    if len(edges) == 0:

        raise ValueError(
            "No valid graph edges created."
        )

    # =====================================================
    # EDGE INDEX
    # =====================================================
    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    print(
        f"   → Final graph edges: "
        f"{edge_index.shape[1]}"
    )

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================
    print_step(
        "Splitting edges..."
    )

    num_edges = edge_index.shape[1]

    perm = torch.randperm(num_edges)

    test_size = int(
        TEST_EDGE_RATIO * num_edges
    )

    test_idx = perm[:test_size]

    train_idx = perm[test_size:]

    train_edge_index = (
        edge_index[:, train_idx]
    )

    test_edge_index = (
        edge_index[:, test_idx]
    )

    print(
        f"   → Train edges: "
        f"{train_edge_index.shape[1]}"
    )

    print(
        f"   → Test edges: "
        f"{test_edge_index.shape[1]}"
    )

    # =====================================================
    # MASKS
    # =====================================================
    print_step(
        "Creating masks..."
    )

    train_mask = torch.zeros(
        num_edges,
        dtype=torch.bool
    )

    test_mask = torch.zeros(
        num_edges,
        dtype=torch.bool
    )

    train_mask[train_idx] = True

    test_mask[test_idx] = True

    # =====================================================
    # BUILD DATA OBJECT
    # =====================================================
    print_step(
        "Converting to tensors..."
    )

    graph_data = Data(

    x=torch.tensor(
        X,
        dtype=torch.float
    ),

    edge_index=edge_index,

    train_edge_index=train_edge_index,

    test_edge_index=test_edge_index,

    train_edges=train_edge_index,

    test_edges=test_edge_index,

    train_mask=train_mask,

    test_mask=test_mask
    )

    # =====================================================
    # SAVE RAW GIS ATTRIBUTES
    # =====================================================
    graph_data.raw_coordx = torch.tensor(
        raw_coordx,
        dtype=torch.float
    )

    graph_data.raw_coordy = torch.tensor(
        raw_coordy,
        dtype=torch.float
    )

    graph_data.raw_altitude = torch.tensor(
        raw_altitude,
        dtype=torch.float
    )

    # =====================================================
    # FINAL STATS
    # =====================================================
    connected_nodes = torch.unique(
        edge_index
    ).shape[0]

    isolated_nodes = (
        graph_data.x.shape[0]
        - connected_nodes
    )

    avg_degree = (
        2 * edge_index.shape[1]
    ) / graph_data.x.shape[0]

    print(
        f"   → Final isolated nodes: "
        f"{isolated_nodes}"
    )

    print_step(
        "Graph constructed"
    )

    print(
        f"Number of nodes: "
        f"{graph_data.x.shape[0]}"
    )

    print(
        f"Number of edges: "
        f"{edge_index.shape[1]}"
    )

    print(
        f"Connected nodes: "
        f"{connected_nodes}"
    )

    print(
        f"Isolated nodes: "
        f"{isolated_nodes}"
    )

    print(
        f"Average degree: "
        f"{avg_degree:.2f}"
    )

    return graph_data