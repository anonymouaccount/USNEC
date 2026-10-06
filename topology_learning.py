import torch
import numpy as np
import pandas as pd

from config import DEVICE


# =========================
# TOPOLOGY RECONSTRUCTION
# =========================
def reconstruct_topology_with_gnn(
    model,
    graph_data,
    threshold=0.45,
    max_distance=80,
    max_slope=0.10
):

    print(
        "   → Running AI-driven topology reconstruction..."
    )

    model.eval()

    # =====================================================
    # INPUTS
    # =====================================================
    x = graph_data.x.to(DEVICE)

    edge_index = (
        graph_data.edge_index
        .to(DEVICE)
    )

    # =====================================================
    # RAW GIS VALUES
    # IMPORTANT:
    # USE REAL GIS VALUES
    # NOT NORMALIZED FEATURES
    # =====================================================
    coordx = (
        graph_data.raw_coordx
        .detach()
        .cpu()
        .numpy()
    )

    coordy = (
        graph_data.raw_coordy
        .detach()
        .cpu()
        .numpy()
    )

    altitude = (
        graph_data.raw_altitude
        .detach()
        .cpu()
        .numpy()
    )

    # =====================================================
    # GNN EMBEDDINGS
    # =====================================================
    with torch.no_grad():

        _, embeddings = model(
            x,
            edge_index
        )

        # =================================================
        # EDGE PREDICTION
        # =================================================
        src = edge_index[0]

        tgt = edge_index[1]

        scores = model.predict_edges(
            embeddings,
            src,
            tgt
        )

        probs = torch.sigmoid(
            scores
        )

        probs = (
            probs
            .detach()
            .cpu()
            .numpy()
        )

    # =====================================================
    # CONVERT EDGE INDICES
    # =====================================================
    src = (
        src
        .detach()
        .cpu()
        .numpy()
    )

    tgt = (
        tgt
        .detach()
        .cpu()
        .numpy()
    )

    # =====================================================
    # EDGE FILTERING
    # =====================================================
    kept_edges = []

    kept_scores = []

    kept_distances = []

    kept_slopes = []

    for i in range(len(src)):

        s = src[i]

        t = tgt[i]

        probability = probs[i]

        # =================================================
        # GNN CONFIDENCE
        # =================================================
        if probability < threshold:

            continue

        # =================================================
        # REAL GIS DISTANCE
        # =================================================
        dx = coordx[s] - coordx[t]

        dy = coordy[s] - coordy[t]

        distance = np.sqrt(
            dx**2 + dy**2
        )

        # =================================================
        # INVALID DISTANCE
        # =================================================
        if distance <= 0:

            continue

        # =================================================
        # MAX DISTANCE FILTER
        # =================================================
        if distance > max_distance:

            continue

        # =================================================
        # REAL HYDRAULIC SLOPE
        # =================================================
        elevation_diff = (
            altitude[s] - altitude[t]
        )

        slope = abs(
            elevation_diff / distance
        )

        # =================================================
        # INVALID SLOPE
        # =================================================
        if np.isnan(slope):

            continue

        if np.isinf(slope):

            continue

        # =================================================
        # HARD SLOPE FILTER
        # =================================================
        if slope > max_slope:

            # =============================================
            # KEEP ONLY VERY CONFIDENT EDGES
            # =============================================
            if probability < 0.95:

                continue

        # =================================================
        # HYDRAULIC DIRECTION
        # FLOW FROM HIGH TO LOW
        # =================================================
        if altitude[s] < altitude[t]:

            s, t = t, s

        kept_edges.append(
            [s, t]
        )

        kept_scores.append(
            float(probability)
        )

        kept_distances.append(
            float(distance)
        )

        kept_slopes.append(
            float(slope)
        )

    # =====================================================
    # FALLBACK SAFETY
    # =====================================================
    if len(kept_edges) == 0:

        print(
            "   → WARNING: "
            "No edges survived filtering."
        )

        print(
            "   → Reverting to original topology."
        )

        graph_data.filtered_edge_index = (
            graph_data.edge_index
        )

        graph_data.edge_scores = torch.ones(
            graph_data.edge_index.shape[1]
        )

        graph_data.mean_edge_distance = 0.0

        graph_data.mean_edge_slope = 0.0

        return graph_data

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================
    unique_edges = []

    unique_scores = []

    unique_distances = []

    unique_slopes = []

    edge_set = set()

    for edge, score, dist, slope in zip(
        kept_edges,
        kept_scores,
        kept_distances,
        kept_slopes
    ):

        edge_tuple = (
            int(edge[0]),
            int(edge[1])
        )

        if edge_tuple not in edge_set:

            edge_set.add(
                edge_tuple
            )

            unique_edges.append(
                edge
            )

            unique_scores.append(
                score
            )

            unique_distances.append(
                dist
            )

            unique_slopes.append(
                slope
            )

    kept_edges = unique_edges

    kept_scores = unique_scores

    kept_distances = unique_distances

    kept_slopes = unique_slopes

    # =====================================================
    # FINAL EDGE INDEX
    # =====================================================
    filtered_edge_index = torch.tensor(
        kept_edges,
        dtype=torch.long
    ).T

    edge_scores = torch.tensor(
        kept_scores,
        dtype=torch.float
    )

    # =====================================================
    # STATISTICS
    # =====================================================
    initial_edges = int(
        edge_index.shape[1]
    )

    final_edges = int(
        filtered_edge_index.shape[1]
    )

    removed_edges = (
        initial_edges - final_edges
    )

    reduction_ratio = (
        removed_edges / initial_edges
    )

    mean_distance = float(
        np.mean(kept_distances)
    )

    mean_slope = float(
        np.mean(kept_slopes)
    )

    print(
        f"   → Initial edges: "
        f"{initial_edges}"
    )

    print(
        f"   → Filtered edges: "
        f"{final_edges}"
    )

    print(
        f"   → Removed edges: "
        f"{removed_edges}"
    )

    print(
        f"   → Reduction ratio: "
        f"{reduction_ratio:.4f}"
    )

    print(
        f"   → Threshold: "
        f"{threshold}"
    )

    print(
        f"   → Max distance: "
        f"{max_distance} m"
    )

    print(
        f"   → Max slope: "
        f"{max_slope}"
    )

    print(
        f"   → Mean kept distance: "
        f"{mean_distance:.2f} m"
    )

    print(
        f"   → Mean kept slope: "
        f"{mean_slope:.4f}"
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    graph_data.filtered_edge_index = (
        filtered_edge_index.cpu()
    )

    graph_data.edge_scores = (
        edge_scores.cpu()
    )

    graph_data.mean_edge_distance = (
        mean_distance
    )

    graph_data.mean_edge_slope = (
        mean_slope
    )

    return graph_data


# =========================
# EDGE DATAFRAME
# =========================
def edge_dataframe(
    graph_data
):

    if (
        not hasattr(
            graph_data,
            "filtered_edge_index"
        )
    ):

        raise ValueError(
            "Run topology reconstruction first."
        )

    edge_index = (
        graph_data.filtered_edge_index
        .detach()
        .cpu()
        .numpy()
    )

    edge_scores = (
        graph_data.edge_scores
        .detach()
        .cpu()
        .numpy()
    )

    df = pd.DataFrame({

        "source":
        edge_index[0],

        "target":
        edge_index[1],

        "confidence":
        edge_scores
    })

    return df


# =========================
# TOPOLOGY STATISTICS
# =========================
def topology_statistics(
    graph_data
):

    if (
        not hasattr(
            graph_data,
            "filtered_edge_index"
        )
    ):

        raise ValueError(
            "Run topology reconstruction first."
        )

    original_edges = (
        graph_data.edge_index
        .shape[1]
    )

    filtered_edges = (
        graph_data.filtered_edge_index
        .shape[1]
    )

    removed_edges = (
        original_edges
        - filtered_edges
    )

    reduction_ratio = (
        removed_edges
        / original_edges
    )

    stats = {

        "Original_Edges":
        int(original_edges),

        "Filtered_Edges":
        int(filtered_edges),

        "Removed_Edges":
        int(removed_edges),

        "Reduction_Ratio":
        float(reduction_ratio),

        "Mean_Edge_Distance":
        float(
            graph_data.mean_edge_distance
        ),

        "Mean_Edge_Slope":
        float(
            graph_data.mean_edge_slope
        )
    }

    return stats