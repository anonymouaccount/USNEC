import os
import torch
import pandas as pd
import numpy as np

from config import (
    OUTPUT_PATH,
    NODE_FEATURES
)


# =========================
# EXPORT CLEAN NODES
# =========================
def export_clean_nodes(
    model,
    graph_data,
    df_nodes,
    device
):

    print(
        "   → Exporting clean nodes..."
    )

    model.eval()

    # =========================
    # GRAPH DATA
    # =========================
    x = graph_data.x.to(device)

    edge_index = (
        graph_data.edge_index
        .to(device)
    )

    # =========================
    # FORWARD
    # =========================
    with torch.no_grad():

        x_pred, _ = model(
            x,
            edge_index
        )

    reconstructed = (
        x_pred.cpu().numpy()
    )

    # =========================
    # COPY ORIGINAL DF
    # =========================
    df_clean = df_nodes.copy()

    # =========================
    # SAFE FEATURE EXPORT
    # =========================
    num_features = min(
        len(NODE_FEATURES),
        reconstructed.shape[1]
    )

    for i in range(num_features):

        col = NODE_FEATURES[i]

        if col in df_clean.columns:

            try:

                original_dtype = (
                    df_clean[col].dtype
                )

                # =====================
                # KEEP NUMERIC ONLY
                # =====================
                if np.issubdtype(
                    original_dtype,
                    np.number
                ):

                    df_clean[col] = (
                        reconstructed[:, i]
                    )

            except Exception:

                continue

    # =========================
    # CLEAN INF/NAN
    # =========================
    df_clean = df_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================
    # SAVE CSV
    # =========================
    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_nodes.csv"
    )

    df_clean.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean nodes saved: "
        f"{save_path}"
    )

    return df_clean


# =========================
# EXPORT CLEAN ARCS
# =========================
def export_clean_arcs(
    graph_data,
    df_arcs
):

    print(
        "   → Exporting clean arcs..."
    )

    df_clean = df_arcs.copy()

    # =========================
    # REMOVE INF
    # =========================
    df_clean = df_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================
    # SAVE CSV
    # =========================
    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_arcs.csv"
    )

    df_clean.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean arcs saved: "
        f"{save_path}"
    )

    return df_clean


# =========================
# EXPORT CLEAN BASINS
# =========================
def export_clean_basins(
    df_basins
):

    print(
        "   → Exporting clean basins..."
    )

    df_clean = df_basins.copy()

    # =========================
    # REMOVE INF
    # =========================
    df_clean = df_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================
    # SAVE CSV
    # =========================
    save_path = os.path.join(
        OUTPUT_PATH,
        "clean_basins.csv"
    )

    df_clean.to_csv(
        save_path,
        index=False
    )

    print(
        f" Clean basins saved: "
        f"{save_path}"
    )

    return df_clean