import os
import torch

from config import (
    DATA_PATH,
    OUTPUT_PATH,
    DEVICE
)

from visualization import (
    plot_training_curve,
    plot_graph_topology,
    plot_embeddings,
    plot_anomaly_map,
    plot_edge_recovery
)

from preprocessing import (
    load_data,
    clean_data
)

from graph_builder import (
    build_graph
)

from train import (
    train_model
)

from evaluation import (
    evaluate_model
)

from topology_learning import (
    reconstruct_topology_with_gnn,
    topology_statistics,
    edge_dataframe
)

from export_results import (
    export_clean_nodes,
    export_clean_arcs,
    export_clean_basins
)

from export_gis import (
    export_nodes_shp,
    export_arcs_shp,
    export_basins_shp
)

from utils import (
    set_seed,
    print_section,
    print_step
)


# =========================================================
# REPRODUCIBILITY
# =========================================================
set_seed(42)


# =========================================================
# SAFE FILE CHECK
# =========================================================
def check_file_exists(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


# =========================================================
# SAVE RESULTS
# =========================================================
def save_results(results):

    results_path = os.path.join(
        OUTPUT_PATH,
        "results.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        for key, value in results.items():

            f.write(
                f"{key}: {value}\n"
            )

    print(
        f" Results saved to: "
        f"{results_path}"
    )


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model):

    model_path = os.path.join(
        OUTPUT_PATH,
        "gnn_model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f" Model saved to: "
        f"{model_path}"
    )


# =========================================================
# SAVE FILTERED EDGES
# =========================================================
def save_filtered_edges(
    graph_data
):

    edge_df = edge_dataframe(
        graph_data
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "filtered_edges.csv"
    )

    edge_df.to_csv(
        save_path,
        index=False
    )

    print(
        f" Filtered edges saved: "
        f"{save_path}"
    )

    return edge_df


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():

    print_section(
        "Drainage Data Quality Pipeline"
    )

    print(
        f"Using device: {DEVICE}\n"
    )

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================
    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    # =====================================================
    # DATA FILES
    # =====================================================
    nodes_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialNoeuds.csv"
    )

    arcs_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialLineaire.csv"
    )

    basins_path = os.path.join(
        DATA_PATH,
        "MMM_MMM_ReseauPluvialBassins.csv"
    )

    # =====================================================
    # CHECK FILES
    # =====================================================
    check_file_exists(nodes_path)

    check_file_exists(arcs_path)

    check_file_exists(basins_path)

    # =====================================================
    # 1. LOAD DATA
    # =====================================================
    print_section(
        "1. Loading Data"
    )

    df_nodes, df_arcs, df_basins = (
        load_data(
            nodes_path,
            arcs_path,
            basins_path
        )
    )

    # =====================================================
    # KEEP ORIGINAL GIS TABLES
    # =====================================================
    df_nodes_gis = df_nodes.copy()

    df_arcs_gis = df_arcs.copy()

    print_step(
        "Data loaded successfully"
    )

    print(
        f"Nodes: {len(df_nodes)}"
    )

    print(
        f"Arcs: {len(df_arcs)}"
    )

    print(
        f"Basins: {len(df_basins)}\n"
    )

    # =====================================================
    # 2. CLEAN DATA
    # =====================================================
    print_section(
        "2. Cleaning Data"
    )

    df_nodes, df_arcs, df_basins = (
        clean_data(
            df_nodes,
            df_arcs,
            df_basins
        )
    )

    print_step(
        "Data cleaning completed"
    )

    print(
        f"Clean nodes: {len(df_nodes)}"
    )

    print(
        f"Clean arcs: {len(df_arcs)}"
    )

    # =====================================================
    # 3. BUILD GRAPH
    # =====================================================
    print_section(
        "3. Building Graph"
    )

    graph_data = build_graph(
        df_nodes,
        df_arcs,
        df_basins
    )

    # =====================================================
    # GRAPH STATS
    # =====================================================
    num_nodes = (
        graph_data.x.shape[0]
    )

    num_edges = (
        graph_data.edge_index.shape[1]
    )

    connected_nodes = torch.unique(
        graph_data.edge_index
    ).shape[0]

    isolated_nodes = (
        num_nodes
        - connected_nodes
    )

    avg_degree = (
        (2 * num_edges)
        / num_nodes
    )

    print_step(
        "Graph constructed"
    )

    print(
        f"Number of nodes: "
        f"{num_nodes}"
    )

    print(
        f"Number of edges: "
        f"{num_edges}"
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

    # =====================================================
    # 4. TRAIN MODEL
    # =====================================================
    print_section(
        "4. Training GNN Model"
    )

    model, training_logs = (
        train_model(
            graph_data
        )
    )

    print_step(
        "Training completed"
    )

    # =====================================================
    # 5. TOPOLOGY RECONSTRUCTION
    # =====================================================
    print_section(
        "5. Topology Reconstruction"
    )

    graph_data = (
        reconstruct_topology_with_gnn(
            model,
            graph_data,
            threshold=0.45,
            max_distance=80,
            max_slope=1.0
        )
    )

    topo_stats = (
        topology_statistics(
            graph_data
        )
    )

    for key, value in topo_stats.items():

        print(
            f"{key}: {value}"
        )

    # =====================================================
    # SAVE FILTERED EDGES
    # =====================================================
    save_filtered_edges(
        graph_data
    )

    # =====================================================
    # 6. MODEL EVALUATION
    # =====================================================
    print_section(
        "6. Model Evaluation"
    )

    results = evaluate_model(
        model,
        graph_data
    )

    # =====================================================
    # ADD TOPOLOGY STATS
    # =====================================================
    results.update(
        topo_stats
    )

    results["IsolatedNodes"] = int(
        isolated_nodes
    )

    results["AverageDegree"] = float(
        avg_degree
    )

    print_step(
        "Evaluation completed"
    )

    print(
        "\nEvaluation Metrics:\n"
    )

    for key, value in results.items():

        print(
            f"{key}: {value}"
        )

    # =====================================================
    # 7. SAVE RESULTS
    # =====================================================
    print_section(
        "7. Saving Results"
    )

    save_results(results)

    save_model(model)

    # =====================================================
    # 8. VISUALIZATION
    # =====================================================
    print_section(
        "8. Visualization"
    )

    plot_training_curve(
        training_logs
    )

    plot_graph_topology(
        graph_data
    )

    plot_embeddings(
        model,
        graph_data,
        DEVICE
    )

    plot_anomaly_map(
        model,
        graph_data,
        DEVICE
    )

    plot_edge_recovery(
        graph_data
    )

    print_step(
        "Visualizations completed"
    )

    # =====================================================
    # 9. EXPORT CSV
    # =====================================================
    print_section(
        "9. Exporting CSV Datasets"
    )

    clean_nodes_df = (
        export_clean_nodes(
            model,
            graph_data,
            df_nodes,
            DEVICE
        )
    )

    clean_arcs_df = (
        export_clean_arcs(
            graph_data,
            df_arcs
        )
    )

    clean_basins_df = (
        export_clean_basins(
            df_basins
        )
    )

    print_step(
        "CSV export completed"
    )

    # =====================================================
    # 10. EXPORT GIS
    # =====================================================
    print_section(
        "10. Exporting GIS Files"
    )

    try:

        export_nodes_shp(
            df_nodes_gis
        )

    except Exception as e:

        print(
            f" Node export failed: {e}"
        )

    try:

        export_arcs_shp(
            df_arcs_gis,
            df_nodes_gis
        )

    except Exception as e:

        print(
            f" Arc export failed: {e}"
        )

    try:

        export_basins_shp(
            clean_basins_df
        )

    except Exception as e:

        print(
            f" Basin export failed: {e}"
        )

    print_step(
        "GIS export completed"
    )

    # =====================================================
    # FINAL MESSAGE
    # =====================================================
    print_section(
        "Pipeline Completed Successfully"
    )

    print(
        "Generated outputs:"
    )

    output_files = [

        "results.txt",

        "gnn_model.pth",

        "training_curve.png",

        "graph_topology.png",

        "latent_embeddings.png",

        "anomaly_distribution.png",

        "edge_recovery.png",

        "filtered_edges.csv",

        "clean_nodes.csv",

        "clean_arcs.csv",

        "clean_basins.csv",

        "clean_nodes.shp",

        "clean_arcs.shp"
    ]

    for file_name in output_files:

        print(
            f"   → {file_name}"
        )


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":

    main()