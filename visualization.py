import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import PCA

from config import OUTPUT_PATH


# =========================
# TRAINING CURVE
# =========================
def plot_training_curve(
    training_logs
):

    if len(training_logs) == 0:

        print(
            " No training logs found."
        )

        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        training_logs,
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training Loss Curve"
    )

    plt.grid(True)

    save_path = os.path.join(
        OUTPUT_PATH,
        "training_curve.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f" Training curve saved: "
        f"{save_path}"
    )


# =========================
# GRAPH TOPOLOGY
# =========================
def plot_graph_topology(
    graph_data,
    max_nodes=3000
):

    print(
        "   → Plotting graph topology..."
    )

    edge_index = (
        graph_data.edge_index
        .cpu()
        .numpy()
    )

    num_nodes = (
        graph_data.x.shape[0]
    )

    # =========================
    # SUBGRAPH FOR VISUALIZATION
    # =========================
    if num_nodes > max_nodes:

        print(
            f"   → Large graph detected "
            f"({num_nodes} nodes)"
        )

        print(
            f"   → Visualizing first "
            f"{max_nodes} nodes only"
        )

        valid_mask = (
            (edge_index[0] < max_nodes)
            &
            (edge_index[1] < max_nodes)
        )

        edge_index = edge_index[
            :,
            valid_mask
        ]

        num_nodes = max_nodes

    # =========================
    # CREATE GRAPH
    # =========================
    G = nx.Graph()

    G.add_nodes_from(
        range(num_nodes)
    )

    edges = edge_index.T.tolist()

    G.add_edges_from(edges)

    # =========================
    # CONNECTIVITY STATS
    # =========================
    isolated_nodes = list(
        nx.isolates(G)
    )

    connected_components = list(
        nx.connected_components(G)
    )

    largest_component = max(
        connected_components,
        key=len
    )

    print(
        f"   → Nodes in visualization: "
        f"{num_nodes}"
    )

    print(
        f"   → Edges in visualization: "
        f"{len(edges)}"
    )

    print(
        f"   → Isolated nodes: "
        f"{len(isolated_nodes)}"
    )

    print(
        f"   → Connected components: "
        f"{len(connected_components)}"
    )

    print(
        f"   → Largest component size: "
        f"{len(largest_component)}"
    )

    # =========================
    # LAYOUT
    # =========================
    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=0.15
    )

    # =========================
    # DRAW CONNECTED NODES
    # =========================
    connected_nodes = list(
        set(G.nodes())
        - set(isolated_nodes)
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=connected_nodes,
        node_size=6,
        alpha=0.7
    )

    # =========================
    # DRAW ISOLATED NODES
    # =========================
    if len(isolated_nodes) > 0:

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=isolated_nodes,
            node_size=10,
            alpha=0.8
        )

    # =========================
    # DRAW EDGES
    # =========================
    nx.draw_networkx_edges(
        G,
        pos,
        width=0.3,
        alpha=0.4
    )

    plt.title(
        "Drainage Network Topology"
    )

    plt.axis("off")

    save_path = os.path.join(
        OUTPUT_PATH,
        "graph_topology.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f" Graph topology saved: "
        f"{save_path}"
    )


# =========================
# LATENT EMBEDDINGS
# =========================
def plot_embeddings(
    model,
    graph_data,
    device
):

    print(
        "   → Plotting embeddings..."
    )

    model.eval()

    x = graph_data.x.to(device)

    edge_index = (
        graph_data.edge_index
        .to(device)
    )

    with torch.no_grad():

        _, embeddings = model(
            x,
            edge_index
        )

    embeddings = (
        embeddings
        .cpu()
        .numpy()
    )

    # =========================
    # NAN SAFETY
    # =========================
    embeddings = np.nan_to_num(
        embeddings,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =========================
    # PCA REDUCTION
    # =========================
    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        embeddings
    )

    explained = (
        pca.explained_variance_ratio_
    )

    print(
        f"   → PCA explained variance: "
        f"{explained.sum():.3f}"
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=3,
        alpha=0.6
    )

    plt.title(
        "Latent Node Embeddings"
    )

    plt.xlabel(
        f"PCA 1 "
        f"({explained[0]:.2%})"
    )

    plt.ylabel(
        f"PCA 2 "
        f"({explained[1]:.2%})"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "latent_embeddings.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f" Embeddings saved: "
        f"{save_path}"
    )


# =========================
# ANOMALY MAP
# =========================
def plot_anomaly_map(
    model,
    graph_data,
    device,
    threshold=None
):

    print(
        "   → Plotting anomaly map..."
    )

    model.eval()

    x = graph_data.x.to(device)

    edge_index = (
        graph_data.edge_index
        .to(device)
    )

    with torch.no_grad():

        x_pred, _ = model(
            x,
            edge_index
        )

    # =========================
    # RECONSTRUCTION ERROR
    # =========================
    errors = torch.mean(

        torch.abs(
            x_pred - x
        ),

        dim=1
    )

    errors = (
        errors.cpu()
        .numpy()
    )

    errors = np.nan_to_num(
        errors,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =========================
    # ADAPTIVE THRESHOLD
    # =========================
    if threshold is None:

        threshold = (
            np.mean(errors)
            + 2 * np.std(errors)
        )

    anomalies = (
        errors > threshold
    )

    anomaly_ratio = (
        anomalies.sum()
        / len(errors)
    )

    print(
        f"   → Threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"   → Anomalies: "
        f"{anomalies.sum()}"
    )

    print(
        f"   → Anomaly ratio: "
        f"{anomaly_ratio:.2%}"
    )

    # =========================
    # HISTOGRAM
    # =========================
    plt.figure(figsize=(10, 6))

    plt.hist(
        errors,
        bins=50
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2
    )

    plt.title(
        "Anomaly Distribution"
    )

    plt.xlabel(
        "Reconstruction Error"
    )

    plt.ylabel(
        "Frequency"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "anomaly_distribution.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f" Anomaly map saved: "
        f"{save_path}"
    )


# =========================
# EDGE RECOVERY
# =========================
def plot_edge_recovery(
    graph_data
):

    print(
        "   → Plotting edge recovery..."
    )

    train_edges = (
        graph_data.train_edges
        .shape[1]
    )

    test_edges = (
        graph_data.test_edges
        .shape[1]
    )

    total_edges = (
        graph_data.edge_index
        .shape[1]
    )

    # =========================
    # FILTERED EDGES
    # =========================
    if hasattr(
        graph_data,
        "filtered_edge_index"
    ):

        filtered_edges = (
            graph_data
            .filtered_edge_index
            .shape[1]
        )

    else:

        filtered_edges = total_edges

    labels = [

        "Train",

        "Test",

        "Original",

        "Filtered"
    ]

    values = [

        train_edges,

        test_edges,

        total_edges,

        filtered_edges
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Edge Recovery Statistics"
    )

    plt.ylabel(
        "Number of Edges"
    )

    save_path = os.path.join(
        OUTPUT_PATH,
        "edge_recovery.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f" Edge recovery figure saved: "
        f"{save_path}"
    )