import torch
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

from config import DEVICE


# =========================================================
# MAIN EVALUATION
# =========================================================
def evaluate_model(
    model,
    graph_data
):

    print(
        "   → Running evaluation..."
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
    # FORWARD PASS
    # =====================================================
    with torch.no_grad():

        reconstructed_x, embeddings = model(
            x,
            edge_index
        )

    # =====================================================
    # NODE RECONSTRUCTION METRICS
    # =====================================================
    rmse = compute_rmse(
        reconstructed_x,
        x
    )

    mae = compute_mae(
        reconstructed_x,
        x
    )

    # =====================================================
    # ANOMALY SCORE
    # =====================================================
    anomaly_score = compute_anomaly_score(
        reconstructed_x,
        x
    )

    # =====================================================
    # CONNECTIVITY
    # =====================================================
    connectivity = compute_connectivity(
        graph_data
    )

    isolated_nodes = compute_isolated_nodes(
        graph_data
    )

    average_degree = compute_average_degree(
        graph_data
    )

    # =====================================================
    # EDGE PREDICTION
    # =====================================================
    print(
        "   → Evaluating edge prediction..."
    )

    edge_metrics = evaluate_edge_prediction(
        model,
        embeddings,
        graph_data
    )

    # =====================================================
    # TOPOLOGY ACCURACY
    # =====================================================
    topology_accuracy = (
        edge_metrics["Edge_F1"]
    )

    # =====================================================
    # RESULTS
    # =====================================================
    results = {

        "RMSE":
        rmse,

        "MAE":
        mae,

        "AnomalyScore":
        anomaly_score,

        "Connectivity":
        connectivity,

        "IsolatedNodes":
        isolated_nodes,

        "AverageDegree":
        average_degree,

        "Edge_AUC":
        edge_metrics["Edge_AUC"],

        "Edge_F1":
        edge_metrics["Edge_F1"],

        "Edge_Precision":
        edge_metrics["Edge_Precision"],

        "Edge_Recall":
        edge_metrics["Edge_Recall"],

        "Topology_Accuracy":
        topology_accuracy
    }

    # =====================================================
    # TOPOLOGY STATISTICS
    # =====================================================
    if hasattr(
        graph_data,
        "filtered_edge_index"
    ):

        original_edges = (
            graph_data.edge_index.shape[1]
        )

        filtered_edges = (
            graph_data.filtered_edge_index.shape[1]
        )

        removed_edges = (
            original_edges
            - filtered_edges
        )

        reduction_ratio = (
            removed_edges
            / original_edges
        )

        results.update({

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
        })

    print(
        "   → Evaluation completed"
    )

    return results


# =========================================================
# RMSE
# =========================================================
def compute_rmse(
    reconstructed_x,
    original_x
):

    mse = torch.mean(
        (
            reconstructed_x
            - original_x
        ) ** 2
    )

    rmse = torch.sqrt(
        mse
    )

    return float(
        rmse.item()
    )


# =========================================================
# MAE
# =========================================================
def compute_mae(
    reconstructed_x,
    original_x
):

    mae = torch.mean(
        torch.abs(
            reconstructed_x
            - original_x
        )
    )

    return float(
        mae.item()
    )


# =========================================================
# ANOMALY SCORE
# =========================================================
def compute_anomaly_score(
    reconstructed_x,
    original_x
):

    reconstruction_error = torch.mean(

        torch.abs(
            reconstructed_x
            - original_x
        ),

        dim=1
    )

    threshold = torch.quantile(
        reconstruction_error,
        0.95
    )

    anomalies = (
        reconstruction_error
        > threshold
    ).sum()

    return int(
        anomalies.item()
    )


# =========================================================
# CONNECTIVITY
# =========================================================
def compute_connectivity(
    graph_data
):

    edge_index = (
        graph_data.edge_index
        .cpu()
        .numpy()
    )

    num_nodes = (
        graph_data.num_nodes
    )

    connected_nodes = set(
        edge_index[0]
    ).union(
        set(edge_index[1])
    )

    connectivity = (
        len(connected_nodes)
        / num_nodes
    )

    return float(
        connectivity
    )


# =========================================================
# ISOLATED NODES
# =========================================================
def compute_isolated_nodes(
    graph_data
):

    edge_index = (
        graph_data.edge_index
        .cpu()
        .numpy()
    )

    num_nodes = (
        graph_data.num_nodes
    )

    connected_nodes = set(
        edge_index[0]
    ).union(
        set(edge_index[1])
    )

    isolated_nodes = (
        num_nodes
        - len(connected_nodes)
    )

    return int(
        isolated_nodes
    )


# =========================================================
# AVERAGE DEGREE
# =========================================================
def compute_average_degree(
    graph_data
):

    edge_index = (
        graph_data.edge_index
        .cpu()
        .numpy()
    )

    num_nodes = (
        graph_data.num_nodes
    )

    degree = np.zeros(
        num_nodes
    )

    for src, tgt in zip(
        edge_index[0],
        edge_index[1]
    ):

        degree[src] += 1

        degree[tgt] += 1

    return float(
        np.mean(degree)
    )


# =========================================================
# EDGE PREDICTION EVALUATION
# =========================================================
def evaluate_edge_prediction(
    model,
    embeddings,
    graph_data
):

    train_edges = (
        graph_data.train_edges
        .to(DEVICE)
    )

    test_edges = (
        graph_data.test_edges
        .to(DEVICE)
    )

    # =====================================================
    # POSITIVE EDGES
    # =====================================================
    positive_edges = test_edges

    # =====================================================
    # NEGATIVE EDGES
    # =====================================================
    negative_edges = generate_negative_edges(
        graph_data,
        positive_edges.shape[1]
    ).to(DEVICE)

    # =====================================================
    # POSITIVE SCORES
    # =====================================================
    pos_scores = model.predict_edges(
        embeddings,
        positive_edges[0],
        positive_edges[1]
    )

    pos_probs = torch.sigmoid(
        pos_scores
    )

    # =====================================================
    # NEGATIVE SCORES
    # =====================================================
    neg_scores = model.predict_edges(
        embeddings,
        negative_edges[0],
        negative_edges[1]
    )

    neg_probs = torch.sigmoid(
        neg_scores
    )

    # =====================================================
    # CONCATENATE
    # =====================================================
    y_true = np.concatenate([

        np.ones(
            pos_probs.shape[0]
        ),

        np.zeros(
            neg_probs.shape[0]
        )
    ])

    y_scores = np.concatenate([

        pos_probs.detach()
        .cpu()
        .numpy(),

        neg_probs.detach()
        .cpu()
        .numpy()
    ])

    # =====================================================
    # BINARY PREDICTIONS
    # =====================================================
    y_pred = (
        y_scores > 0.5
    ).astype(int)

    # =====================================================
    # METRICS
    # =====================================================
    auc = roc_auc_score(
        y_true,
        y_scores
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    return {

        "Edge_AUC":
        float(auc),

        "Edge_F1":
        float(f1),

        "Edge_Precision":
        float(precision),

        "Edge_Recall":
        float(recall)
    }


# =========================================================
# NEGATIVE EDGE SAMPLING
# =========================================================
def generate_negative_edges(
    graph_data,
    num_samples
):

    num_nodes = (
        graph_data.num_nodes
    )

    edge_index = (
        graph_data.edge_index
        .cpu()
        .numpy()
    )

    existing_edges = set()

    for s, t in zip(
        edge_index[0],
        edge_index[1]
    ):

        existing_edges.add(
            (int(s), int(t))
        )

        existing_edges.add(
            (int(t), int(s))
        )

    negative_edges = []

    while len(
        negative_edges
    ) < num_samples:

        s = np.random.randint(
            0,
            num_nodes
        )

        t = np.random.randint(
            0,
            num_nodes
        )

        if s == t:

            continue

        if (
            s,
            t
        ) in existing_edges:

            continue

        negative_edges.append(
            [s, t]
        )

    negative_edges = torch.tensor(
        negative_edges,
        dtype=torch.long
    ).T

    return negative_edges


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================
def evaluate_edges(
    model,
    embeddings,
    graph_data
):

    return evaluate_edge_prediction(
        model,
        embeddings,
        graph_data
    )