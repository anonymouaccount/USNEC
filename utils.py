import random
import numpy as np
import torch


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # =========================
    # DETERMINISTIC MODE
    # =========================
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(
        f"   → Random seed set to {seed}"
    )


# =========================
# STANDARD NORMALIZATION
# =========================
def normalize_features(X):

    """
    Standard normalization:
    mean = 0
    std = 1
    """

    X = np.asarray(
        X,
        dtype=np.float32
    )

    mean = np.nanmean(
        X,
        axis=0
    )

    std = np.nanstd(
        X,
        axis=0
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    X_norm = (
        X - mean
    ) / std

    X_norm = np.nan_to_num(
        X_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X_norm


# =========================
# MIN-MAX SCALING
# =========================
def min_max_scaling(X):

    """
    Scale features between 0 and 1
    """

    X = np.asarray(
        X,
        dtype=np.float32
    )

    min_val = np.nanmin(
        X,
        axis=0
    )

    max_val = np.nanmax(
        X,
        axis=0
    )

    denominator = (
        max_val - min_val
    )

    denominator = np.where(
        denominator < 1e-8,
        1.0,
        denominator
    )

    X_scaled = (
        X - min_val
    ) / denominator

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X_scaled


# =========================
# EUCLIDEAN DISTANCE
# =========================
def euclidean_distance(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    return np.sqrt(
        np.sum(
            (a - b) ** 2
        )
    )


# =========================
# PAIRWISE DISTANCES
# =========================
def pairwise_distances(coords):

    """
    Compute full pairwise
    Euclidean distance matrix
    """

    coords = np.asarray(
        coords,
        dtype=np.float32
    )

    n = coords.shape[0]

    dist_matrix = np.zeros(
        (n, n),
        dtype=np.float32
    )

    for i in range(n):

        for j in range(i, n):

            dist = euclidean_distance(

                coords[i],

                coords[j]
            )

            dist_matrix[i, j] = dist

            dist_matrix[j, i] = dist

    return dist_matrix


# =========================
# LOGGING UTILITIES
# =========================
def print_section(title):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f" {title}"
    )

    print(
        "=" * 60
        + "\n"
    )


def print_step(message):

    print(
        f"   → {message}"
    )


def print_warning(message):

    print(
        f"   ⚠️ {message}"
    )


def print_success(message):

    print(
        f"   ✅ {message}"
    )


# =========================
# MASK UTILITIES
# =========================
def create_missing_mask(X):

    """
    1 = observed
    0 = missing
    """

    X = np.asarray(X)

    mask = (
        ~np.isnan(X)
    ).astype(np.float32)

    return mask


def apply_mask_loss(
    diff,
    mask
):

    """
    Apply mask safely
    to loss computation
    """

    diff = np.nan_to_num(
        diff,
        nan=0.0
    )

    mask = np.nan_to_num(
        mask,
        nan=0.0
    )

    denominator = (
        mask.sum() + 1e-8
    )

    return (
        (diff * mask).sum()
        / denominator
    )


# =========================
# GRAPH UTILITIES
# =========================
def make_bidirectional(
    edge_index
):

    """
    Convert directed graph
    into bidirectional graph
    """

    edge_index = np.asarray(
        edge_index,
        dtype=np.int64
    )

    src = edge_index[0]

    tgt = edge_index[1]

    reversed_edges = np.vstack(
        (tgt, src)
    )

    bidirectional = np.hstack(
        (
            edge_index,
            reversed_edges
        )
    )

    # =========================
    # REMOVE DUPLICATES
    # =========================
    bidirectional = np.unique(
        bidirectional,
        axis=1
    )

    return bidirectional


# =========================
# REMOVE SELF LOOPS
# =========================
def remove_self_loops(
    edge_index
):

    edge_index = np.asarray(
        edge_index,
        dtype=np.int64
    )

    src = edge_index[0]

    tgt = edge_index[1]

    mask = src != tgt

    cleaned = edge_index[
        :,
        mask
    ]

    return cleaned


# =========================
# CONNECTED NODE RATIO
# =========================
def connected_node_ratio(
    edge_index,
    num_nodes
):

    edge_index = np.asarray(
        edge_index
    )

    connected_nodes = np.unique(
        edge_index
    )

    ratio = (
        len(connected_nodes)
        / max(num_nodes, 1)
    )

    return float(ratio)


# =========================
# DEBUG HELPERS
# =========================
def check_nan(
    X,
    name="Tensor"
):

    X = np.asarray(X)

    if np.isnan(X).any():

        print_warning(
            f"{name} contains NaN values"
        )

    else:

        print_success(
            f"{name} has no NaN values"
        )


def check_inf(
    X,
    name="Tensor"
):

    X = np.asarray(X)

    if np.isinf(X).any():

        print_warning(
            f"{name} contains Inf values"
        )

    else:

        print_success(
            f"{name} has no Inf values"
        )


def check_tensor(
    tensor,
    name="Tensor"
):

    if isinstance(
        tensor,
        torch.Tensor
    ):

        tensor_cpu = (
            tensor.detach()
            .cpu()
        )

        print(
            f"{name} shape: "
            f"{tuple(tensor_cpu.shape)}"
        )

        print(
            f"{name} min: "
            f"{tensor_cpu.min().item():.4f}"
        )

        print(
            f"{name} max: "
            f"{tensor_cpu.max().item():.4f}"
        )

        print(
            f"{name} mean: "
            f"{tensor_cpu.mean().item():.4f}"
        )

        print(
            f"{name} std: "
            f"{tensor_cpu.std().item():.4f}"
        )

    else:

        arr = np.asarray(tensor)

        print(
            f"{name} shape: "
            f"{arr.shape}"
        )

        print(
            f"{name} min: "
            f"{np.min(arr):.4f}"
        )

        print(
            f"{name} max: "
            f"{np.max(arr):.4f}"
        )


# =========================
# SAFE TORCH CONVERSION
# =========================
def to_tensor(
    X,
    dtype=torch.float
):

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return torch.tensor(
        X,
        dtype=dtype
    )