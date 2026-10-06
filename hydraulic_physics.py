import torch
import torch.nn.functional as F

from config import (
    DEVICE,
    NODE_FEATURES,
    MIN_DEPTH,
    MIN_SLOPE
)


# =========================
# DEFAULT HYDRAULIC PARAMS
# =========================

# Manning roughness coefficient
DEFAULT_MANNING_N = 0.013

# Numerical stability
EPS = 1e-8


# =========================
# FEATURE INDEX HELPER
# =========================
def get_feature_index(
    feature_name,
    default_idx=0
):

    try:

        return NODE_FEATURES.index(
            feature_name
        )

    except Exception:

        return default_idx


# =========================
# SAFE FEATURE EXTRACTION
# =========================
def get_feature_column(
    x,
    idx,
    default_value=0.0
):

    # =========================
    # INVALID INDEX
    # =========================
    if idx >= x.shape[1]:

        return torch.full(
            (x.shape[0],),
            default_value,
            device=x.device
        )

    column = x[:, idx]

    # =========================
    # CLEAN NAN/INF
    # =========================
    column = torch.nan_to_num(
        column,
        nan=default_value,
        posinf=default_value,
        neginf=default_value
    )

    return column


# =========================
# COMPUTE PIPE AREA
# =========================
def compute_pipe_area(
    depth,
    width=None
):

    # =========================
    # DEFAULT WIDTH
    # =========================
    if width is None:

        width = torch.ones_like(
            depth
        )

    depth = torch.clamp(
        depth,
        min=EPS
    )

    width = torch.clamp(
        width,
        min=EPS
    )

    # =========================
    # RECTANGULAR APPROX
    # =========================
    area = width * depth

    area = torch.clamp(
        area,
        min=EPS
    )

    return area


# =========================
# HYDRAULIC RADIUS
# =========================
def compute_hydraulic_radius(
    depth,
    width=None
):

    if width is None:

        width = torch.ones_like(
            depth
        )

    area = compute_pipe_area(
        depth,
        width
    )

    wetted_perimeter = (
        width + 2 * depth
    )

    wetted_perimeter = torch.clamp(
        wetted_perimeter,
        min=EPS
    )

    radius = (
        area / wetted_perimeter
    )

    radius = torch.clamp(
        radius,
        min=EPS
    )

    return radius


# =========================
# MANNING FLOW
# =========================
def manning_flow(
    depth,
    slope,
    roughness=DEFAULT_MANNING_N,
    width=None
):

    # =========================
    # SAFE POSITIVE VALUES
    # =========================
    depth = torch.clamp(
        depth,
        min=EPS
    )

    slope = torch.clamp(
        slope,
        min=EPS
    )

    # =========================
    # AREA
    # =========================
    area = compute_pipe_area(
        depth,
        width
    )

    # =========================
    # HYDRAULIC RADIUS
    # =========================
    radius = compute_hydraulic_radius(
        depth,
        width
    )

    # =========================
    # MANNING EQUATION
    # =========================
    flow = (

        (1.0 / roughness)

        * area

        * torch.pow(
            radius,
            2.0 / 3.0
        )

        * torch.sqrt(slope)
    )

    # =========================
    # CLEAN NAN/INF
    # =========================
    flow = torch.nan_to_num(
        flow,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return flow


# =========================
# FLOW CONTINUITY LOSS
# =========================
def continuity_loss(
    edge_index,
    flow
):

    if edge_index.shape[1] == 0:

        return torch.tensor(
            0.0,
            device=flow.device
        )

    src = edge_index[0]

    tgt = edge_index[1]

    flow_diff = torch.abs(
        flow[src] - flow[tgt]
    )

    loss = torch.mean(
        flow_diff
    )

    return loss


# =========================
# DOWNSTREAM FLOW LOSS
# =========================
def downstream_loss(
    elevation,
    edge_index
):

    if edge_index.shape[1] == 0:

        return torch.tensor(
            0.0,
            device=elevation.device
        )

    src = edge_index[0]

    tgt = edge_index[1]

    elev_src = elevation[src]

    elev_tgt = elevation[tgt]

    # =========================
    # PENALIZE UPHILL FLOW
    # =========================
    uphill = F.relu(
        elev_tgt - elev_src
    )

    loss = torch.mean(
        uphill
    )

    return loss


# =========================
# ENERGY CONSERVATION LOSS
# =========================
def energy_loss(
    flow
):

    if flow.shape[0] < 2:

        return torch.tensor(
            0.0,
            device=flow.device
        )

    flow_variation = torch.abs(
        flow[1:] - flow[:-1]
    )

    return torch.mean(
        flow_variation
    )


# =========================
# HYDRAULIC PHYSICS LOSS
# =========================
def hydraulic_physics_loss(
    x_pred,
    edge_index
):

    try:

        # =====================
        # FEATURE INDICES
        # =====================
        elevation_idx = get_feature_index(
            "altitude",
            2
        )

        depth_idx = get_feature_index(
            "profondeur",
            3
        )

        # NOTE:
        # slope is NOT node feature
        # using radier as approximation
        slope_idx = get_feature_index(
            "radier",
            4
        )

        # =====================
        # EXTRACT FEATURES
        # =====================
        elevation = get_feature_column(
            x_pred,
            elevation_idx
        )

        depth = get_feature_column(
            x_pred,
            depth_idx
        )

        slope = get_feature_column(
            x_pred,
            slope_idx
        )

        # =====================
        # SAFE POSITIVE SLOPE
        # =====================
        slope = torch.abs(
            slope
        )

        # =====================
        # BASIC CONSTRAINTS
        # =====================
        depth_loss = torch.mean(
            F.relu(
                MIN_DEPTH - depth
            )
        )

        slope_loss = torch.mean(
            F.relu(
                MIN_SLOPE - slope
            )
        )

        # =====================
        # MANNING FLOW
        # =====================
        flow = manning_flow(
            depth,
            slope
        )

        # =====================
        # CONTINUITY
        # =====================
        cont_loss = continuity_loss(
            edge_index,
            flow
        )

        # =====================
        # DOWNSTREAM
        # =====================
        down_loss = downstream_loss(
            elevation,
            edge_index
        )

        # =====================
        # ENERGY
        # =====================
        eng_loss = energy_loss(
            flow
        )

        # =====================
        # TOTAL LOSS
        # =====================
        total_loss = (

            1.0 * depth_loss

            + 1.0 * slope_loss

            + 0.5 * cont_loss

            + 0.5 * down_loss

            + 0.2 * eng_loss
        )

        # =====================
        # FINAL SAFETY
        # =====================
        total_loss = torch.nan_to_num(
            total_loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return total_loss

    except Exception as e:

        print(
            f"   → Hydraulic loss error: {e}"
        )

        return torch.tensor(
            0.0,
            device=DEVICE
        )