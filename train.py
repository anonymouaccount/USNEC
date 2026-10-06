import torch
import torch.nn.functional as F
import torch.optim as optim

from model import FullModel

from hydraulic_physics import (
    hydraulic_physics_loss
)

from config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LAMBDA_REC,
    LAMBDA_LINK,
    LAMBDA_PHYS,
    DEVICE
)


# =========================
# MAIN TRAIN FUNCTION
# =========================
def train_model(graph_data):

    print(
        "   → Initializing model..."
    )

    model = FullModel().to(DEVICE)

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # =========================
    # GRAPH DATA
    # =========================
    x = graph_data.x.to(DEVICE)

    # =========================
    # USE TRAIN EDGES
    # =========================
    if hasattr(
        graph_data,
        "train_edges"
    ):

        edge_index = (
            graph_data.train_edges
            .to(DEVICE)
        )

    else:

        edge_index = (
            graph_data.edge_index
            .to(DEVICE)
        )

    # =========================
    # FEATURE MASK
    # =========================
    if hasattr(
        graph_data,
        "train_mask"
    ):

        mask = (
            graph_data.train_mask
            .float()
            .to(DEVICE)
        )

    elif hasattr(
        graph_data,
        "mask"
    ):

        mask = (
            graph_data.mask
            .float()
            .to(DEVICE)
        )

    else:

        mask = torch.ones_like(
            x,
            dtype=torch.float,
            device=DEVICE
        )

    # =========================
    # SAFETY
    # =========================
    if mask.shape != x.shape:

        mask = torch.ones_like(
            x,
            dtype=torch.float,
            device=DEVICE
        )

    print(
        "   → Starting training...\n"
    )

    training_logs = []

    best_loss = float("inf")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        # =====================
        # FORWARD
        # =====================
        x_pred, embeddings = model(
            x,
            edge_index
        )

        # =====================
        # NUMERICAL STABILITY
        # =====================
        x_pred = torch.nan_to_num(

            x_pred,

            nan=0.0,

            posinf=0.0,

            neginf=0.0
        )

        embeddings = torch.nan_to_num(

            embeddings,

            nan=0.0,

            posinf=0.0,

            neginf=0.0
        )

        # =====================
        # RECONSTRUCTION LOSS
        # =====================
        loss_rec = reconstruction_loss(

            x_pred,

            x,

            mask
        )

        # =====================
        # LINK PREDICTION LOSS
        # =====================
        loss_link = link_prediction_loss(

            model,

            embeddings,

            edge_index
        )

        # =====================
        # HYDRAULIC LOSS
        # =====================
        try:

            loss_phys = (
                hydraulic_physics_loss(

                    x_pred,

                    edge_index
                )
            )

            if (
                torch.isnan(loss_phys)
                or torch.isinf(loss_phys)
            ):

                loss_phys = torch.tensor(

                    0.0,

                    device=DEVICE
                )

        except Exception as e:

            print(
                f"   → Hydraulic loss skipped: {e}"
            )

            loss_phys = torch.tensor(

                0.0,

                device=DEVICE
            )

        # =====================
        # TOTAL LOSS
        # =====================
        loss = (

            LAMBDA_REC * loss_rec

            + LAMBDA_LINK * loss_link

            + LAMBDA_PHYS * loss_phys
        )

        # =====================
        # FINAL SAFETY
        # =====================
        if (
            torch.isnan(loss)
            or torch.isinf(loss)
        ):

            print(
                "   → Invalid loss detected."
            )

            break

        # =====================
        # BACKPROP
        # =====================
        loss.backward()

        # =====================
        # GRADIENT CLIPPING
        # =====================
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )

        optimizer.step()

        # =====================
        # SAVE BEST
        # =====================
        if loss.item() < best_loss:

            best_loss = loss.item()

        # =====================
        # LOGGING
        # =====================
        if epoch % 10 == 0:

            print(

                f"Epoch {epoch:03d} | "

                f"Total: {loss.item():.4f} | "

                f"Rec: {loss_rec.item():.4f} | "

                f"Link: {loss_link.item():.4f} | "

                f"Hydraulic: {loss_phys.item():.4f}"
            )

        training_logs.append(
            float(loss.item())
        )

    print(
        f"\n   → Best loss: {best_loss:.4f}"
    )

    return model, training_logs


# =========================
# RECONSTRUCTION LOSS
# =========================
def reconstruction_loss(
    x_pred,
    x_true,
    mask
):

    # =========================
    # SAFETY
    # =========================
    x_pred = torch.nan_to_num(

        x_pred,

        nan=0.0,

        posinf=0.0,

        neginf=0.0
    )

    x_true = torch.nan_to_num(

        x_true,

        nan=0.0,

        posinf=0.0,

        neginf=0.0
    )

    mask = torch.nan_to_num(

        mask,

        nan=0.0,

        posinf=0.0,

        neginf=0.0
    )

    # =========================
    # SHAPE SAFETY
    # =========================
    if mask.shape != x_true.shape:

        mask = torch.ones_like(
            x_true
        )

    # =========================
    # MSE
    # =========================
    diff = (
        x_pred - x_true
    ) ** 2

    masked_diff = diff * mask

    loss = masked_diff.sum() / (
        mask.sum() + 1e-8
    )

    return loss


# =========================
# NEGATIVE EDGE SAMPLING
# =========================
def negative_sampling(
    edge_index,
    num_nodes,
    num_neg_samples
):

    neg_src = torch.randint(

        0,

        num_nodes,

        (num_neg_samples,),

        device=edge_index.device
    )

    neg_tgt = torch.randint(

        0,

        num_nodes,

        (num_neg_samples,),

        device=edge_index.device
    )

    # =========================
    # REMOVE SELF LOOPS
    # =========================
    valid_mask = (
        neg_src != neg_tgt
    )

    neg_src = neg_src[
        valid_mask
    ]

    neg_tgt = neg_tgt[
        valid_mask
    ]

    return neg_src, neg_tgt


# =========================
# LINK PREDICTION LOSS
# =========================
def link_prediction_loss(
    model,
    embeddings,
    edge_index
):

    src = edge_index[0]

    tgt = edge_index[1]

    # =========================
    # POSITIVE EDGES
    # =========================
    pos_scores = model.predict_edges(

        embeddings,

        src,

        tgt
    )

    pos_labels = torch.ones_like(
        pos_scores
    )

    # =========================
    # NEGATIVE SAMPLING
    # =========================
    num_nodes = embeddings.shape[0]

    neg_src, neg_tgt = negative_sampling(

        edge_index,

        num_nodes,

        src.shape[0]
    )

    # =========================
    # NEGATIVE SCORES
    # =========================
    neg_scores = model.predict_edges(

        embeddings,

        neg_src,

        neg_tgt
    )

    neg_labels = torch.zeros_like(
        neg_scores
    )

    # =========================
    # CONCATENATE
    # =========================
    scores = torch.cat(

        [pos_scores, neg_scores],

        dim=0
    )

    labels = torch.cat(

        [pos_labels, neg_labels],

        dim=0
    )

    # =========================
    # BCE LOSS
    # =========================
    loss = (
        F.binary_cross_entropy_with_logits(

            scores,

            labels
        )
    )

    return loss