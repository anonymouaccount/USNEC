import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    BatchNorm
)

from config import (
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    DROPOUT,
    NUM_HEADS
)


# =========================
# LINK PREDICTOR
# =========================
class LinkPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                OUTPUT_DIM * 2,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM // 2
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM // 2,
                1
            )
        )

    def forward(
        self,
        z_src,
        z_tgt
    ):

        edge_features = torch.cat(
            [z_src, z_tgt],
            dim=1
        )

        scores = self.mlp(
            edge_features
        )

        return scores.squeeze(-1)


# =========================
# FULL MODEL
# =========================
class FullModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =========================
        # FIRST GAT LAYER
        # =========================
        self.gat1 = GATConv(

            in_channels=INPUT_DIM,

            out_channels=HIDDEN_DIM,

            heads=NUM_HEADS,

            dropout=DROPOUT,

            concat=True,

            add_self_loops=True
        )

        self.bn1 = BatchNorm(
            HIDDEN_DIM * NUM_HEADS
        )

        # =========================
        # SECOND GAT LAYER
        # =========================
        self.gat2 = GATConv(

            in_channels=HIDDEN_DIM * NUM_HEADS,

            out_channels=OUTPUT_DIM,

            heads=1,

            dropout=DROPOUT,

            concat=False,

            add_self_loops=True
        )

        self.bn2 = BatchNorm(
            OUTPUT_DIM
        )

        # =========================
        # NODE RECONSTRUCTION HEAD
        # =========================
        self.reconstruction_head = nn.Sequential(

            nn.Linear(
                OUTPUT_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM // 2
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM // 2,
                INPUT_DIM
            )
        )

        # =========================
        # LINK PREDICTOR
        # =========================
        self.link_predictor = LinkPredictor()

        # =========================
        # INITIALIZATION
        # =========================
        self.reset_parameters()

    # =========================
    # PARAMETER INITIALIZATION
    # =========================
    def reset_parameters(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # =========================
    # ENCODE
    # =========================
    def encode(
        self,
        x,
        edge_index
    ):

        # =====================
        # FIRST GAT
        # =====================
        h = self.gat1(
            x,
            edge_index
        )

        h = self.bn1(h)

        h = F.elu(h)

        h = F.dropout(

            h,

            p=DROPOUT,

            training=self.training
        )

        # =====================
        # SECOND GAT
        # =====================
        embeddings = self.gat2(
            h,
            edge_index
        )

        embeddings = self.bn2(
            embeddings
        )

        embeddings = F.elu(
            embeddings
        )

        return embeddings

    # =========================
    # DECODE
    # =========================
    def decode(
        self,
        embeddings
    ):

        reconstructed = (
            self.reconstruction_head(
                embeddings
            )
        )

        return reconstructed

    # =========================
    # FORWARD
    # =========================
    def forward(
        self,
        x,
        edge_index
    ):

        # =====================
        # LATENT EMBEDDINGS
        # =====================
        embeddings = self.encode(
            x,
            edge_index
        )

        # =====================
        # RECONSTRUCTION
        # =====================
        x_reconstructed = self.decode(
            embeddings
        )

        return (

            x_reconstructed,

            embeddings
        )

    # =========================
    # EDGE PREDICTION
    # =========================
    def predict_edges(
        self,
        embeddings,
        src,
        tgt
    ):

        # =====================
        # SOURCE/TARGET EMBEDS
        # =====================
        h_src = embeddings[src]

        h_tgt = embeddings[tgt]

        # =====================
        # EDGE SCORE
        # =====================
        scores = self.link_predictor(

            h_src,
            h_tgt
        )

        return scores