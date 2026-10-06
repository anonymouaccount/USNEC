import os
import torch


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)

# =========================
# CREATE DIRECTORIES
# =========================
os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# =========================
# DATA SETTINGS
# =========================

# -------------------------
# NODE IDENTIFIERS
# -------------------------
NODE_ID_COL = "id_node"

# -------------------------
# ARC TOPOLOGY
# -------------------------
SOURCE_COL = "f_id_node"

TARGET_COL = "t_id_node"

# -------------------------
# NODE FEATURES
# IMPORTANT:
# RAW GIS FEATURES
# -------------------------
FEATURE_COLUMNS = [

    # Coordinates
    "coordx",
    "coordy",

    # Elevation
    "altitude",

    # Hydraulic
    "profondeur",
    "radier",

    # Terrain
    "ztn",
    "zfe"
]

# Backward compatibility
NODE_FEATURES = FEATURE_COLUMNS

# -------------------------
# EDGE FEATURES
# -------------------------
EDGE_FEATURES = [

    "longueur",
    "diametre",
    "pente",
    "ffil",
    "tfil",
    "capacite"
]


# =========================
# CLEANING SETTINGS
# =========================

# -------------------------
# DUPLICATES
# -------------------------
REMOVE_DUPLICATES = True

# -------------------------
# GRAPH CLEANING
# IMPORTANT:
# KEEP ISOLATED NODES
# -------------------------
REMOVE_SELF_LOOPS = True

REMOVE_INVALID_ARCS = True

REMOVE_ISOLATED_NODES = False

# -------------------------
# MISSING VALUES
# -------------------------
FILL_NUMERIC_WITH = "median"

FILL_CATEGORICAL_WITH = "mode"


# =========================
# HYDRAULIC CONSTRAINTS
# =========================

# -------------------------
# SLOPE
# -------------------------
MIN_SLOPE = -0.01

# IMPORTANT:
# realistic urban drainage
MAX_SLOPE = 0.10

# -------------------------
# DEPTH
# -------------------------
MIN_DEPTH = 0

MAX_DEPTH = 100

# -------------------------
# DIAMETER
# -------------------------
MIN_DIAMETER = 10

MAX_DIAMETER = 5000


# =========================
# GRAPH SETTINGS
# =========================

# -------------------------
# SPATIAL RECONSTRUCTION
# -------------------------
ENABLE_SPATIAL_RECONSTRUCTION = True

# IMPORTANT:
# enough neighbors for topology repair
K_NEIGHBORS = 6

# Maximum KNN reconstruction distance
# REAL GIS METERS
MAX_NEIGHBOR_DISTANCE = 80

# Backward compatibility
MAX_EDGE_DISTANCE = MAX_NEIGHBOR_DISTANCE

# -------------------------
# EDGE SPLIT
# -------------------------
TEST_SIZE = 0.20

VALIDATION_EDGE_RATIO = 0.10

# Backward compatibility
TEST_EDGE_RATIO = TEST_SIZE

RANDOM_STATE = 42


# =========================
# TOPOLOGY RECONSTRUCTION
# =========================

# -------------------------
# LINK PREDICTION
# -------------------------
EDGE_SCORE_THRESHOLD = 0.45

# -------------------------
# TOPOLOGY PRESERVATION
# -------------------------
MIN_NODE_DEGREE = 1

ENABLE_ISOLATED_NODE_RECOVERY = True

# -------------------------
# PHYSICAL FILTERING
# -------------------------
TOPOLOGY_MAX_DISTANCE = 80

TOPOLOGY_MAX_SLOPE = 0.10


# =========================
# MODEL SETTINGS
# =========================
MODEL_TYPE = "GAT"

# -------------------------
# DIMENSIONS
# -------------------------
INPUT_DIM = len(
    FEATURE_COLUMNS
)

HIDDEN_DIM = 64

OUTPUT_DIM = 32

EMBEDDING_DIM = 32

# -------------------------
# GAT SETTINGS
# -------------------------
NUM_HEADS = 4

NUM_LAYERS = 2

# -------------------------
# REGULARIZATION
# -------------------------
DROPOUT = 0.20


# =========================
# TRAINING SETTINGS
# =========================
EPOCHS = 300

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0


# =========================
# LOSS WEIGHTS
# =========================

# -------------------------
# NODE RECONSTRUCTION
# -------------------------
LAMBDA_REC = 1.0

# -------------------------
# LINK PREDICTION
# IMPORTANT:
# strong topology learning
# -------------------------
LAMBDA_LINK = 0.5

# -------------------------
# HYDRAULIC PHYSICS
# -------------------------
LAMBDA_PHYS = 0.1

# -------------------------
# ANOMALY CONSISTENCY
# -------------------------
LAMBDA_ANOMALY = 0.05


# =========================
# ANOMALY DETECTION
# =========================
ANOMALY_THRESHOLD = 0.10


# =========================
# RANDOM SEED
# =========================
RANDOM_SEED = 42

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# =========================
# DEVICE
# =========================
DEVICE = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(
    f"\nUsing device: {DEVICE}"
)

print(
    f"Input node features: "
    f"{INPUT_DIM}"
)