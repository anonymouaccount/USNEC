import pandas as pd
import numpy as np

from config import (
    NODE_ID_COL,
    SOURCE_COL,
    TARGET_COL,
    REMOVE_DUPLICATES,
    FILL_NUMERIC_WITH,
    FILL_CATEGORICAL_WITH,
    MIN_SLOPE,
    MIN_DEPTH,
    MAX_DIAMETER
)


# =========================
# SAFE NUMERIC CLEANING
# =========================
def clean_numeric_column(series):

    return pd.to_numeric(

        series.astype(str)

        .str.strip()

        .str.replace(
            ",",
            ".",
            regex=False
        )

        .str.replace(
            " ",
            "",
            regex=False
        )

        .str.replace(
            "\xa0",
            "",
            regex=False
        )

        .str.replace(
            "\u00A0",
            "",
            regex=False
        ),

        errors="coerce"
    )


# =========================
# SAFE ID CLEANING
# =========================
def clean_id_column(series):

    return (

        series.astype(str)

        .str.strip()

        .str.replace(
            ".0",
            "",
            regex=False
        )

        .replace({
            "nan": np.nan,
            "None": np.nan,
            "": np.nan
        })
    )


# =========================
# LOAD DATA
# =========================
def load_data(
    nodes_path,
    arcs_path,
    basins_path
):

    read_params = {

        "sep": ",",

        "encoding": "utf-8",

        "engine": "python",

        "on_bad_lines": "skip"
    }

    # =====================
    # LOAD FILES
    # =====================
    df_nodes = pd.read_csv(
        nodes_path,
        **read_params
    )

    df_arcs = pd.read_csv(
        arcs_path,
        **read_params
    )

    df_basins = pd.read_csv(
        basins_path,
        **read_params
    )

    # =====================
    # CLEAN COLUMN NAMES
    # =====================
    for df in [
        df_nodes,
        df_arcs,
        df_basins
    ]:

        df.columns = (

            df.columns

            .str.strip()

            .str.replace(
                "\ufeff",
                "",
                regex=False
            )
        )

    # =====================
    # DEBUG
    # =====================
    print("\nNODE COLUMNS:")
    print(df_nodes.columns.tolist())

    print("\nARC COLUMNS:")
    print(df_arcs.columns.tolist())

    print("\nBASIN COLUMNS:")
    print(df_basins.columns.tolist())

    # =====================
    # CLEAN IDS
    # VERY IMPORTANT
    # =====================
    if NODE_ID_COL in df_nodes.columns:

        df_nodes[NODE_ID_COL] = clean_id_column(
            df_nodes[NODE_ID_COL]
        )

    if SOURCE_COL in df_arcs.columns:

        df_arcs[SOURCE_COL] = clean_id_column(
            df_arcs[SOURCE_COL]
        )

    if TARGET_COL in df_arcs.columns:

        df_arcs[TARGET_COL] = clean_id_column(
            df_arcs[TARGET_COL]
        )

    # =====================
    # NODE NUMERIC COLUMNS
    # =====================
    node_numeric_cols = [

        "altitude",
        "angle",

        "coordx",
        "coordx_m",

        "coordy",
        "coordy_m",

        "profondeur",

        "radier",

        "ztn",
        "zfe"
    ]

    for col in node_numeric_cols:

        if col in df_nodes.columns:

            df_nodes[col] = clean_numeric_column(
                df_nodes[col]
            )

    # =====================
    # ARC NUMERIC COLUMNS
    # =====================
    arc_numeric_cols = [

        "diametre",
        "dimension",

        "ffil",
        "ffil_prof",

        "long_mes",
        "longueur",

        "pente",
        "penter",

        "tfil",
        "tfil_prof",

        "capacite",

        "dim_num",

        "st_length(shape)"
    ]

    for col in arc_numeric_cols:

        if col in df_arcs.columns:

            df_arcs[col] = clean_numeric_column(
                df_arcs[col]
            )

    # =====================
    # BASIN NUMERIC COLUMNS
    # =====================
    basin_numeric_cols = [

        "volume",

        "st_area(shape)",

        "st_length(shape)"
    ]

    for col in basin_numeric_cols:

        if col in df_basins.columns:

            df_basins[col] = clean_numeric_column(
                df_basins[col]
            )

    # =====================
    # REMOVE FULLY EMPTY ROWS
    # =====================
    df_nodes = df_nodes.dropna(
        how="all"
    )

    df_arcs = df_arcs.dropna(
        how="all"
    )

    df_basins = df_basins.dropna(
        how="all"
    )

    # =====================
    # COORDINATE DEBUG
    # =====================
    if (
        "coordx" in df_nodes.columns
        and "coordy" in df_nodes.columns
    ):

        valid_coords = df_nodes.dropna(
            subset=["coordx", "coordy"]
        )

        if len(valid_coords) > 0:

            print("\n   → Coordinate sample:")

            print(
                valid_coords[
                    ["coordx", "coordy"]
                ].sample(
                    min(5, len(valid_coords))
                )
            )

            print("\n   → Coordinate stats:")

            print(
                valid_coords[
                    ["coordx", "coordy"]
                ].describe()
            )

    return (

        df_nodes,

        df_arcs,

        df_basins
    )


# =========================
# MAIN CLEAN FUNCTION
# =========================
def clean_data(
    df_nodes,
    df_arcs,
    df_basins
):

    print("   → Cleaning nodes...")

    df_nodes = clean_nodes(
        df_nodes
    )

    print("   → Cleaning arcs...")

    df_arcs = clean_arcs(
        df_arcs,
        df_nodes
    )

    print("   → Cleaning basins...")

    df_basins = clean_basins(
        df_basins
    )

    # =====================
    # TOPOLOGY DIAGNOSTICS
    # =====================
    print("\n=== TOPOLOGY DIAGNOSTICS ===")

    connected_nodes = set(
        df_arcs[SOURCE_COL]
    ).union(
        set(df_arcs[TARGET_COL])
    )

    isolated_nodes = (
        len(df_nodes)
        - len(connected_nodes)
    )

    print(
        f"Connected nodes: "
        f"{len(connected_nodes)}"
    )

    print(
        f"Isolated nodes preserved: "
        f"{isolated_nodes}"
    )

    return (

        df_nodes,

        df_arcs,

        df_basins
    )


# =========================
# CLEAN NODES
# =========================
def clean_nodes(df):

    df = df.copy()

    # =====================
    # REMOVE DUPLICATES
    # =====================
    if (
        REMOVE_DUPLICATES
        and NODE_ID_COL in df.columns
    ):

        df = df.drop_duplicates(
            subset=[NODE_ID_COL]
        )

    # =====================
    # REMOVE INVALID IDS ONLY
    # =====================
    df = df.dropna(
        subset=[NODE_ID_COL]
    )

    # =====================
    # KEEP MOST NODES
    # DO NOT REMOVE
    # DISCONNECTED NODES
    # =====================

    # =====================
    # REMOVE ONLY INVALID
    # COORDINATES
    # =====================
    coord_cols = []

    if "coordx" in df.columns:
        coord_cols.append("coordx")

    if "coordy" in df.columns:
        coord_cols.append("coordy")

    if len(coord_cols) > 0:

        df = df.dropna(
            subset=coord_cols
        )

    # =====================
    # HANDLE MISSING VALUES
    # =====================
    df = fill_missing_values(df)

    print(
        f"   → Clean nodes: {len(df)}"
    )

    return df.reset_index(
        drop=True
    )


# =========================
# CLEAN ARCS
# =========================
def clean_arcs(
    df,
    df_nodes
):

    df = df.copy()

    # =====================
    # REMOVE DUPLICATES
    # =====================
    if REMOVE_DUPLICATES:

        df = df.drop_duplicates()

    # =====================
    # REMOVE NULL TOPOLOGY
    # =====================
    df = df.dropna(
        subset=[
            SOURCE_COL,
            TARGET_COL
        ]
    )

    # =====================
    # REMOVE SELF LOOPS
    # =====================
    df = df[
        df[SOURCE_COL]
        != df[TARGET_COL]
    ]

    # =====================
    # VALID NODE IDS
    # =====================
    valid_nodes = set(
        df_nodes[NODE_ID_COL]
    )

    # =====================
    # KEEP ONLY VALID ARCS
    # =====================
    before = len(df)

    df = df[

        df[SOURCE_COL].isin(
            valid_nodes
        )

        &

        df[TARGET_COL].isin(
            valid_nodes
        )
    ]

    removed = before - len(df)

    print(
        f"   → Invalid topology arcs removed: "
        f"{removed}"
    )

    # =====================
    # HYDRAULIC VALIDATION
    # =====================
    if "pente" in df.columns:

        df.loc[
            df["pente"] < MIN_SLOPE,
            "pente"
        ] = np.nan

    if "ffil_prof" in df.columns:

        df.loc[
            df["ffil_prof"] < MIN_DEPTH,
            "ffil_prof"
        ] = np.nan

    if "diametre" in df.columns:

        df.loc[
            df["diametre"] > MAX_DIAMETER,
            "diametre"
        ] = np.nan

    # =====================
    # HANDLE MISSING VALUES
    # =====================
    df = fill_missing_values(df)

    print(
        f"   → Clean arcs: {len(df)}"
    )

    return df.reset_index(
        drop=True
    )


# =========================
# CLEAN BASINS
# =========================
def clean_basins(df):

    df = df.copy()

    if REMOVE_DUPLICATES:

        df = df.drop_duplicates()

    df = fill_missing_values(df)

    print(
        f"   → Clean basins: {len(df)}"
    )

    return df.reset_index(
        drop=True
    )


# =========================
# HANDLE MISSING VALUES
# =========================
def fill_missing_values(df):

    df = df.copy()

    for col in df.columns:

        # =====================
        # NUMERIC
        # =====================
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            if df[col].isnull().sum() > 0:

                valid_values = df[col].dropna()

                if len(valid_values) == 0:

                    fill_value = 0

                elif (
                    FILL_NUMERIC_WITH
                    == "mean"
                ):

                    fill_value = (
                        valid_values.mean()
                    )

                elif (
                    FILL_NUMERIC_WITH
                    == "median"
                ):

                    fill_value = (
                        valid_values.median()
                    )

                else:

                    fill_value = 0

                if pd.isna(fill_value):

                    fill_value = 0

                df[col] = df[col].fillna(
                    fill_value
                )

        # =====================
        # CATEGORICAL
        # =====================
        else:

            if df[col].isnull().sum() > 0:

                if (
                    FILL_CATEGORICAL_WITH
                    == "mode"
                ):

                    mode_values = (
                        df[col].mode()
                    )

                    if len(mode_values) > 0:

                        fill_value = (
                            mode_values[0]
                        )

                    else:

                        fill_value = "unknown"

                else:

                    fill_value = "unknown"

                df[col] = df[col].fillna(
                    fill_value
                )

    return df