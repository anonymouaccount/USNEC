import os
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import (
    Point,
    LineString
)

from shapely.wkt import loads

from config import (
    OUTPUT_PATH,
    NODE_ID_COL,
    SOURCE_COL,
    TARGET_COL
)


# =========================
# SAFE NUMERIC CONVERSION
# =========================
def pd_to_numeric_safe(series):

    return pd.to_numeric(

        series.astype(str)

        .str.strip()

        .str.replace(" ", "", regex=False)

        .str.replace("\u00A0", "", regex=False)

        .str.replace(",", ".", regex=False),

        errors="coerce"
    )


# =========================
# NORMALIZE IDS
# =========================
def normalize_id(series):

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(-1)
        .astype(np.int64)
        .astype(str)
    )


# =========================
# SAFE GEOMETRY LOADING
# =========================
def safe_load_geometry(g):

    try:

        if isinstance(g, str):

            return loads(g)

        return g

    except:

        return None


# =========================
# DETECT COORDINATE COLUMNS
# =========================
def detect_coordinate_columns(df):

    possible_x = [
        "x",
        "coordx",
        "coord_x",
        "longitude",
        "lon"
    ]

    possible_y = [
        "y",
        "coordy",
        "coord_y",
        "latitude",
        "lat"
    ]

    x_col = None
    y_col = None

    lower_cols = {
        c.lower(): c
        for c in df.columns
    }

    for col in possible_x:

        if col.lower() in lower_cols:

            x_col = lower_cols[
                col.lower()
            ]

            break

    for col in possible_y:

        if col.lower() in lower_cols:

            y_col = lower_cols[
                col.lower()
            ]

            break

    if x_col is None or y_col is None:

        raise ValueError(
            "Coordinate columns not found."
        )

    return x_col, y_col


# =========================
# DETECT GEOMETRY COLUMN
# =========================
def detect_geometry_column(df):

    possible_geom = [
        "geometry",
        "geom",
        "wkt",
        "shape",
        "the_geom"
    ]

    lower_cols = {
        c.lower(): c
        for c in df.columns
    }

    for col in possible_geom:

        if col.lower() in lower_cols:

            return lower_cols[
                col.lower()
            ]

    return None


# =========================
# CLEAN COLUMN NAMES
# SHAPEFILE LIMIT = 10
# =========================
def clean_column_names(df):

    rename_dict = {}

    used_names = set()

    for col in df.columns:

        clean = (
            col.lower()
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

        clean = clean[:10]

        counter = 1

        base = clean

        while clean in used_names:

            suffix = str(counter)

            clean = (
                base[:10 - len(suffix)]
                + suffix
            )

            counter += 1

        used_names.add(clean)

        rename_dict[col] = clean

    return df.rename(
        columns=rename_dict
    )


# =========================
# EXPORT NODE SHAPEFILE
# =========================
def export_nodes_shp(df_nodes):

    print(
        "   → Exporting node shapefile..."
    )

    df = df_nodes.copy()

    x_col, y_col = detect_coordinate_columns(df)

    # =========================
    # SAFE CONVERSION
    # =========================
    df[x_col] = pd_to_numeric_safe(
        df[x_col]
    )

    df[y_col] = pd_to_numeric_safe(
        df[y_col]
    )

    # =========================
    # REMOVE INVALID
    # =========================
    df = df.dropna(
        subset=[x_col, y_col]
    )

    # =========================
    # FILTER LAMBERT-93
    # =========================
    df = df[
        (df[x_col] > 500000)
        & (df[x_col] < 1300000)
        & (df[y_col] > 6000000)
        & (df[y_col] < 7200000)
    ].copy()

    print("\n   → Coordinate sample:")

    print(
        df[[x_col, y_col]]
        .sample(min(5, len(df)))
    )

    print("\n   → Coordinate stats:")

    print(
        df[[x_col, y_col]]
        .describe()
    )

    # =========================
    # CREATE GEOMETRY
    # =========================
    geometry = [

        Point(xy)

        for xy in zip(
            df[x_col],
            df[y_col]
        )
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:2154"
    )

    gdf = gdf[
        gdf.geometry.is_valid
    ]

    # =========================
    # CLEAN COLUMNS
    # =========================
    gdf = clean_column_names(
        gdf
    )

    # =========================
    # SAVE
    # =========================
    shp_path = os.path.join(
        OUTPUT_PATH,
        "clean_nodes.shp"
    )

    gpkg_path = os.path.join(
        OUTPUT_PATH,
        "clean_nodes.gpkg"
    )

    gdf.to_file(
        shp_path,
        driver="ESRI Shapefile"
    )

    gdf.to_file(
        gpkg_path,
        driver="GPKG"
    )

    print(
        f" Node shapefile saved: "
        f"{shp_path}"
    )


# =========================
# EXPORT ARC SHAPEFILE
# =========================
def export_arcs_shp(
    df_arcs,
    df_nodes
):

    print(
        "   → Exporting arc shapefile..."
    )

    nodes = df_nodes.copy()

    x_col, y_col = detect_coordinate_columns(
        nodes
    )

    # =========================
    # SAFE CONVERSION
    # =========================
    nodes[x_col] = pd_to_numeric_safe(
        nodes[x_col]
    )

    nodes[y_col] = pd_to_numeric_safe(
        nodes[y_col]
    )

    nodes = nodes.dropna(
        subset=[x_col, y_col]
    )

    # =========================
    # FILTER LAMBERT-93
    # =========================
    nodes = nodes[
        (nodes[x_col] > 500000)
        & (nodes[x_col] < 1300000)
        & (nodes[y_col] > 6000000)
        & (nodes[y_col] < 7200000)
    ].copy()

    # =========================
    # NORMALIZE IDS
    # =========================
    nodes[NODE_ID_COL] = normalize_id(
        nodes[NODE_ID_COL]
    )

    df_arcs = df_arcs.copy()

    df_arcs[SOURCE_COL] = normalize_id(
        df_arcs[SOURCE_COL]
    )

    df_arcs[TARGET_COL] = normalize_id(
        df_arcs[TARGET_COL]
    )

    # =========================
    # NODE ID → COORDS
    # =========================
    node_coords = {}

    for _, row in nodes.iterrows():

        node_id = row[
            NODE_ID_COL
        ]

        node_coords[node_id] = (

            float(row[x_col]),
            float(row[y_col])

        )

    # =========================
    # BUILD GIS LINES
    # =========================
    lines = []

    attributes = []

    valid_edges = 0

    skipped_edges = 0

    for _, row in df_arcs.iterrows():

        src = row.get(
            SOURCE_COL
        )

        tgt = row.get(
            TARGET_COL
        )

        # =====================
        # INVALID IDS
        # =====================
        if (
            src is None
            or tgt is None
        ):

            skipped_edges += 1
            continue

        # =====================
        # NODE EXISTS
        # =====================
        if (
            src not in node_coords
            or tgt not in node_coords
        ):

            skipped_edges += 1
            continue

        x1, y1 = node_coords[src]

        x2, y2 = node_coords[tgt]

        # =====================
        # FINITE CHECK
        # =====================
        if not (

            np.isfinite(x1)
            and np.isfinite(y1)
            and np.isfinite(x2)
            and np.isfinite(y2)

        ):

            skipped_edges += 1
            continue

        # =====================
        # SELF LOOP
        # =====================
        if (
            x1 == x2
            and y1 == y2
        ):

            skipped_edges += 1
            continue

        try:

            line = LineString([
                (x1, y1),
                (x2, y2)
            ])

            if not line.is_valid:

                skipped_edges += 1
                continue

            # =====================
            # OPTIONAL ATTRIBUTES
            # =====================
            longueur = row.get(
                "longueur",
                np.nan
            )

            diametre = row.get(
                "diametre",
                np.nan
            )

            pente = row.get(
                "pente",
                np.nan
            )

            lines.append(line)

            attributes.append({

                "source": str(src),

                "target": str(tgt),

                "length_m": round(
                    float(line.length),
                    2
                ),

                "longueur": longueur,

                "diametre": diametre,

                "pente": pente
            })

            valid_edges += 1

        except:

            skipped_edges += 1

    # =========================
    # EMPTY CASE
    # =========================
    if len(lines) == 0:

        print(
            "   → No valid arcs to export."
        )

        return

    # =========================
    # CREATE GDF
    # =========================
    gdf = gpd.GeoDataFrame(
        attributes,
        geometry=lines,
        crs="EPSG:2154"
    )

    gdf = gdf[
        gdf.geometry.is_valid
    ]

    # =========================
    # REMOVE DUPLICATES
    # =========================
    gdf = gdf.drop_duplicates(
        subset=[
            "source",
            "target"
        ]
    )

    # =========================
    # CLEAN COLUMNS
    # =========================
    gdf = clean_column_names(
        gdf
    )

    # =========================
    # SAVE
    # =========================
    shp_path = os.path.join(
        OUTPUT_PATH,
        "clean_arcs.shp"
    )

    gpkg_path = os.path.join(
        OUTPUT_PATH,
        "clean_arcs.gpkg"
    )

    gdf.to_file(
        shp_path,
        driver="ESRI Shapefile"
    )

    gdf.to_file(
        gpkg_path,
        driver="GPKG"
    )

    print(
        f" Arc shapefile saved: "
        f"{shp_path}"
    )

    print(
        f" Valid exported arcs: "
        f"{valid_edges}"
    )

    print(
        f" Skipped arcs: "
        f"{skipped_edges}"
    )


# =========================
# EXPORT BASINS
# =========================
def export_basins_shp(df_basins):

    print(
        "   → Exporting basin shapefile..."
    )

    df = df_basins.copy()

    geom_col = detect_geometry_column(df)

    if geom_col is not None:

        try:

            print(
                f"   → Using geometry column: "
                f"{geom_col}"
            )

            df = df.dropna(
                subset=[geom_col]
            )

            df["geometry"] = df[
                geom_col
            ].apply(
                safe_load_geometry
            )

            df = df.dropna(
                subset=["geometry"]
            )

            gdf = gpd.GeoDataFrame(
                df,
                geometry="geometry",
                crs="EPSG:2154"
            )

        except Exception as e:

            print(
                f"   Geometry parsing failed: {e}"
            )

            return

    else:

        print(
            "   No valid basin geometry found."
        )

        print(
            "   Skipping basin GIS export."
        )

        return

    gdf = gdf[
        gdf.geometry.is_valid
    ]

    if len(gdf) == 0:

        print(
            "   No valid basin geometries."
        )

        return

    # =========================
    # CLEAN COLUMNS
    # =========================
    gdf = clean_column_names(
        gdf
    )

    # =========================
    # SAVE
    # =========================
    shp_path = os.path.join(
        OUTPUT_PATH,
        "clean_basins.shp"
    )

    gpkg_path = os.path.join(
        OUTPUT_PATH,
        "clean_basins.gpkg"
    )

    gdf.to_file(
        shp_path,
        driver="ESRI Shapefile"
    )

    gdf.to_file(
        gpkg_path,
        driver="GPKG"
    )

    print(
        f" Basin shapefile saved: "
        f"{shp_path}"
    )