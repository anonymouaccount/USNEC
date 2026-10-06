import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


# =========================
# SAFE CONNECTIVITY CHECK
# =========================
def enforce_global_connectivity(
    edge_index,
    coords,
    max_connection_distance=30,
    reconnect_components=True
):

    print(
        "   → Checking graph connectivity..."
    )

    # =========================
    # EMPTY CASE
    # =========================
    if edge_index is None:

        print(
            "   → Empty edge index"
        )

        return np.empty(
            (2, 0),
            dtype=int
        )

    if edge_index.shape[1] == 0:

        print(
            "   → No edges available"
        )

        return edge_index

    # =========================
    # BUILD GRAPH
    # =========================
    G = nx.Graph()

    num_nodes = len(coords)

    G.add_nodes_from(
        range(num_nodes)
    )

    # =========================
    # CLEAN EXISTING EDGES
    # =========================
    valid_edges = []

    for src, tgt in edge_index.T:

        try:

            src = int(src)
            tgt = int(tgt)

            if (
                src >= 0
                and tgt >= 0
                and src < num_nodes
                and tgt < num_nodes
                and src != tgt
            ):

                G.add_edge(
                    src,
                    tgt
                )

                valid_edges.append([
                    src,
                    tgt
                ])

        except Exception:
            continue

    # =========================
    # NO VALID EDGES
    # =========================
    if len(valid_edges) == 0:

        print(
            "   → No valid edges"
        )

        return np.empty(
            (2, 0),
            dtype=int
        )

    # =========================
    # CLEAN EDGE INDEX
    # =========================
    edge_index = np.array(
        valid_edges,
        dtype=int
    ).T

    # =========================
    # COMPONENT ANALYSIS
    # =========================
    components = list(
        nx.connected_components(G)
    )

    print(
        f"   → Connected components: "
        f"{len(components)}"
    )

    component_sizes = sorted(
        [len(c) for c in components],
        reverse=True
    )

    print(
        f"   → Largest component size: "
        f"{component_sizes[0]}"
    )

    print(
        f"   → Smallest component size: "
        f"{component_sizes[-1]}"
    )

    # =========================
    # NODE DEGREES
    # =========================
    degrees = np.zeros(
        num_nodes,
        dtype=np.int32
    )

    for src, tgt in edge_index.T:

        degrees[src] += 1
        degrees[tgt] += 1

    isolated_nodes = np.where(
        degrees == 0
    )[0]

    print(
        f"   → Isolated nodes: "
        f"{len(isolated_nodes)}"
    )

    # =========================
    # PRESERVE ORIGINAL TOPOLOGY
    # =========================
    if not reconnect_components:

        print(
            "   → Preserving original topology"
        )

        return edge_index

    # =========================
    # SPATIAL RECONNECTION
    # =========================
    print(
        "   → Reconnecting fragmented components..."
    )

    coords = np.asarray(coords)

    nbrs = NearestNeighbors(
        n_neighbors=5,
        algorithm="ball_tree"
    )

    nbrs.fit(coords)

    added_edges = []

    edge_set = set()

    for src, tgt in edge_index.T:

        edge_set.add((src, tgt))
        edge_set.add((tgt, src))

    # =========================
    # RECONNECT ISOLATED NODES
    # =========================
    distances, indices = nbrs.kneighbors(
        coords
    )

    for node_idx in isolated_nodes:

        neigh_ids = indices[
            node_idx
        ][1:]

        neigh_dist = distances[
            node_idx
        ][1:]

        for neigh_idx, dist in zip(
            neigh_ids,
            neigh_dist
        ):

            if dist > max_connection_distance:
                continue

            if (
                (node_idx, neigh_idx)
                not in edge_set
            ):

                G.add_edge(
                    node_idx,
                    neigh_idx
                )

                added_edges.append([
                    node_idx,
                    neigh_idx
                ])

                edge_set.add(
                    (node_idx, neigh_idx)
                )

                break

    # =========================
    # CONNECT SMALL COMPONENTS
    # =========================
    components = list(
        nx.connected_components(G)
    )

    components = sorted(
        components,
        key=len,
        reverse=True
    )

    main_component = list(
        components[0]
    )

    main_coords = coords[
        main_component
    ]

    main_nbrs = NearestNeighbors(
        n_neighbors=1
    )

    main_nbrs.fit(main_coords)

    for component in components[1:]:

        component = list(component)

        best_edge = None

        best_distance = np.inf

        for node in component:

            node_coord = coords[
                node
            ].reshape(1, -1)

            dist, idx = main_nbrs.kneighbors(
                node_coord
            )

            dist = dist[0][0]

            nearest_main = main_component[
                idx[0][0]
            ]

            if (
                dist < best_distance
                and dist <= max_connection_distance
            ):

                best_distance = dist

                best_edge = (
                    node,
                    nearest_main
                )

        if best_edge is not None:

            src, tgt = best_edge

            if (
                (src, tgt)
                not in edge_set
            ):

                G.add_edge(
                    src,
                    tgt
                )

                added_edges.append([
                    src,
                    tgt
                ])

                edge_set.add(
                    (src, tgt)
                )

    # =========================
    # FINAL GRAPH
    # =========================
    final_edges = np.array(
        list(G.edges),
        dtype=int
    )

    final_edge_index = final_edges.T

    final_components = list(
        nx.connected_components(G)
    )

    final_degrees = np.zeros(
        num_nodes,
        dtype=np.int32
    )

    for src, tgt in final_edges:

        final_degrees[src] += 1
        final_degrees[tgt] += 1

    final_isolated = np.sum(
        final_degrees == 0
    )

    print(
        f"   → Added spatial edges: "
        f"{len(added_edges)}"
    )

    print(
        f"   → Final connected components: "
        f"{len(final_components)}"
    )

    print(
        f"   → Final isolated nodes: "
        f"{final_isolated}"
    )

    return final_edge_index