import os
import json
import random
import faiss
import h5py
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cosine


def load_faiss_index(index_path):
    """
    Load a FAISS index from disk.

    Parameters
    ----------
    index_path : str
        Path to the FAISS index file.

    Returns
    -------
    faiss.Index
        The loaded FAISS index.
    """
    print(f"Loading FAISS index from {index_path}...")
    return faiss.read_index(index_path)


def load_embeddings(h5_path):
    """
    Load embeddings from an H5 file.

    Parameters
    ----------
    h5_path : str
        Path to the H5 file containing embeddings.

    Returns
    -------
    pd.DataFrame
        DataFrame with filenames as index and embeddings as columns.
    """
    print(f"Loading embeddings from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        # check if we have 'embeddings' or 'histograms' as the key
        emb_key = "embeddings" if "embeddings" in f else "histograms"
        # Convert to numpy arrays first to make them iterable
        emb_array = np.array(f[emb_key])
        filenames_array = np.array(f["filenames"])

        # Now process the numpy arrays
        embeddings = [e / np.linalg.norm(e, keepdims=True) for e in emb_array]
        filenames = [str(name[0], "utf-8") for name in filenames_array]

    return pd.DataFrame(
        {
            "embeddings": embeddings,
            "normed_embeddings": embeddings,  # already normalized above
        },
        index=filenames,
    )


def select_random_midi(midi_dir):
    """
    Select a random MIDI file from the specified directory.

    Parameters
    ----------
    midi_dir : str
        Directory containing MIDI files.

    Returns
    -------
    str
        The selected MIDI filename.
    """
    print(f"Selecting random MIDI file from {midi_dir}...")
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith((".mid", ".midi"))]
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    selected_file = random.choice(midi_files)
    print(f"Selected: {selected_file}")
    return selected_file


def load_graph(track_name, graph_dir):
    """
    Load the similarity graph for a specific track.

    Parameters
    ----------
    track_name : str
        Name of the track.
    graph_dir : str
        Directory containing graph files.

    Returns
    -------
    nx.Graph
        The loaded graph.
    """
    graph_path = os.path.join(graph_dir, f"{track_name}.json")
    print(f"Loading graph from {graph_path}...")

    with open(graph_path, "r") as f:
        graph_data = json.load(f)

    return nx.node_link_graph(graph_data)


def find_similar_files(query_file, query_embedding, faiss_index, embeddings_df, k=10):
    """
    Find the k most similar files to the query file using FAISS.

    Parameters
    ----------
    query_file : str
        The query file name.
    query_embedding : np.ndarray
        The embedding of the query file.
    faiss_index : faiss.Index
        The FAISS index.
    embeddings_df : pd.DataFrame
        DataFrame containing embeddings.
    k : int, optional
        Number of similar files to find, by default 10.

    Returns
    -------
    list
        List of tuples (filename, similarity score).
    """
    print(f"Finding {k} most similar files to {query_file}...")

    # Reshape query embedding for FAISS
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Search using FAISS - get many more than needed to filter for different tracks
    distances, indices = faiss_index.search(
        query_embedding, k + 500
    )  # Get more than needed to filter later

    # Get filenames and scores
    similar_files = []
    query_track = query_file.split("_")[0]

    for idx, dist in zip(indices[0], distances[0]):
        filename = embeddings_df.index[idx]
        track = filename.split("_")[0]

        # Only include files from different tracks
        if track != query_track:
            similarity = float(dist)  # Convert to float for better printing
            similar_files.append((filename, similarity))

            # Break once we have enough files
            if len(similar_files) >= k:
                break

    # If we couldn't find enough files from different tracks, include files from the same track
    if len(similar_files) < k:
        print(
            f"Warning: Could only find {len(similar_files)} files from different tracks"
        )

        # Add files from the same track if needed
        if len(similar_files) == 0:
            print(
                "No files from different tracks found, including files from the same track"
            )
            for idx, dist in zip(indices[0], distances[0]):
                filename = embeddings_df.index[idx]
                # Skip the query file itself
                if filename != query_file.split(".")[0]:
                    similarity = float(dist)
                    similar_files.append((filename, similarity))
                    if len(similar_files) >= k:
                        break

    return similar_files


def find_path(source_file, target_file, graph):
    """
    Find the shortest path between source and target files in the graph.

    Parameters
    ----------
    source_file : str
        Source file name.
    target_file : str
        Target file name.
    graph : nx.Graph
        The graph to search in.

    Returns
    -------
    list
        List of nodes in the path.
    """
    print(f"Finding path from {source_file} to {target_file}...")

    # Remove file extension if present
    source_file = source_file.split(".")[0]
    target_file = target_file.split(".")[0]

    # Check if nodes exist in the graph
    if source_file not in graph:
        raise ValueError(f"Source file {source_file} not found in graph")
    if target_file not in graph:
        raise ValueError(f"Target file {target_file} not found in graph")

    try:
        path = nx.shortest_path(
            graph, source=source_file, target=target_file, weight="weight"
        )
        print(f"Found path with {len(path)} nodes")
        return path
    except nx.NetworkXNoPath:
        print(f"No path found between {source_file} and {target_file}")
        return []


def visualize_graph(graph, path=None, title="Similarity Graph"):
    """
    Visualize the graph and highlight the path if provided.

    Parameters
    ----------
    graph : nx.Graph
        The graph to visualize.
    path : list, optional
        List of nodes in the path to highlight, by default None.
    title : str, optional
        Title of the plot, by default "Similarity Graph".
    """
    plt.figure(figsize=(12, 10))

    # Create a spring layout
    pos = nx.spring_layout(graph, seed=42)

    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_size=50, alpha=0.6)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)

    # Highlight the path if provided
    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(
            graph, pos, nodelist=path, node_size=100, node_color="red", alpha=0.8
        )
        nx.draw_networkx_edges(
            graph, pos, edgelist=path_edges, width=2, edge_color="red", alpha=0.8
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_path_with_neighbors(
    combined_graph, path, embeddings_df, title="Path with First-Order Connections"
):
    """
    Visualize only the path and first-order connections of nodes in the path.

    Parameters
    ----------
    combined_graph : nx.Graph
        The combined graph containing all nodes.
    path : list
        List of nodes in the path to highlight.
    embeddings_df : pd.DataFrame
        DataFrame containing embeddings for calculating similarities.
    title : str, optional
        Title of the plot, by default "Path with First-Order Connections".
    """
    if not path or len(path) < 2:
        print("path is too short to visualize")
        return

    # create a subgraph with only the path nodes and their first-order neighbors
    path_nodes = set(path)
    neighbor_nodes = set()

    for node in path:
        if node in combined_graph:
            neighbor_nodes.update(combined_graph.neighbors(node))

    # create the subgraph with path nodes and their neighbors
    nodes_to_include = path_nodes.union(neighbor_nodes)
    subgraph = combined_graph.subgraph(nodes_to_include)

    # create a spring layout with path nodes fixed in a line
    pos = {}
    path_length = len(path)

    # position path nodes in a line
    for i, node in enumerate(path):
        pos[node] = np.array([i / (path_length - 1), 0.5])

    # use spring layout for the neighbors
    other_nodes = nodes_to_include - path_nodes
    other_pos = nx.spring_layout(subgraph.subgraph(other_nodes), seed=42)
    pos.update(other_pos)

    plt.figure(figsize=(14, 8))

    # draw all nodes and edges with low alpha
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, alpha=0.4)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5)

    # draw path nodes and edges with high alpha
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=path, node_size=100, node_color="red", alpha=0.8
    )

    # draw path edges
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(
        subgraph, pos, edgelist=path_edges, width=2, edge_color="red", alpha=0.8
    )

    # add labels to path nodes
    labels = {node: node.split("_")[0] + "_" + node.split("_")[1] for node in path}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.

    Parameters
    ----------
    embedding1 : np.ndarray
        First embedding.
    embedding2 : np.ndarray
        Second embedding.

    Returns
    -------
    float
        Cosine similarity between the embeddings.
    """
    return 1 - cosine(embedding1, embedding2)


def find_best_connection_points(
    source_graph, target_graph, source_track, target_track, embeddings_df
):
    """
    Find the best connection points between two tracks based on embedding similarity.

    Parameters
    ----------
    source_graph : nx.Graph
        Graph for the source track.
    target_graph : nx.Graph
        Graph for the target track.
    source_track : str
        Name of the source track.
    target_track : str
        Name of the target track.
    embeddings_df : pd.DataFrame
        DataFrame containing embeddings.

    Returns
    -------
    tuple
        (best_source_node, best_target_node, similarity)
    """
    print(
        f"Finding best connection points between {source_track} and {target_track}..."
    )

    best_similarity = -1
    best_source_node = None
    best_target_node = None

    # Get all nodes from source and target graphs
    source_nodes = [n for n in source_graph.nodes() if n.split("_")[0] == source_track]
    target_nodes = [n for n in target_graph.nodes() if n.split("_")[0] == target_track]

    print(
        f"Comparing {len(source_nodes)} source nodes with {len(target_nodes)} target nodes"
    )

    # Sample nodes if there are too many to compare efficiently
    max_comparisons = 100
    if len(source_nodes) * len(target_nodes) > max_comparisons:
        if len(source_nodes) > 20:
            source_nodes = random.sample(source_nodes, 20)
        if len(target_nodes) > 20:
            target_nodes = random.sample(target_nodes, 20)
        print(
            f"Sampled down to {len(source_nodes)} source nodes and {len(target_nodes)} target nodes"
        )

    # Find the most similar pair of nodes between the two graphs
    for source_node in source_nodes:
        if source_node not in embeddings_df.index:
            continue

        source_embedding = embeddings_df.loc[source_node, "normed_embeddings"]

        for target_node in target_nodes:
            if target_node not in embeddings_df.index:
                continue

            target_embedding = embeddings_df.loc[target_node, "normed_embeddings"]

            # Calculate similarity
            similarity = calculate_cosine_similarity(source_embedding, target_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_source_node = source_node
                best_target_node = target_node

    if best_source_node and best_target_node:
        print(
            f"Best connection: {best_source_node} -> {best_target_node} (similarity: {best_similarity:.4f})"
        )
    else:
        print("Could not find a good connection between the tracks")

    return best_source_node, best_target_node, best_similarity


def main():
    # Paths
    faiss_index_path = "data/tables/20250110/specdiff.faiss"
    h5_path = "data/tables/20250110/specdiff.h5"
    midi_dir = "data/datasets/20250110/augmented"
    graph_dir = "data/datasets/20250110/graphs"

    # Load FAISS index and embeddings
    faiss_index = load_faiss_index(faiss_index_path)
    embeddings_df = load_embeddings(h5_path)
    print(f"Loaded {faiss_index.ntotal} embeddings")

    # Select random MIDI file
    selected_midi = select_random_midi(midi_dir)
    track_name = selected_midi.split("_")[0]
    print(f"Selected track: {track_name}")

    # Load the corresponding graph
    graph = load_graph(track_name, graph_dir)
    print(
        f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # Get the embedding for the selected file
    selected_file_base = selected_midi.split(".")[0]
    if selected_file_base in embeddings_df.index:
        query_embedding = embeddings_df.loc[selected_file_base, "normed_embeddings"]
        print(f"Found embedding for {selected_file_base}")
    else:
        print(
            f"Warning: {selected_file_base} not found in embeddings, trying with extension..."
        )
        if selected_midi in embeddings_df.index:
            query_embedding = embeddings_df.loc[selected_midi, "normed_embeddings"]
            print(f"Found embedding for {selected_midi}")
        else:
            raise ValueError(f"Could not find embedding for {selected_midi}")

    # Find similar files from different tracks
    similar_files = find_similar_files(
        selected_midi, query_embedding, faiss_index, embeddings_df
    )

    print("\nTop 10 most similar files from different tracks:")
    for i, (filename, score) in enumerate(similar_files, 1):
        print(f"{i}. {filename} (similarity: {score:.4f})")

    # Select the most similar file from a different track
    if similar_files:
        most_similar_file = similar_files[0][0]
        most_similar_track = most_similar_file.split("_")[0]

        print(f"\nMapping path from {selected_midi} to {most_similar_file}")

        # We need to load the graph for the target track to find a path to it
        target_graph = load_graph(most_similar_track, graph_dir)

        # Find the best connection points between the two tracks
        source_node = selected_file_base
        target_node = most_similar_file.split(".")[0]

        # Try to find interesting paths within each graph
        source_end_node = None
        target_start_node = None

        # Find a path within the source graph
        source_nodes = list(graph.nodes())
        if len(source_nodes) > 1 and source_node in source_nodes:
            # Find a node that's a few hops away from the source
            for node in source_nodes:
                if node != source_node and nx.has_path(graph, source_node, node):
                    try:
                        path_length = nx.shortest_path_length(graph, source_node, node)
                        if 2 <= path_length <= 4:  # Look for a node 2-4 hops away
                            source_end_node = node
                            break
                    except nx.NetworkXNoPath:
                        continue

        # Find a path within the target graph
        target_nodes = list(target_graph.nodes())
        if len(target_nodes) > 1 and target_node in target_nodes:
            # Find a node that's a few hops away from the target
            for node in target_nodes:
                if node != target_node and nx.has_path(target_graph, node, target_node):
                    try:
                        path_length = nx.shortest_path_length(
                            target_graph, node, target_node
                        )
                        if 2 <= path_length <= 4:  # Look for a node 2-4 hops away
                            target_start_node = node
                            break
                    except nx.NetworkXNoPath:
                        continue

        # Find the best connection between the two tracks
        if source_end_node and target_start_node:
            print(f"Found path endpoints: {source_end_node} -> {target_start_node}")

            # Calculate similarity for the connection
            if (
                source_end_node in embeddings_df.index
                and target_start_node in embeddings_df.index
            ):
                source_embedding = embeddings_df.loc[
                    source_end_node, "normed_embeddings"
                ]
                target_embedding = embeddings_df.loc[
                    target_start_node, "normed_embeddings"
                ]
                similarity = calculate_cosine_similarity(
                    source_embedding, target_embedding
                )
                weight = max(float(1 - similarity), 0.01)
                print(f"Connection similarity: {similarity:.4f} (weight: {weight:.4f})")
            else:
                print(
                    "Warning: Could not calculate similarity for connection, using default weight"
                )
                weight = 1.0
        else:
            # If we couldn't find good path endpoints, use the original nodes
            print("Could not find good path endpoints, using direct connection")
            source_end_node = source_node
            target_start_node = target_node

            # Calculate similarity for the direct connection
            if (
                source_node in embeddings_df.index
                and target_node in embeddings_df.index
            ):
                source_embedding = embeddings_df.loc[source_node, "normed_embeddings"]
                target_embedding = embeddings_df.loc[target_node, "normed_embeddings"]
                similarity = calculate_cosine_similarity(
                    source_embedding, target_embedding
                )
                weight = max(float(1 - similarity), 0.01)
                print(
                    f"Direct connection similarity: {similarity:.4f} (weight: {weight:.4f})"
                )
            else:
                print(
                    "Warning: Could not calculate similarity for direct connection, using default weight"
                )
                weight = 1.0

        # Create a combined graph
        combined_graph = nx.compose(graph, target_graph)

        # Add the connection between the two tracks
        combined_graph.add_edge(source_end_node, target_start_node, weight=weight)

        # Find the complete path
        source_path = []
        target_path = []

        if source_node != source_end_node:
            try:
                source_path = nx.shortest_path(
                    graph, source=source_node, target=source_end_node, weight="weight"
                )
                print(f"Source path: {' -> '.join(source_path)}")
            except nx.NetworkXNoPath:
                print(
                    f"No path found within source graph from {source_node} to {source_end_node}"
                )
                source_path = [source_node, source_end_node]
        else:
            source_path = [source_node]

        if target_start_node != target_node:
            try:
                target_path = nx.shortest_path(
                    target_graph,
                    source=target_start_node,
                    target=target_node,
                    weight="weight",
                )
                print(f"Target path: {' -> '.join(target_path)}")
            except nx.NetworkXNoPath:
                print(
                    f"No path found within target graph from {target_start_node} to {target_node}"
                )
                target_path = [target_start_node, target_node]
        else:
            target_path = [target_node]

        # Combine the paths - ensure both are lists
        complete_path = []
        if len(source_path) > 0 and len(target_path) > 0:
            # Avoid duplicating the connection node
            if source_path[-1] == target_path[0]:
                complete_path = list(source_path) + list(target_path[1:])
            else:
                complete_path = list(source_path) + list(target_path)
        elif len(source_path) > 0:
            complete_path = list(source_path)
        elif len(target_path) > 0:
            complete_path = list(target_path)
        else:
            complete_path = [source_node, target_node]

        print(
            f"Complete path ({len(complete_path)} nodes): {' -> '.join(complete_path)}"
        )

        # Visualize the path with first-order connections
        print("\nVisualizing path with first-order connections...")
        visualize_path_with_neighbors(
            combined_graph,
            complete_path,
            embeddings_df,
            title=f"Path from {track_name} to {most_similar_track} with neighbors",
        )
    else:
        print("\nNo similar files found. Cannot create a path.")
        return

    print("\nDone!")


if __name__ == "__main__":
    main()
