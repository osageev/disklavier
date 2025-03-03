import os
import json
import h5py
import faiss
import heapq
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

input_file = "20240401-065-03_0228-0236_t00s00"
metric = "specdiff"
dataset = "20250110"
embeddings_file = f"../data/tables/{dataset}/{metric}.h5"
index_file = f"../data/tables/{dataset}/{metric}.faiss"
segmented_dir = f"../data/datasets/{dataset}/segmented"
augmented_dir = f"../data/datasets/{dataset}/augmented"
graph_dir = f"../data/datasets/{dataset}/graphs"
candidate_count = 3  # number of neighbors from different tracks to find


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_distance(file1: str, file2: str) -> float:
    """Computes cosine distance (1 - cosine similarity)."""
    return 1 - cosine_similarity(
        np.asarray(df.loc[file1, "embeddings"], dtype=np.float32),
        np.asarray(df.loc[file2, "embeddings"], dtype=np.float32),
    )


with h5py.File(embeddings_file, "r") as f:
    embeddings = [e / np.linalg.norm(e, keepdims=True) for e in f["embeddings"]]  # type: ignore
    filenames = [str(name[0], "utf-8") for name in f["filenames"]]  # type: ignore
df = pd.DataFrame({"embeddings": embeddings}, index=filenames, dtype=np.float32)

faiss_index = faiss.read_index(index_file)

grouped_files = defaultdict(list)

for filename in os.listdir(augmented_dir):
    if filename.endswith(".mid"):
        track_name = filename.split("_")[0]
        grouped_files[track_name].append(filename.split(".")[0])  # + "_t00s00")

for track, files in grouped_files.items():
    print(f"Track: {track}")
    for file in files:
        print(f"  {file}")
    break

# generate graph for each track
# Create a new graph
track_graphs = {}

for track, files in grouped_files.items():
    G = nx.Graph(name=track)

    # add nodes
    for file in tqdm(files, "adding nodes"):
        G.add_node(file)

    # add edges
    for i in tqdm(range(len(files)), "adding edges"):
        for j in range(i + 1, len(files)):
            G.add_edge(files[i], files[j], weight=cosine_distance(files[i], files[j]))

    track_graphs[track] = G

    with open(f"{graph_dir}/{track}.json", "w") as f:
        json.dump(nx.node_link_data(G), f)
print("DONE")
