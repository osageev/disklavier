import os
import json
import h5py
import faiss
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

metric = "specdiff"
dataset = "20250110"
embeddings_file = f"data/tables/{dataset}/{metric}.h5"
index_file = f"data/tables/{dataset}/{metric}.faiss"
augmented_dir = f"data/datasets/{dataset}/augmented"
graph_dir = f"data/datasets/{dataset}/graphs"
top_k = 100

print("loading files")
with h5py.File(embeddings_file, "r") as f:
    embeddings = [e / np.linalg.norm(e, keepdims=True) for e in f["embeddings"]]  # type: ignore
    filenames = [str(name[0], "utf-8") for name in f["filenames"]]  # type: ignore
df = pd.DataFrame({"embeddings": embeddings}, index=filenames, dtype=np.float32)

faiss_index = faiss.read_index(index_file)

print("grouping files")
grouped_files = defaultdict(list)
for filename in os.listdir(augmented_dir):
    if filename.endswith(".mid"):
        track_name = filename.split("_")[0]
        grouped_files[track_name].append(filename.split(".")[0])

print("building graphs")
for track, files in tqdm(grouped_files.items()):
    print(f"building graphs for track {track}")

    # extract embeddings for current group of files
    emb_group = np.stack([df.loc[file, "embeddings"] for file in files])
    # compute cosine similarity matrix (dot product works since embeddings are normalized)
    similarity = emb_group.dot(emb_group.T)
    # exclude self-similarity by setting the diagonal to -infinity
    np.fill_diagonal(similarity, -np.inf)

    # for each node, get indices of top_k neighbors with highest cosine similarity
    neighbors = np.argsort(-similarity, axis=1)[:, :top_k]

    edges = []
    for i in range(len(files)):
        for j in neighbors[i]:
            if i < j:
                weight = 1 - similarity[i, j]
                edges.append((files[i], files[j], float(weight)))

    G = nx.Graph(name=track)
    G.add_nodes_from(files)
    G.add_weighted_edges_from(edges)

    with open(f"{graph_dir}/{track}.json", "w") as f:
        json.dump(nx.node_link_data(G), f)
    del G

print("DONE")
