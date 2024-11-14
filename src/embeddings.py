import redis
import h5py
import pandas as pd
from rich import print
from rich.progress import track

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

n_rows = 0
with h5py.File('../notebooks/data/clamp_embeddings.h5', 'w') as hdf:
    for key in track(r.keys('files:*'), "downloading embeddings"): # type: ignore
        embedding = r.json().get(key, "$.clamp")
        if embedding is not None:
            hdf.create_dataset(key.split(':')[-1], data=embedding[0])
            n_rows += 1
        else:
          print(f"[orange]no embedding found for '{key}'")

print(f"{n_rows} rows have been written to notebooks/data/clamp_embeddings.h5")

# with h5py.File('../notebooks/data/clamp_embeddings.h5', 'w') as hdf:
#     keys = r.keys('files:*')
#     all_embeddings = hdf.create_dataset('clamp_embeddings', (len(keys), 768), dtype='float32')
#     for i, key in track(enumerate(keys), "downloading embeddings"): # type: ignore
#         embedding = r.json().get(key, "$.clamp")
#         if embedding is not None:
#             all_embeddings[i] = embedding[0]  # Append the new embedding
#             n_rows += 1
#         else:
#             print(f"[orange]no embedding found for '{key}'")
# print(f"{n_rows} rows have been written to notebooks/data/clamp_embeddings.h5")
