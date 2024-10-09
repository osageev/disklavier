import redis
import h5py
from rich import print
from rich.progress import track

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

n_rows = 0
skipped_rows = 0
with h5py.File('../notebooks/data/output.h5', 'w') as hdf:
    for key in track(r.keys('files:*')): # type: ignore
        embedding = r.json().get(key, "$.clamp")
        if embedding is not None:
            hdf.create_dataset(key.split(':')[-1], data=embedding[0])
            n_rows += 1
        else:
          print(f"[orange]no embedding found for '{key}'")
          skipped_rows += 1

print(f"{n_rows}/{skipped_rows} rows have been written to notebooks/data/output.h5")
