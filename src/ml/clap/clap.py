import os
import h5py
import faiss
import laion_clap
import numpy as np
from glob import glob
from datetime import datetime

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

start_time = datetime.now()
sf_name = "sgm"
input_path = f"/media/scratch/sageev-midi/20250320/wavs-{sf_name}"
output_path = f"/media/scratch/sageev-midi/20250320/clap-{sf_name}.h5"
checkpoint_path = "checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"
audio_files = list(glob(os.path.join(input_path, "*.wav")))
audio_files.sort()
num_files = len(audio_files)
print(audio_files[:3])

faiss_path = os.path.join(os.path.dirname(output_path), f"clap-{sf_name}.faiss")
index = faiss.IndexFlatIP(512)
vecs = np.zeros((num_files, 512), dtype=np.float32)

model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt(checkpoint_path)

with h5py.File(output_path, "w") as out_file:
    # create output datasets
    d_embeddings = out_file.create_dataset("embeddings", (num_files, 512), fillvalue=0)
    d_filenames = out_file.create_dataset(
        "filenames",
        (num_files, 1),
        dtype=h5py.string_dtype(encoding="utf-8"),
        fillvalue="",
    )

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    emb_task = progress.add_task("embedding", total=num_files)
    with progress:
        for i, audio_file in enumerate(audio_files):
            embedding = model.get_audio_embedding_from_filelist(
                x=[audio_file], use_tensor=False
            )
            d_embeddings[i] = embedding
            d_filenames[i] = audio_file
            vecs[i] = embedding / np.linalg.norm(embedding, axis=0, keepdims=True)
            progress.advance(emb_task, 1)

    index.add(vecs)  # type: ignore
    faiss.write_index(index, faiss_path)

print(f"CLAP embeddings saved to '{output_path}'")
print(f"FAISS index saved to '{faiss_path}'")

keys = ["filenames", "embeddings"]
with h5py.File(f"/media/scratch/sageev-midi/20250320/clap-{sf_name}.h5", "r") as f:
    print(f"checking HDF5 file with keys {f.keys()} for keys: {keys}")
    for key in keys:
        if key not in f:
            print(f"key {key} not found in HDF5 file")

        print(f"{key}:")
        if key == "filenames":
            for filename in f[key][:5]:  # type: ignore
                print(f"\t{str(filename[0], 'utf-8')}")
        else:
            for val in f[key][:5]:  # type: ignore
                print(f"\t{val.shape}")

print(f"embeddings generated in {datetime.now() - start_time}")
