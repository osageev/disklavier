import os
import random
import torch
import numpy as np
import pretty_midi
import tempfile
import shutil
from glob import glob
from rich.pretty import pprint

from src.utils import basename
from src.utils.midi import transpose_midi

from src.ml.specdiff.model import SpectrogramDiffusion

# --- configuration ---
DATA_DIR = "data/datasets/20250410/segmented"
NUM_FILES_TO_SELECT = 10
TRANSPOSE_SEMITONES = 12  # 1 octave up
NEIGHBORS_TO_FIND = 3


# --- main script ---
def main():
    print("\nInitializing Spectrogram Diffusion model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    specdiff = SpectrogramDiffusion({"device": device}, verbose=True)

    all_files = glob(os.path.join(DATA_DIR, "*.mid"))

    selected_files = random.sample(all_files, NUM_FILES_TO_SELECT)
    print("\nSelected Files:")
    pprint([os.path.basename(f) for f in selected_files])

    print(f"\nProcessing {len(selected_files)} files...")
    temp_dir = tempfile.mkdtemp()  # create a temporary directory for transposed files
    try:
        for i, file_path in enumerate(selected_files):
            print(
                f"\n--- Processing file {i+1}/{len(selected_files)}: {basename(file_path)} ---"
            )

            # 1. generate an embedding using specdiff
            print("  Generating embedding for original...")
            emb_original = specdiff.embed(file_path)

            # 2. transpose the midi up an octave and embed that
            print(f"  Transposing by {TRANSPOSE_SEMITONES} semitones...")
            transposed_filename = (
                f"{basename(file_path)}_transposed_{TRANSPOSE_SEMITONES}.mid"
            )
            transposed_file_path = os.path.join(temp_dir, transposed_filename)
            transpose_midi(file_path, transposed_file_path, TRANSPOSE_SEMITONES)

            print("  Generating embedding for transposed...")
            emb_transposed = specdiff.embed(transposed_file_path)

            # ensure embeddings are on the same device and float32 for calculations
            emb_original = emb_original.to(device).float()
            emb_transposed = emb_transposed.to(device).float()

            # 3. take the delta between the two vectors and print the magnitude and cosine difference
            delta = emb_original - emb_transposed
            magnitude = torch.linalg.norm(delta).item()
            # cosine similarity requires inputs of shape (batch_size, embedding_dim)
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_original.unsqueeze(0), emb_transposed.unsqueeze(0)
            ).item()
            cos_diff = 1 - cos_sim

            print(f"  Delta Magnitude: {magnitude:.4f}")
            print(f"  Cosine Difference: {cos_diff:.4f}")

    finally:
        # clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
