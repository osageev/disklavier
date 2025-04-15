import os
import random
import numpy as np
import pretty_midi
from glob import glob
from rich.pretty import pprint


from ml.specdiff.model import SpectrogramDiffusion, config as specdiff_config

data_dir = "data/datasets/test/synthetic"
table_path = "data/datasets/test/specdiff"
all_files = glob(os.path.join(data_dir, "*.mid"))
og_files = [f for f in all_files if not f.endswith("+2.mid")]
# for file in og_files:
#     new_path = file.replace(".mid", "+2.mid")
#     midi = pretty_midi.PrettyMIDI(file)
#     for instrument in midi.instruments:
#         for note in instrument.notes:
#             note.pitch += 24
#     midi.write(new_path)
#     all_files.append(new_path)
pprint(all_files)
pprint(og_files)

specdiff = SpectrogramDiffusion(specdiff_config, verbose=True)

# randomly select two files from the list
selected_files = random.sample(og_files, 5)
pprint(f"randomly selected files: {selected_files}")

# get the embeddings for the selected files
embeddings = [specdiff.embed(f) for f in selected_files]

# get the distance between the embeddings
proxy_base_file = selected_files[-1].replace(".mid", "+2.mid")
proxy_base_embed = specdiff.embed(proxy_base_file)
two_oct_dist = np.linalg.norm(np.abs(embeddings[-1] - proxy_base_embed))
pprint(f"proxy base file: {proxy_base_file}")
for i in range(len(embeddings) - 1):
    proxy_file = selected_files[i].replace(".mid", "+2.mid")
    proxy_embed = specdiff.embed(proxy_file)
    pprint(f"analyzing distance between {selected_files[i]} and {selected_files[-1]}")
    dtd = np.abs(np.linalg.norm(np.abs(embeddings[i] - proxy_embed)) - two_oct_dist)
    pprint(f"dummy test distance: {dtd}")
    
    norm_distance = np.linalg.norm(embeddings[i] - embeddings[-1])
    pprint(f"norm distance: {norm_distance}")
    diff_vector = (proxy_embed + norm_distance) - proxy_base_embed
    magnitude = np.linalg.norm(diff_vector)
    direction = diff_vector / magnitude if magnitude > 0 else diff_vector
    pprint(f"distance from pf + d to pb is {magnitude:.3f}")
    
    cosine_distance = np.dot(embeddings[i], embeddings[-1].T) / (
        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[-1])
    )
    pprint(f"cosine distance: {cosine_distance}")
    diff_vector = (proxy_embed + cosine_distance) - proxy_base_embed
    magnitude = np.linalg.norm(diff_vector)
    direction = diff_vector / magnitude if magnitude > 0 else diff_vector
    pprint(f"distance from pf + d to pb is {magnitude:.3f}")

    difference = np.abs(embeddings[i] - embeddings[-1])
    pprint(f"difference")
    diff_vector = (proxy_embed + difference) - proxy_base_embed
    magnitude = np.linalg.norm(diff_vector)
    direction = diff_vector / magnitude if magnitude > 0 else diff_vector
    pprint(f"distance from pf + d to pb is {magnitude:.3f}")
