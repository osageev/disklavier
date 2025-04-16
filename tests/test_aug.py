import os
import sys

import mido

# Add project root to sys.path instead of src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# sys.path.append(
#     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
# )

import faiss
import pretty_midi
import numpy as np
from glob import glob

from src import utils
from src.ml.specdiff.model import SpectrogramDiffusion, config as specdiff_config
from src.utils import basename, console, midi

np.random.seed(0)

FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
pf_augmentations = os.path.join("outputs", "augmentations")
os.makedirs(pf_augmentations, exist_ok=True)

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
specdiff = SpectrogramDiffusion(specdiff_config, verbose=True)

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()

chosen_files = np.random.choice(all_files, size=1, replace=False)
file_data = {}
for chosen_file in chosen_files:
    bpm = midi.get_bpm(chosen_file)
    new_path = os.path.join(pf_augmentations, f"{basename(chosen_file)}_")

    # add notes to start of file
    random_notes = midi.generate_random_midi(2, bpm)
    md_midi_joined = mido.MidiFile(chosen_file)
    md_midi = mido.MidiFile(ticks_per_beat=220)
    clean_track = mido.MidiTrack()
    md_midi.tracks.append(clean_track)

    for note in random_notes:
        clean_track.append(note)

    for note in md_midi_joined.tracks[1]:
        if note.type == "note_on" or note.type == "note_off":
            clean_track.append(note)

    start_path = new_path + "_start.mid"
    md_midi.save(start_path)
    file_data[start_path] = specdiff.embed(start_path)
    file_data[start_path] /= np.linalg.norm(
        file_data[start_path], axis=1, keepdims=True
    )

    # add notes to end of file
    random_notes = midi.generate_random_midi(2, bpm)
    md_midi_joined = mido.MidiFile(chosen_file)

    for track in md_midi_joined.tracks:
        for msg in track:
            if msg.type == "end_of_track":
                track.remove(msg)

    md_midi_joined.tracks[1].extend(random_notes)
    md_midi_joined.tracks[1].append(mido.MetaMessage("end_of_track", time=0))

    end_path = new_path + "_end.mid"
    md_midi_joined.save(end_path)
    file_data[end_path] = specdiff.embed(end_path)
    file_data[end_path] /= np.linalg.norm(file_data[end_path], axis=1, keepdims=True)

print(file_data.keys())

for filepath, embedding in file_data.items():
    bpm = midi.get_bpm(filepath)
    console.log(f"augmenting '{filepath}' (bpm {bpm})")

    midi_paths = []
    split_beats = midi.beat_split(filepath, bpm)
    console.log(f"\tsplit '{basename(filepath)}' into {len(split_beats)} beats")
    ids = list(range(len(split_beats)))
    rearrangements: list[list[int]] = [
        ids,  # original
        ids[1 : len(ids) // 2],  # first half
        ids[1 : len(ids) // 2] * 2,  # first half twice
        ids[len(ids) // 2 + 1 :],  # second half
        ids[len(ids) // 2 + 1 :] * 2,  # second half twice
    ]
    for i, arrangement in enumerate(rearrangements):
        console.log(f"\trearranging seed:\t{arrangement}")
        joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
            split_beats, arrangement, bpm
        )

        pf_joined_midi = os.path.join(
            pf_augmentations, f"{basename(filepath)}_a{i:02d}.mid"
        )
        joined_midi.write(pf_joined_midi)
        console.log(f"\tjoined midi:\t{joined_midi.get_end_time()}")
        midi_paths.append(pf_joined_midi)

    # find best match for each augmentation
    best_aug = ""
    best_match = ""
    best_similarity = 0.0
    for path in midi_paths:
        console.log(f"\t\t{basename(path)}")
    for mid in midi_paths:
        embedding = specdiff.embed(mid)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        similarities, indices = faiss_index.search(embedding, 10)
        match = all_files[indices[0][0]]
        similarity = similarities[0][0]
        console.log(
            f"\t\tbest match for '{basename(mid)}' is '{basename(match)}' with similarity {similarity}"
        )
        if similarity > best_similarity:
            best_aug = mid
            best_match = match
            best_similarity = similarity

    console.log(
        f"\tbest augmentation is '{basename(best_aug)}' with similarity {best_similarity} matches to {best_match}"
    )
