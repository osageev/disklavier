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

seed_rearrange = True
seed_remove = 0.25

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
specdiff = SpectrogramDiffusion(specdiff_config, verbose=True)

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()

chosen_files = np.random.choice(all_files, size=2, replace=False)
file_data = {}
for chosen_file in chosen_files:
    file_data[chosen_file] = specdiff.embed(chosen_file)
    file_data[chosen_file] /= np.linalg.norm(
        file_data[chosen_file], axis=1, keepdims=True
    )

for filepath, embedding in file_data.items():
    bpm = midi.get_bpm(filepath)
    console.log(f"augmenting '{filepath}' (bpm {bpm})")

    midi_paths = []
    if seed_rearrange:
        split_beats = midi.beat_split(filepath, bpm)
        console.log(f"\tsplit '{basename(filepath)}' into {len(split_beats)} beats")
        ids = list(range(len(split_beats)))
        rearrangements: list[list[int]] = [
            ids,  # original
            ids[1 : len(ids) // 2],  # first half
            ids[1 : len(ids) // 2],  # first half (will have random notes added after)
            ids[1 : len(ids) // 2] * 2,  # first half twice
            ids[len(ids) // 2 + 1 :],  # second half
            ids[
                len(ids) // 2 + 1 :
            ],  # second half (will have random notes added before)
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
            if i == 2:
                random_notes = midi.generate_random_midi(4, bpm)
                md_midi_joined = mido.MidiFile(pf_joined_midi)
                md_midi = mido.MidiFile(ticks_per_beat=220)
                clean_track = mido.MidiTrack()
                md_midi.tracks.append(clean_track)

                for note in random_notes:
                    clean_track.append(note)

                print(md_midi_joined.print_tracks())
                raise Exception("stop")
                for note in md_midi_joined.tracks[1]:
                    clean_track.append(note)

                print(md_midi.print_tracks())

                md_midi.save(pf_joined_midi)
            elif i == 5:
                random_notes = midi.generate_random_midi(4, bpm)
                md_midi_joined = mido.MidiFile(pf_joined_midi)

                md_midi_joined.tracks[1].extend(random_notes)

                print(md_midi_joined.print_tracks())

                md_midi_joined.save(pf_joined_midi)
            console.log(f"\tjoined midi:\t{joined_midi.get_end_time()}")
            midi_paths.append(pf_joined_midi)
    else:
        midi_paths.append(filepath)

    if seed_remove:
        joined_paths = midi_paths
        midi_paths = []  # TODO: stop overloading this
        num_options = 0
        for mid in joined_paths:
            stripped_paths = midi.remove_notes(mid, pf_augmentations, seed_remove)
            console.log(
                f"\tstripped notes from '{basename(mid)}' (+{len(stripped_paths)})"
            )
            midi_paths.append(stripped_paths)
            num_options += len(stripped_paths)
        console.log(f"\taugmented '{basename(filepath)}' into {num_options} files")

        best_aug = ""
        best_path = []
        best_match = ""
        best_similarity = 0.0
        for ps in midi_paths:
            console.log(f"\tps: {ps}")
            for m in ps:
                embedding = seeker.get_embedding(m)
                match, similarity = seeker.get_match(embedding)
                console.log(
                    f"\t\tbest match for '{basename(m)}' is '{basename(best_match)}' with similarity {best_similarity}"
                )
                if similarity > best_similarity:
                    best_aug = m
                    best_path = ps
                    best_match = match
                    best_similarity = similarity
    else:
        best_aug = ""
        best_match = ""
        best_similarity = 0.0
        for ps in midi_paths:
            for m in ps:
                console.log(f"\t\t{basename(m)}")

        # find best match for each augmentation
        for mid in midi_paths:
            embedding = specdiff.embed(mid)
            match, similarity = seeker.get_match(embedding)
            console.log(
                f"\t\tbest match for '{basename(mid)}' is '{basename(best_match)}' with similarity {best_similarity}"
            )
            if similarity > best_similarity:
                best_aug = mid
                best_match = match
                best_similarity = similarity

    console.log(
        f"\tbest augmentation is '{basename(best_aug)}' with similarity {best_similarity} matches to {best_match}"
    )

    if seed_remove:
        # add em all up
        midi_paths = [
            *best_path[: best_path.index(best_aug) + 1],
            os.path.join(DATA_DIR, best_match + ".mid"),
        ]
        console.log(f"\t\tbp: {best_path}")
        console.log(f"\t\tmp: {midi_paths}")

os.rmdir(pf_augmentations)
