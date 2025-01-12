import os
import zipfile
from argparse import ArgumentParser
import mido
import pretty_midi
from PIL import Image
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from utils import console

from typing import List

PR_FS = 50
N_BEATS = 9


def get_bpm(file_path: str) -> int:
    """
    Extracts the bpm from a MIDI file.

    Args:
        file_path (str): Path to the MIDI file.

    Returns:
        int: The BPM. Default is 120 BPM if not explicitly set.
    """
    try:
        tempo = int(os.path.basename(file_path).split("-")[1])
    except ValueError:
        tempo = 120
        midi_file = mido.MidiFile(file_path)
        for track in midi_file.tracks:
            for message in track:
                if message.type == "set_tempo":
                    tempo = mido.tempo2bpm(message.tempo)

    return tempo


def calculate_piano_roll_columns(tempo: int, fs: int = 100, beats: int = 9) -> int:
    """
    Calculate the number of columns in a piano roll for a given tempo and duration.

    Args:
        tempo (int): Tempo in beats per minute (BPM).
        fs (int): Sampling frequency in Hz (frames per second). Default is 100.
        beats (int): Number of beats for the duration. Default is 9.

    Returns:
        int: Number of columns in the piano roll.
    """
    # Calculate duration of one beat in seconds
    seconds_per_beat = 60 / tempo

    # Calculate total duration for the given number of beats
    total_duration = beats * seconds_per_beat

    # Calculate the number of columns for the given sampling frequency
    num_columns = int(total_duration * fs)

    return num_columns

def process_files(pf_files: List[str], p_outputs: str, index: int = 1) -> int:
    """
    Generate piano rolls from MIDI files with pixel-perfect resolution,
    trimming them to a calculated number of columns, cropping to 128 columns,
    zero-padding vertically to 128 rows, and saving with the correct folder structure.

    Arguments
    --------
    pf_files : List[str]
        paths of all files to convert
    p_outputs : str
        top level of directory to write outputs to
    index : int, default=1
        index of process, used for printing progress

    Returns
    --------
    int
        process index, again just used for printing
    """

    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=0.1,
    )
    task_s = p.add_task(f"[SUBR{index:02d}] segmenting", total=len(pf_files))

    with p:
        for pf_file in pf_files:
            try:
                # Derive relative path and create corresponding output path
                input_root = os.path.commonpath(pf_files)  # Determine the common root
                rel_path = os.path.relpath(pf_file, start=input_root)
                output_path = os.path.join(p_outputs, rel_path)
                output_path = os.path.splitext(output_path)[0] + ".png"  # Change extension to .png
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)

                # Load MIDI file and generate piano roll
                midi = pretty_midi.PrettyMIDI(pf_file)
                piano_roll = midi.get_piano_roll(fs=PR_FS)

                # Trim the piano roll to the number of columns defined by tempo and beats
                tempo = get_bpm(pf_file)
                num_columns = calculate_piano_roll_columns(
                    tempo, fs=PR_FS, beats=N_BEATS
                )
                trimmed_piano_roll = piano_roll[:, :num_columns]

                # Trim the piano roll to start from the first note
                non_empty_columns = np.any(piano_roll > 0, axis=0)
                first_note_idx = np.argmax(non_empty_columns)
                trimmed_piano_roll = piano_roll[:, first_note_idx:].astype(np.uint8)

                # Randomly select a 128-column range
                total_columns = trimmed_piano_roll.shape[1]
                if total_columns > 128:
                    start_idx = random.randint(0, total_columns - 128)
                    cropped_piano_roll = trimmed_piano_roll[
                        :, start_idx : start_idx + 128
                    ]
                else:
                    cropped_piano_roll = trimmed_piano_roll

                # Zero-pad vertically to ensure 128 rows
                current_rows = cropped_piano_roll.shape[0]
                if current_rows < 128:
                    padded_piano_roll = np.zeros(
                        (128, cropped_piano_roll.shape[1]), dtype=np.uint8
                    )
                    padded_piano_roll[128 - current_rows :, :] = cropped_piano_roll
                else:
                    padded_piano_roll = cropped_piano_roll

                # Save the resized piano roll as an image
                img = Image.fromarray(padded_piano_roll, mode="L")
                img.save(output_path)

                p.update(task_s, advance=1)
            except Exception as e:
                console.log(f"[red]Error processing {pf_file}: {e}")

    return index


def main(args):
    # set up filesystem
    if not os.path.exists(args.data_dir):
        console.log(f"no data dir found at {args.data_dir}")
        raise IsADirectoryError

    # p_outputs = os.path.join(args.out_dir, "piano_rolls")

    # collect all paths
    tracks = []
    for path, _, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".mid") or filename.endswith(".midi"):
                tracks.append(os.path.join(path, filename))
    np.random.shuffle(tracks)
    # tracks.sort()
    if args.limit is not None:
        tracks = tracks[: args.limit]
    num_process = os.cpu_count()
    if num_process is None:
        print(f"Unable to count CPUs. Defaulting to 1 process.")
        num_process = 1
    split_keys = np.array_split(tracks, num_process)

    # do the work
    console.log(f"generating {len(tracks)} piano rolls")
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_files,
                list(chunk),
                args.out_dir,
                index=i,
            ): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            console.log(f"subprocess {index} returned")

    n_files = 0
    if args.zip:
        zip_path = os.path.join(args.out_dir, f"{args.dataset_name}_prs.zip")
        console.log(f"compressing to zipfile '{zip_path}'")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for path, _, files in os.walk(args.data_dir):
                for file in files:
                    file_path = os.path.join(path, file)
                    arcname = os.path.relpath(file_path, args.data_dir)
                    zipf.write(file_path, arcname)
                    n_files += 1

    console.log(
        f"[green bold]piano roll generation complete, {n_files} files generated"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="path to read MIDI files from"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path to write image files to"
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="the name of the dataset"
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "-z",
        "--zip",
        action="store_true",
        help="make a zip file of all generated files",
    )
    args = parser.parse_args()
    console.log(args)

    main(args)
