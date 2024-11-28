import os
import csv
import uuid
import zipfile
from shutil import copy2
from argparse import ArgumentParser
import mido
import pretty_midi
from itertools import product
import numpy as np
from pathlib import Path
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import console

from typing import Dict, Tuple, List


def generate_unique_uuid(pf_uuid_map: str, pf_file: str) -> str:
    """Generates a unique UUID for the track, ensuring it doesn't already exist in the CSV file."""
    existing_uuids = set()
    if os.path.exists(pf_uuid_map):
        with open(pf_uuid_map, "r") as file:
            reader = csv.DictReader(file)
            existing_uuids = {row["uuid"] for row in reader}

    while True:
        track_uuid = str(uuid.uuid3(uuid.NAMESPACE_URL, pf_file)).split("-")[0]
        if track_uuid not in existing_uuids:
            return track_uuid


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


def set_bpm(input_file_path: str, bpm: int) -> None:
    """Sets the tempo of a MIDI file to a specified target tempo, provided as a bpm.

    Args:
        input_file_path (str): The path to the MIDI file whose tempo is to be adjusted.
        bpm (int): The target tempo in beats per minute (BPM) to set for the MIDI file.

    This function modifies the specified MIDI file by inserting a tempo change meta-message at the beginning of the first track, effectively setting the entire file to the specified tempo. The change is saved to the same file path, overwriting the original MIDI file.
    """
    midi = mido.MidiFile(input_file_path)
    midi.tracks[0].insert(
        0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
    )
    midi.save(input_file_path)


def change_tempo(in_path: str, out_path: str, bpm: int):
    midi = mido.MidiFile(in_path)
    new_message = mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
    tempo_added = False

    for i, track in enumerate(midi.tracks):
        # remove existing set_tempo messages
        tempo_messages = []
        for j, msg in enumerate(track):
            if msg.type == "set_tempo":
                tempo_messages.append(j)

        for index in tempo_messages:
            midi.tracks[i][index] = new_message

        # add new set_tempo message to the first track
        if not tempo_added:
            track.insert(0, new_message)
            tempo_added = True

    # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
    if not tempo_added:
        new_track = mido.MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(out_path)


def semitone_transpose(
    midi_path: str, output_dir: str, num_iterations: int = 1
) -> List[str]:
    """vertically shift a matrix
    chatgpt
    """
    new_filename = Path(midi_path).stem.split("_")
    new_filename = f"{new_filename[0]}_{new_filename[1]}"
    lowest_note, highest_note = get_note_min_max(midi_path)
    max_up = 108 - highest_note  # TODO double-check this IRL
    max_down = lowest_note

    # zipper up & down
    up = 1
    down = -1
    new_files = []
    for i in range(num_iterations):
        up_filename = f"{new_filename}_u{up:02d}.mid"
        up_filepath = os.path.join(output_dir, up_filename)
        down_filename = f"{new_filename}_d{abs(down):02d}.mid"
        down_filepath = os.path.join(output_dir, down_filename)

        if i % 2 == 0:
            if (
                up > max_up
            ):  # If exceeding max_up, adjust by switching to down immediately
                transpose_midi(midi_path, down_filepath, down)
                new_files.append(down_filepath)
                down -= 1
            else:
                transpose_midi(midi_path, up_filepath, up)
                new_files.append(up_filepath)
                up += 1
        else:
            if (
                abs(down) > max_down
            ):  # If exceeding max_down, adjust by switching to up immediately
                transpose_midi(midi_path, up_filepath, up)
                new_files.append(up_filepath)
                up += 1
            else:
                transpose_midi(midi_path, down_filepath, down)
                new_files.append(down_filepath)
                down -= 1

    return new_files


def transpose_midi(input_file_path: str, output_file_path: str, semitones: int) -> None:
    """
    Transposes all the notes in a MIDI file by a specified number of semitones.

    Args:
    - input_file_path: Path to the input MIDI file.
    - output_file_path: Path where the transposed MIDI file will be saved.
    - semitones: Number of semitones to transpose the notes. Positive for up, negative for down.
    """

    midi = pretty_midi.PrettyMIDI(input_file_path)
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += semitones
    midi.write(output_file_path)


def get_note_min_max(input_file_path) -> Tuple[int, int]:
    """returns the values of the highest and lowest notes in a midi file"""
    mid = mido.MidiFile(input_file_path)
    lowest_note = 127
    highest_note = 0

    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta and msg.type in ["note_on", "note_off"]:
                if msg.velocity > 0:
                    lowest_note = min(lowest_note, msg.note)
                    highest_note = max(highest_note, msg.note)

    return (lowest_note, highest_note)


def transform(
    file_path: str, out_dir: str, bpm: int, transformations: Dict, num_beats: int = 8
) -> str:
    new_filename = f"{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}"
    out_path = os.path.join(out_dir, f"{new_filename}.mid")
    mido.MidiFile(file_path).save(out_path)  # in case transpose is 0
    if transformations["transpose"] != 0:
        t_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

        for instrument in pretty_midi.PrettyMIDI(out_path).instruments:
            transposed_instrument = pretty_midi.Instrument(
                program=instrument.program, name=new_filename
            )

            for note in instrument.notes:
                transposed_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )

            t_midi.instruments.append(transposed_instrument)

        t_midi.write(out_path)

    if transformations["shift"] != 0:
        midi_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        seconds_per_beat = 60 / bpm
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 1) * seconds_per_beat

        for instrument in pretty_midi.PrettyMIDI(out_path).instruments:
            shifted_instrument = pretty_midi.Instrument(
                program=instrument.program, name=new_filename
            )
            for note in instrument.notes:
                dur = note.end - note.start
                shifted_start = (note.start + shift_seconds) % loop_point
                shifted_end = shifted_start + dur

                if note.start + shift_seconds >= loop_point:
                    shifted_start += seconds_per_beat
                    shifted_end += seconds_per_beat

                shifted_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=shifted_start,
                        end=shifted_end,
                    )
                )

            midi_pm.instruments.append(shifted_instrument)

        midi_pm.write(out_path)

    change_tempo(out_path, out_path, bpm)

    return out_path


def augment_midi(
    p: Progress, filename: str, new_segments: List[str], output_path: str
) -> List[str]:
    augmented_files = []
    task_a = p.add_task(f"augmenting {filename}", total=len(new_segments) * 12 * 8)

    with p:
        for segment_filename in new_segments:
            transformations = [
                {"transpose": t, "shift": s} for t, s in product(range(12), range(8))
            ]
            for transformation in transformations:
                augmented_files.append(
                    transform(
                        segment_filename,
                        output_path,
                        get_bpm(segment_filename),
                        transformation,
                    )
                )
                p.update(task_a, advance=1)
        p.remove_task(task_a)

    return augmented_files


def segment_midi(
    midi_file_path: str,
    file_uuid: str,
    output_dir: str,
    num_beats: int = 8,
) -> List[str]:
    # preserve tempo across all segments
    target_bpm = get_bpm(midi_file_path)
    set_bpm(midi_file_path, target_bpm)

    # calculate times and stuff
    midi_pm = pretty_midi.PrettyMIDI(midi_file_path)
    total_file_length_s = midi_pm.get_end_time()
    segment_length_s = num_beats * 60 / target_bpm
    n_segments = int(np.round(total_file_length_s / segment_length_s))
    pre_beat_window_s = (
        segment_length_s / num_beats / 8
    )  # to capture when first note is a bit early

    # console.log(
    #     f"\tbreaking '{filename}' ({total_file_length_s:.03f} s at {target_tempo} bpm) into {n_segments:03d} segments of {segment_length_s:.03f}s\n\t(pre window is {pre_beat_window_s:.03f} s)"
    # )

    new_files = []
    for n in list(range(n_segments)):
        start = n * segment_length_s
        end = start + segment_length_s - pre_beat_window_s
        if n > 0:
            start -= pre_beat_window_s

        # console.log(f"\t{n:03d} splitting from {start:08.03f} s to {end:07.03f} s")

        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_bpm)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{file_uuid}_{int(start):04d}-{int(end):04d}",
        )

        # add notes from the original MIDI that fall within the current segment
        for note in midi_pm.instruments[0].notes:
            if start <= note.start < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=note.end - start,
                )
                instrument.notes.append(new_note)

        # pad front of track to full bar for easier playback
        if n > 0:
            for note in instrument.notes:
                note.start += pre_beat_window_s * 7
                note.end += pre_beat_window_s * 7

        # write out
        segment_filename = os.path.join(
            output_dir, f"{file_uuid}_{int(start):04d}-{int(end):04d}.mid"
        )

        segment_midi.instruments.append(instrument)
        segment_midi.write(segment_filename)
        set_bpm(segment_filename, target_bpm)
        modify_end_of_track(segment_filename, segment_length_s, target_bpm)

        new_files.append(segment_filename)

    return new_files


def modify_end_of_track(midi_file_path: str, new_end_time: float, bpm: int) -> None:
    midi = mido.MidiFile(midi_file_path)
    new_end_time_t = mido.second2tick(new_end_time, 220, mido.bpm2tempo(bpm))

    for i, track in enumerate(midi.tracks):
        total_time_t = 0
        # Remove existing 'end_of_track' messages and calculate last note time
        for msg in track:
            if msg.type == "note_on":
                total_time_t += msg.time
            if msg.type == "end_of_track":
                track.remove(msg)
                # Add a new 'end_of_track' message at the calculated offset time
                offset = (
                    new_end_time_t - total_time_t
                    if new_end_time_t > total_time_t
                    else 0
                )
                track.append(mido.MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    os.remove(midi_file_path)
    midi.save(midi_file_path)


def process_files(
    pf_files: List[str], pf_uuid_map: str, p_segments: str, p_augments: str, index: int
) -> int:
    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=0.1,
    )
    task_s = p.add_task(f"[SUBR{index:02d}] segmenting", total=len(pf_files))

    # segment files
    segment_paths = []
    augment_paths = []
    with p:
        for pf_file in pf_files:
            # generate UUID for track and only use first part to keep filenames short
            track_uuid = generate_unique_uuid(pf_uuid_map, pf_file)
            with open(pf_uuid_map, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([os.path.basename(pf_file), track_uuid])

            # make folders to hold track segments and augments
            p_track_segments = os.path.join(p_segments, track_uuid)
            os.mkdir(p_track_segments)
            p_track_augments = os.path.join(p_augments, track_uuid)
            os.mkdir(p_track_augments)

            # segment
            if args.segment:
                new_segments = segment_midi(
                    os.path.join(args.data_dir, pf_file),
                    track_uuid,
                    p_track_segments,
                )
            else:
                copy2(
                    os.path.join(args.data_dir, pf_file),
                    os.path.join(p_track_segments, pf_file),
                )
                new_segments = [os.path.join(args.data_dir, pf_file)]
            segment_paths.extend(new_segments)

            # augment
            if args.augment:
                augment_paths.extend(
                    augment_midi(
                        p, os.path.splitext(pf_file)[0], new_segments, p_track_augments
                    )
                )

            p.update(task_s, advance=1)
    return index


def main(args):
    # set up filesystem
    if not os.path.exists(args.data_dir):
        console.log(f"no data dir found at {args.data_dir}")
        raise IsADirectoryError

    p_segments = os.path.join(args.out_dir, "segmented")
    p_augments = os.path.join(args.out_dir, "augmented")
    pf_uuid_map = os.path.join(args.out_dir, "uuid_map.csv")
    with open(pf_uuid_map, "w") as file:
        file.write("track,uuid\n")

    for dir in [p_segments, p_augments]:
        if not os.path.exists(dir):
            os.mkdir(dir)
            console.log(f"created new folder: '{dir}'")

    tracks = []
    for root, _, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".mid") or filename.endswith(".midi"):
                tracks.append(os.path.join(root, filename))
    tracks.sort()
    if args.limit is not None:
        tracks = tracks[: args.limit]
    split_keys = np.array_split(tracks, os.cpu_count())  # type: ignore

    console.log(f"segmenting {len(tracks)} tracks")
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_files,
                chunk,
                pf_uuid_map,
                p_segments,
                p_augments,
                index=i,
            ): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            console.log(f"subprocess {index} returned")

    zip_path = os.path.join("data", "datasets", f"{args.dataset_name}_segmented.zip")
    console.log(f"compressing to zipfile '{zip_path}'")
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.data_dir)
                zipf.write(file_path, arcname)
                n_files += 1

    console.log(f"[green bold]segmentation complete, {n_files} files generated")


if __name__ == "__main__":
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="path to read MIDI files from"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path to write MIDI files to"
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="the name of the dataset"
    )
    parser.add_argument(
        "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have, not including the leading and trailing sections of each segment",
    )
    parser.add_argument(
        "-t",
        "--strip_tempo",
        action="store_true",
        help="strip all tempo messages from files",
    )
    parser.add_argument(
        "-s",
        "--segment",
        action="store_true",
        help="generate a segment for a number of semitone shifts",
    )
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="augment dataset and store files",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    args = parser.parse_args()
    console.log(args)

    main(args)
