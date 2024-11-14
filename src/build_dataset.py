import os
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from mido import MidiFile, MetaMessage, second2tick
import mido
import pretty_midi
import numpy as np
from rich import print
from rich.pretty import pprint
from rich.progress import track

from typing import List, Tuple


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


def set_tempo(input_file_path, target_tempo) -> None:
    """
    Sets the tempo of a MIDI file to a specified target tempo.

    This function modifies the input MIDI file by inserting a "set_tempo" meta message
    at the beginning of the first track, setting the tempo to the specified target tempo.

    Args:
        input_file_path (str): Path to the input MIDI file.
        target_tempo (int): The target tempo in beats per minute (BPM) to set in the MIDI file.
    """
    mid = MidiFile(input_file_path)
    tempo = mido.bpm2tempo(target_tempo)
    mid.tracks[0].insert(0, MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.save(input_file_path)


def get_note_min_max(input_file_path) -> Tuple[int, int]:
    """returns the values of the highest and lowest notes in a midi file"""
    mid = MidiFile(input_file_path)

    lowest_note = 127
    highest_note = 0

    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta and msg.type in ["note_on", "note_off"]:
                # Update lowest and highest note if this is a note_on message
                if msg.velocity > 0:  # Considering note_on messages only
                    lowest_note = min(lowest_note, msg.note)
                    highest_note = max(highest_note, msg.note)

    return (lowest_note, highest_note)


def semitone_transpose(
    midi_path: str, output_dir: str, num_iterations: int = 1
) -> List[str]:
    """
    Transposes a MIDI file by a specified number of semitones, alternating between up and down.

    This function takes a MIDI file path, an output directory, and an optional number of iterations.
    It calculates the maximum possible transpositions up and down without exceeding the MIDI note range.
    Then, it iterates the specified number of times, alternating between transposing up and down by one semitone each iteration.
    If the maximum up or down transposition is exceeded, it switches to the opposite direction immediately.
    The transposed MIDI files are saved in the specified output directory with a modified filename indicating the direction and amount of transposition.

    Args:
        midi_path (str): Path to the input MIDI file.
        output_dir (str): Directory where the transposed MIDI files will be saved.
        num_iterations (int, optional): Number of iterations to transpose the MIDI file. Defaults to 1.

    Returns:
        List[str]: A list of file paths to the newly created transposed MIDI files.
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


def segment_midi(input_file_path: str, params):
    """
    Segments a MIDI file into smaller segments based on the provided parameters.

    Args:
        input_file_path (str): Path to the input MIDI file.
        params (object): Object containing parameters for segmenting the MIDI file.

    Returns:
        int: The number of new files created after segmenting the MIDI file.
    """
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    set_tempo(input_file_path, target_tempo)

    # remove "-t" from filename
    filename = Path(input_file_path).stem
    filename_components = filename.split("-")
    filename = f"{filename_components[0]}-{int(np.round(float(filename_components[1]))):03d}-{filename_components[2]}"

    # calculate timings
    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = 60 * params.n__num_beats / target_tempo  # in seconds
    num_segments_float = total_length / segment_length
    num_segments = int(np.round(num_segments_float))

    new_files = 0
    for n in list(range(num_segments)):
        start = n * segment_length
        end = start + segment_length
        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{filename}_{int(start):04d}-{int(end):04d}",
        )

        # add notes from the original MIDI that fall within the current segment
        for note in midi_pm.instruments[0].notes:
            if start <= note.start < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=min(note.end, end) - start,
                )
                instrument.notes.append(new_note)

        # write out
        segment_midi.instruments.append(instrument)
        segment_filename = os.path.join(
            params.output_dir, f"{filename}_{int(start):04d}-{int(end):04d}_n00.mid"
        )
        segment_midi.write(segment_filename)

        # semitone shift
        if params.do_shift > 1:
            tpose_files = semitone_transpose(
                segment_filename, params.output_dir, params.do_shift
            )
            new_files += len(tpose_files)
        else:
            new_files += 1

        for filepath in tpose_files:
            # set tempo properly
            if params.strip_tempo:
                midi_md = MidiFile(filepath)
                for track in midi_md.tracks:
                    for message in track:
                        if message.type == "set_tempo":
                            track.remove(message)
                            track.append(
                                MetaMessage(
                                    "set_tempo",
                                    tempo=mido.bpm2tempo(target_tempo),
                                    time=0,
                                )
                            )
                os.remove(filepath)
                midi_md.save(filepath)
            else:
                set_tempo(filepath, target_tempo)

            # make sure track end is correct
            modify_end_of_track(filepath, segment_length, target_tempo)
            test_mid = MidiFile(filepath)

    return new_files


def modify_end_of_track(midi_file_path, new_end_time, tempo):
    """
    This function modifies the end of a MIDI track by removing existing 'end_of_track' messages
    and adding a new 'end_of_track' message at the calculated offset time.

    Args:
        midi_file_path (str): The path to the MIDI file to be modified.
        new_end_time (float): The new end time of the track in seconds.
        tempo (int): The tempo of the track in beats per minute.

    Returns:
        None
    """
    mid = MidiFile(midi_file_path)
    total_time_t = -1
    new_e_time_t = second2tick(new_end_time, 220, mido.bpm2tempo(tempo))

    for track in mid.tracks:
        # Remove existing 'end_of_track' messages and calculate last note time
        for msg in track:
            total_time_t += msg.time
            if msg.type == "end_of_track":
                track.remove(msg)
                # Add a new 'end_of_track' message at the calculated offset time
                offset = (
                    new_e_time_t - total_time_t if new_e_time_t > total_time_t else 0
                )
                track.append(MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    os.remove(midi_file_path)
    mid.save(midi_file_path)


def main(args):
    # set up filesystem
    if not os.path.exists(args.data_dir):
        print(f"no data dir found at {args.data_dir}")
        exit()
    if os.path.exists(args.output_dir):
        i = 0
        for i, file in enumerate(os.listdir(args.output_dir)):
            os.remove(os.path.join(args.output_dir, file))
            i += 1
        print(f"cleaned {i} files out of output folder: '{args.output_dir}'")
    else:
        print(f"creating new output folder: '{args.output_dir}'")
        os.mkdir(args.output_dir)

    graveyard = os.path.join("outputs", "graveyard")
    if os.path.exists(graveyard):
        i = 0
        for i, file in enumerate(os.listdir(graveyard)):
            os.remove(os.path.join(graveyard, file))
            i += 1
        print(f"cleaned {i} files out of graveyard: '{graveyard}'")
    else:
        print(f"creating new graveyard: '{graveyard}'")
        os.mkdir(graveyard)

    if args.limit is None:
        dataset = os.listdir(args.data_dir)
    else:
        dataset = os.listdir(args.data_dir)[: args.limit]

    # segment files
    num_files = 0
    for filename in track(dataset, description="generating segments"):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            num_files += segment_midi(os.path.join(args.data_dir, filename), args)

    print(f"[green]segmentation complete, {num_files} files generated")


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--output_dir", default=None, help="location to write segments to"
    )
    parser.add_argument(
        "-m",
        "--store_metrics",
        default=f"metrics-{datetime.now().strftime('%y%m%d-%H%M%S')}.json",
        help="file to write segment metrics to (must be JSON)",
    )
    parser.add_argument(
        "-n" "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have",
    )
    parser.add_argument(
        "-t",
        "--strip_tempo",
        action="store_true",
        help="strip all tempo messages from files",
    )
    parser.add_argument(
        "-s",
        "--do_shift",
        type=int,
        default=12,
        help="generate a segment for a number of semitone shifts",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    args = parser.parse_args()
    pprint(args)

    main(args)
