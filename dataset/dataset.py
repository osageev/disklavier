import os
import random
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

import numpy as np

from rich.progress import track
from rich import print
from rich.pretty import pprint


def set_tempo(input_file_path, target_tempo):
    mid = MidiFile(input_file_path)
    tempo = mido.bpm2tempo(target_tempo)
    mid.tracks[0].insert(0, MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.save(input_file_path)


def get_tempo_from_midi(midi_file_path):
    """"""
    midi_file = MidiFile(midi_file_path)

    # Default MIDI tempo is 120 BPM, which equals 500000 microseconds per beat
    tempo = 500000  # Default tempo

    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        if tempo != 500000:
            break

    return mido.tempo2bpm(tempo)


def vertical_shift(array, name: str, num_iterations: int = 1):
    """vertically shift an image
    NOTE: in some sense, up and down is flipped.
    """
    shifted_images = []

    def find_non_zero_bounds(arr):
        """Find the first and last row index with a non-zero element"""
        rows_with_non_zero = np.where(arr.any(axis=1))[0]
        return rows_with_non_zero[0], rows_with_non_zero[-1]

    def shift_array(arr, up=0, down=0):
        """Shift array vertically within bounds"""
        if up > 0:
            arr = np.roll(arr, -up, axis=0)
            arr[-up:] = 0
        elif down > 0:
            arr = np.roll(arr, down, axis=0)
            arr[:down] = 0
        return arr

    highest, lowest = find_non_zero_bounds(array)
    maximum_up = highest
    maximum_down = array.shape[0] - lowest - 1

    for _ in range(num_iterations):
        # Shift up and then down, decreasing the shift amount in each iteration
        for i in range(maximum_up, 0, -1):
            new_key = f"{Path(name).stem}_u{i:02d}"
            shifted_images.append((new_key, np.copy(shift_array(array, up=i))))
        for i in range(maximum_down, 0, -1):
            new_key = f"{Path(name).stem}_d{i:02d}"
            shifted_images.append((new_key, np.copy(shift_array(array, down=i))))

    random.shuffle(shifted_images)

    return shifted_images[:num_iterations]


def segment_midi(input_file_path: str, params) -> int:
    """"""
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    if not params.strip_tempo:
        set_tempo(input_file_path, target_tempo)

    # remove "-t" from filename
    filename = Path(input_file_path).stem
    filename_components = filename.split('-')
    filename = f"{filename_components[0]}-{int(np.round(float(filename_components[1]))):03d}-{filename_components[2]}"

    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = 60 * params.n__num_beats / target_tempo
    num_segments_float = total_length / segment_length
    num_segments = int(np.round(num_segments_float))

    init_bpm = get_tempo_from_midi(input_file_path)
    print(
        f"breaking '{filename}' ({total_length:08.03f} s) into {num_segments:03d} segments of {segment_length:.03f} s at {int(np.round(init_bpm)):03d} BPM"
    )

    # for start in np.arange(0, int(total_length), segment_length)[:params.limit]:
    #     end = start + segment_length
    for n in range(num_segments):
        start = n * segment_length
        end = start + segment_length
        # print(f"\tsplitting from {start:07.03f} s to {end:07.03f} s")
        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
        instrument = pretty_midi.Instrument(program=midi_pm.instruments[0].program, name=f"{filename}_{int(start):04d}-{int(end):04d}")

        # add notes from the original MIDI that fall within the current segment
        for note in midi_pm.instruments[0].notes:
            if start <= note.start < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=min(note.end, end) - start
                )
                instrument.notes.append(new_note)

        # write out
        segment_midi.instruments.append(instrument)
        segment_filename = os.path.join(params.output_dir, f"{filename}_{int(start):04d}-{int(end):04d}.mid")
        segment_midi.write(segment_filename)

        if params.strip_tempo:
            midi_md = MidiFile(segment_filename)
            for track in midi_md.tracks:
                for message in track:
                    if message.type == 'set_tempo':
                        track.remove(message)
            midi_md.save(segment_filename)
        else:
            set_tempo(segment_filename, target_tempo)

    return num_segments


if __name__=="__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", default=None, help="location of MIDI files"
    )
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
        "-n"
        "--num_beats", 
        type=int,
        default=8,
        help="number of beats each segment should have"
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
        action="store_true",
        help="generate a segment for each possible semitone shift",
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

    if args.limit is None:
        dataset = os.listdir(args.data_dir)
    else:
        dataset = os.listdir(args.data_dir)[:args.limit]

    # segment files
    total_segs = 0
    for filename in track(dataset, description="generating segments"):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            total_segs += segment_midi(os.path.join(args.data_dir, filename), args)
    print(f"[green]segmentation complete, {total_segs} files generated")
