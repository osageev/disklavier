import os
import random
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

from utils.midi import set_tempo, get_tempo, semitone_shift

import numpy as np

from rich.progress import track
from rich import print
from rich.pretty import pprint


def segment_midi(input_file_path: str, params) -> int:
    """do the segmentation"""
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    set_tempo(input_file_path, target_tempo)

    # remove "-t" from filename
    filename = Path(input_file_path).stem
    filename_components = filename.split("-")
    filename = f"{filename_components[0]}-{int(np.round(float(filename_components[1]))):03d}-{filename_components[2]}"

    # figure out timings
    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = 60 * params.n__num_beats / target_tempo  # in seconds
    num_segments_float = total_length / segment_length
    num_segments = int(np.round(num_segments_float))
    init_bpm = get_tempo(input_file_path)

    pprint([total_length, segment_length, num_segments_float, num_segments, init_bpm])

    print(
        f"breaking '{filename}' ({total_length:.03f} s at {target_tempo} bpm) into {num_segments:03d} segments of {segment_length:.03f} s"
    )

    for n in range(num_segments):
        start = n * segment_length
        end = start + segment_length
        # print(f"\tsplitting from {start:07.03f} s to {end:07.03f} s")
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
            params.output_dir, f"{filename}_{int(start):04d}-{int(end):04d}.mid"
        )
        segment_midi.write(segment_filename)

        if params.strip_tempo:
            midi_md = MidiFile(segment_filename)
            for track in midi_md.tracks:
                for message in track:
                    if message.type == "set_tempo":
                        track.remove(message)
            midi_md.save(segment_filename)
        else:
            set_tempo(segment_filename, target_tempo)

        if params.do_shift:
            semitone_shift(segment_filename, params.output_dir, 12)

    return num_segments


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
        dataset = os.listdir(args.data_dir)[: args.limit]

    # segment files
    total_segs = 0
    for filename in track(dataset, description="generating segments"):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            total_segs += segment_midi(os.path.join(args.data_dir, filename), args)
    print(f"[green]segmentation complete, {total_segs} files generated")
