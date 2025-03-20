import os
import zipfile
from shutil import copy2
from argparse import ArgumentParser
import mido
import pretty_midi
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import basename, console
from utils.midi import get_bpm, set_bpm, transform, trim_piano_roll
from utils.novelty import gen_ssm_and_novelty
from typing import List

plt.style.use("dark_background")


class SegmentOptions:
    def __init__(
        self,
        num_beats: int,
        metronome: bool,
        drop_last: bool,
        novelty: bool,
        pic_dir: str,
    ):
        self.num_beats = num_beats
        self.metronome = metronome
        self.drop_last = drop_last
        self.novelty = novelty
        self.pic_dir = pic_dir


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
    output_dir: str,
    options: SegmentOptions,
) -> List[str]:
    """
    Break a MIDI file into segments.

    Parameters
    ----------
    midi_file_path : str
        path to the midi file to segment
    output_dir : str
        path to the directory to save the segmented files
    options : SegmentOptions
        options for the segmentation

    Returns
    -------
    List[str]
        list of paths to the segmented files
    """
    trackname = basename(midi_file_path)
    # preserve tempo across all segments
    target_bpm = get_bpm(midi_file_path)
    set_bpm(midi_file_path, target_bpm)

    # calculate times and stuff
    midi_pm = pretty_midi.PrettyMIDI(midi_file_path)
    total_file_length_s = midi_pm.get_end_time()
    segment_length_s = options.num_beats * 60 / target_bpm
    n_segments = int(np.round(total_file_length_s / segment_length_s))
    pre_beat_window_s = (
        segment_length_s / options.num_beats / 8
    )  # capture when first note is a bit early

    if options.novelty:
        piano_roll = midi_pm.get_piano_roll()
        ssm, novelty = gen_ssm_and_novelty(midi_file_path)
        # save novelty curve
        plt.figure(figsize=(16, 4))
        plt.imshow(
            trim_piano_roll(piano_roll),
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        plt.plot(
            (1 - novelty / novelty.max()) * 17,
            "g",
            linewidth=1.0,
            alpha=0.7,
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(options.pic_dir, f"{basename(midi_file_path)}_novelty.png")
        )
        plt.close()

        # save ssm
        plt.figure(figsize=(16, 16))
        plt.imshow(ssm / ssm.max(), cmap="magma")
        plt.axis("off")
        plt.savefig(
            os.path.join(options.pic_dir, f"{basename(midi_file_path)}_ssm.png")
        )
        plt.close()

    console.log(
        f"\tbreaking '{trackname}' ({total_file_length_s:.03f} s at {target_bpm} bpm) into {n_segments:03d} segments of {segment_length_s:.03f} s\n\t(pre window is {pre_beat_window_s:.03f} s)"
    )

    new_files = []
    for n in list(range(n_segments - (1 if options.drop_last else 0))):
        start = n * segment_length_s
        end = start + segment_length_s - pre_beat_window_s
        if n > 0:
            start -= pre_beat_window_s

        # console.log(f"\t{n:03d} splitting from {start:08.03f} s to {end:07.03f} s")

        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_bpm)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{trackname}_{int(start):04d}-{int(end):04d}",
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
            output_dir, f"{trackname}_{int(start):04d}-{int(end):04d}.mid"
        )

        segment_midi.instruments.append(instrument)
        segment_midi.write(segment_filename)
        set_bpm(segment_filename, target_bpm)
        if options.metronome:
            add_metronome(segment_filename, options.num_beats)
        if options.novelty:
            add_novelty(
                segment_filename,
                novelty,
                options.num_beats,
                (start, end),
                options.pic_dir,
            )
        modify_end_of_track(segment_filename, segment_length_s, target_bpm)

        new_files.append(segment_filename)

    return new_files


def add_metronome(midi_file_path: str, num_beats: int) -> None:
    """
    adds a clave on each beat of the segment.

    Parameters
    ----------
    midi_file_path : str
        path to the midi file.
    num_beats : int
        number of beats in the segment.

    Returns
    -------
    None
    """
    midi = mido.MidiFile(midi_file_path)

    # create a new track for the clave
    clave_track = mido.MidiTrack()
    clave_track.append(mido.MetaMessage("track_name", name="metronome", time=0))

    # add clave notes on each beat
    # +1 because we want to include the last beat
    for beat in range(num_beats + 1):
        time_ticks = midi.ticks_per_beat - midi.ticks_per_beat // 8 if beat > 0 else 0
        clave_track.append(
            mido.Message("note_on", note=76, velocity=100, time=time_ticks, channel=9)
        )

        # note off - make it short (1/8 of a beat)
        clave_track.append(
            mido.Message(
                "note_off",
                note=76,
                velocity=0,
                time=midi.ticks_per_beat // 8,
                channel=9,
            )
        )

    clave_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(clave_track)
    midi.save(midi_file_path)


def add_novelty(
    midi_file_path: str,
    novelty: np.ndarray,
    num_beats: int,
    times: tuple[float, float],
    pic_dir: str,
) -> None:
    """
    Adds a novelty track to the MIDI file.
    TODO: properly convert time from PR to ticks (100 -> 220 and only log every 16th note?)
            i dunno but do this carefully and check

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file.
    novelty : np.ndarray
        The novelty curve.
    num_beats : int
        The number of beats in the segment.
    times : tuple[float, float]
        The start and end times of the segment.
    pic_dir : str
        The directory to save the novelty curve.
    """
    midi = mido.MidiFile(midi_file_path)
    piano_roll = pretty_midi.PrettyMIDI(midi_file_path).get_piano_roll()
    novelty_track = mido.MidiTrack()
    novelty_track.append(mido.MetaMessage("track_name", name="novelty", time=0))

    # find the index of the start and end times in the novelty curve
    start_index = int(times[0] * 100)
    end_index = int(times[1] * 100)
    novelty = novelty[start_index:end_index]

    # add the novelty curve to the track
    n_msgs = [
        mido.MetaMessage("text", text=f"{n:.03f}", time=i)
        for i, n in enumerate(novelty)
    ]
    novelty_track.extend(n_msgs)

    novelty_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(novelty_track)
    midi.save(midi_file_path)

    # save novelty curve
    plt.figure(figsize=(8, 4))
    plt.imshow(
        trim_piano_roll(piano_roll),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    plt.plot(
        (1 - novelty / novelty.max()) * 17,
        "g",
        linewidth=1.0,
        alpha=0.7,
    )
    plt.axis("off")
    plt.savefig(os.path.join(pic_dir, f"{basename(midi_file_path)}_novelty.png"))
    plt.close()


def modify_end_of_track(midi_file_path: str, new_end_time: float, bpm: int) -> None:
    """
    Modifies the 'end_of_track' message in a MIDI file to match the new end time.

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file.
    new_end_time : float
        The new end time.
    bpm : int
        The BPM of the MIDI file.
    """
    midi = mido.MidiFile(midi_file_path)
    new_end_time_t = mido.second2tick(new_end_time, 220, mido.bpm2tempo(bpm))

    for _, track in enumerate(midi.tracks):
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
    midi.save(midi_file_path)


def process_files(
    args,
    pf_files: List[str],
    p_segments: str,
    p_augments: str,
    p_pictures: str,
    index: int,
) -> int:
    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=0.1,
    )
    task_s = p.add_task(f"[SUBR{index:02d}] segmenting", total=len(pf_files))

    seg_opts = SegmentOptions(
        num_beats=args.num_beats,
        metronome=args.clave,
        drop_last=args.drop_last,
        novelty=args.novelty,
        pic_dir=p_pictures,
    )

    # segment files
    segment_paths = []
    augment_paths = []
    with p:
        for pf_file in pf_files:
            # segment
            if args.segment:
                new_segments = segment_midi(
                    os.path.join(args.data_dir, pf_file),
                    p_segments,
                    seg_opts,
                )
            else:
                copy2(
                    os.path.join(args.data_dir, pf_file),
                    os.path.join(p_segments, pf_file),
                )
                new_segments = [os.path.join(args.data_dir, pf_file)]
            segment_paths.extend(new_segments)

            # augment
            if args.augment:
                augment_paths.extend(
                    augment_midi(
                        p, os.path.splitext(pf_file)[0], new_segments, p_augments
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
    p_pictures = os.path.join(args.out_dir, "pictures")

    for dir in [p_segments, p_augments, p_pictures]:
        if not os.path.exists(dir):
            os.makedirs(dir)
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
                args,
                chunk,
                p_segments,
                p_augments,
                p_pictures,
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
        "-c",
        "--clave",
        action="store_true",
        help="add clave track on each beat of the segments",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "-n",
        "--novelty",
        action="store_true",
        help="add novelty score to the segments",
    )
    parser.add_argument(
        "-d",
        "--drop_last",
        action="store_true",
        help="drop the last segment",
    )
    args = parser.parse_args()
    console.log(args)

    main(args)
