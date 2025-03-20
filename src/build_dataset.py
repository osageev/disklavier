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
from utils.config import load_config, merge_cli_args
from utils.midi import get_bpm, set_bpm, transform, trim_piano_roll
from utils.novelty import gen_ssm_and_novelty
from utils.dataset import add_metronome, add_novelty, modify_end_of_track
from typing import List

plt.style.use("dark_background")


class SegmentOptions:
    def __init__(
        self,
        num_beats: int,
        metronome: bool,
        novelty: bool,
        pic_dir: str,
    ):
        self.num_beats = num_beats
        self.metronome = metronome
        self.novelty = novelty
        self.pic_dir = pic_dir


class AugmentOptions:
    def __init__(self, num_shifts: int, num_transposes: int, tempo_fold: bool):
        self.num_shifts = num_shifts
        self.num_transposes = num_transposes
        self.tempo_fold = tempo_fold


def augment_midi(
    p: Progress,
    filename: str,
    new_segments: List[str],
    output_path: str,
    options: AugmentOptions,
) -> List[str]:
    augmented_files = []
    task_a = p.add_task(
        f"augmenting {basename(filename)}",
        total=len(new_segments) * options.num_transposes * options.num_shifts,
    )

    with p:
        for segment_filename in new_segments:
            transformations = [
                {"transpose": t, "shift": s}
                for t, s in product(
                    range(options.num_transposes), range(options.num_shifts)
                )
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
    )  # capture when first note is a bit early (1/8th of a beat)

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
    # -1 because the last segment often contains a mistake or isn't otherwise suitable for playback
    for n in list(range(n_segments - 1)):
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
        novelty=args.novelty,
        pic_dir=p_pictures,
    )
    aug_opts = AugmentOptions(
        num_shifts=args.num_beats,
        num_transposes=args.num_transposes,
        tempo_fold=args.tempo_fold,
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
                        p,
                        os.path.splitext(pf_file)[0],
                        new_segments,
                        p_augments,
                        aug_opts,
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
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.data_dir)
                zipf.write(file_path, arcname)
                n_files += 1

    console.log(f"[green bold]segmentation complete, {n_files} files generated")


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset builder arguments")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    args = parser.parse_args()
    console.log(args)

    if args.config is not None:
        config = load_config(args.config)
        args = merge_cli_args(config, vars(args))

    console.log(args)

    main(args)
