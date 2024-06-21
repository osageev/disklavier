import os
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from itertools import product
from pretty_midi import PrettyMIDI
import numpy as np

from rich import print
from rich.pretty import pprint
from rich.progress import Progress

from dataset.dataset import segment_midi
from utils.midi import transform

from typing import List


def main(args):
    pprint(args)

    # set up filesystem
    if not os.path.exists(args.data_dir):
        print(f"no data dir found at {args.data_dir}")
        raise IsADirectoryError

    p_path = os.path.join(args.data_dir, "play")
    t_path = os.path.join(args.data_dir, "train")
    u_path = os.path.join(args.data_dir, "unsegmented")

    build_fs(
        [
            p_path,
            t_path,
            u_path,
        ]
    )

    if args.limit is None:
        tracks = os.listdir(args.data_dir)
    else:
        tracks = os.listdir(args.data_dir)[: args.limit]

    # segment files
    segment_paths = []
    augment_paths = []
    for filename in tracks:
        if filename.endswith(".mid"):
            print(f"segmenting '{filename}'")
            new_segments = segment_midi(
                os.path.join(args.data_dir, filename),
                p_path,
            )
            segment_paths.extend(new_segments)

            p = Progress()
            t = p.add_task("augmenting", total=len(new_segments) * 96)
            with p:
                if args.build_train:
                    for segment_filename in new_segments:
                        transformations = [
                            {"transpose": t, "shift": s}
                            for t, s in product(range(12), range(8))
                        ]
                        for transformation in transformations:
                            augment_paths.append(
                                transform(
                                    segment_filename,
                                    t_path,
                                    int(filename.split("-")[1]),
                                    transformation,
                                )
                            )
                            p.advance(t)

            os.rename(
                os.path.join(args.data_dir, filename),
                os.path.join(u_path, filename),
            )

    print(
        f"[green bold]segmentation complete, {len(segment_paths)} play files generated and {len(augment_paths)} train files generated"
    )

    prs = {}
    p = Progress()
    t = p.add_task("saving prs", total=len(augment_paths))
    with p:
        for augmentation in augment_paths:
            prs[Path(augmentation).stem] = PrettyMIDI(augmentation).get_piano_roll()
            p.advance(t)
    print("prs calculated, saving...")
    np.savez_compressed(os.path.join(args.data_dir, "all_prs.npz"), **prs)
    print("DONE")


def build_fs(dirs: List[str]) -> None:
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f"created new folder: '{dir}'")


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--store_metrics",
        default=f"metrics-{datetime.now().strftime('%y%m%d-%H%M%S')}.json",
        help="file to write segment metrics to (must be JSON)",
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
        "--build_train",
        action="store_true",
        help="augment dataset and store files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        help="upload files to redis",
    )
    args = parser.parse_args()

    main(args)
