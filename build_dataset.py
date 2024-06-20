import os
from datetime import datetime
from argparse import ArgumentParser

from rich import print
from rich.pretty import pprint
from rich.progress import track

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
    p = Progress()
    t1 = p.add_task("segmenting", total=len(tracks))
    with p:
        for filename in tracks:
            if filename.endswith(".mid"):
                print(f"segmenting '{filename}'")
                new_segments = segment_midi(
                    os.path.join(args.data_dir, filename),
                    p_path,
                )
                segment_paths.extend(new_segments)

                t2 = p.add_task("augmenting", total=len(new_segments) * 12 * 8)

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
                            p.advance(t2)

                    p.remove_task(t2)

                p.advance(t1)

                os.rename(
                    os.path.join(args.data_dir, filename),
                    os.path.join(u_path, filename),
                )

    print(
        f"segmentation complete, {len(segment_paths)} play files generated and {len(augment_paths)} train files generated"
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

    print("[green bold]DONE")


def build_fs(dirs: List[str]) -> None:
    for dir in dirs:
        if os.path.exists(dir):
            i = 0
            # for i, file in enumerate(os.listdir(dir)):
            #     os.remove(os.path.join(dir, file))
            #     i += 1
            print(f"cleaned {i} files out of folder: '{dir}'")
        else:
            os.makedirs(dir)
            print(f"created new folder path: '{dir}'")


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

    graveyard = os.path.join("data", "outputs", "graveyard")
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

    total_segs = len(os.listdir(args.output_dir))
    print(f"[green]segmentation complete, {num_files} files generated")
