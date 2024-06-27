import os
from pretty_midi import PrettyMIDI
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import redis
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser

from utils.midi import insert_transformations
from utils.redis import store_vector, load_vector, load_vectors

from typing import List


def calc_sims(rows: List[str], all_rows: List[str], metric: str, index: int):
    print(f"[SUBR{index:02d}] starting subprocess {index:02d}")
    r = redis.Redis(host="localhost", port=6379, db=0)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    sim_task = progress.add_task(
        f"[SUBR{index:02d}] calculating sims", total=len(rows) * len(all_rows)
    )
    with progress:
        for i, row_file in enumerate(rows):
            print(
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file {row_file}"
            )

            # find optimal transformation for each other file in dataset to maximize similarity
            for col_file in all_rows:
                best_sim = -1
                best_shift = -1
                best_trans = -1

                row_metric = load_vector(r, insert_transformations(row_file), metric)
                col_metrics = load_vectors(r, insert_transformations(col_file), metric)

                for key in col_metrics.keys():
                    similarity = np.dot(row_metric, col_metrics[key]) / (
                        np.linalg.norm(row_metric) * np.linalg.norm(col_metrics[key])
                    )

                    if similarity > best_sim:
                        best_sim = np.round(similarity, 5)
                        best_shift = f"{key}"[:2]
                        best_trans = f"{key}"[2:]

                value = {
                    "sim": best_sim,
                    "shift": best_shift,
                    "trans": best_trans,
                    "row_file": row_file,
                    "col_file": col_file,
                    "metric": metric,
                }

                # update comparison object
                for k, v in value.items():
                    r.hset(f"cmp:{row_file[:-4]}:{col_file[:-4]}:{metric}", k, v)

                progress.advance(sim_task)

    print(f"[SUBR{index:02d}] subprocess complete")

    return index


def main(args):
    r = redis.Redis(host="localhost", port=6379, db=0)
    train_dir = os.path.join(args.data_dir, "train")
    all_filenames = [f for f in os.listdir(train_dir) if f.endswith(".mid")]

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    pitch_histogram_task = progress.add_task(
        f"uploading {args.metric}s", total=(len(all_filenames))
    )
    with progress:
        for file in all_filenames:
            vector = PrettyMIDI(
                os.path.join(train_dir, file)
            ).get_pitch_class_histogram(True, True)
            store_vector(r, file, args.metric, vector)
            progress.advance(pitch_histogram_task)

    # calculate similarities
    play_dir = os.path.join(args.data_dir, "play")
    base_filenames = [f for f in os.listdir(play_dir) if f.endswith(".mid")]
    split_keys = np.array_split(base_filenames, os.cpu_count())  # type: ignore

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calc_sims, chunk, base_filenames, args.metric, i): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            print(f"[MAIN]   subprocess {index} returned")
            
    if args.build_sim:


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--metric", "-m", default="pitch_histogram", help="metric to use/calculate"
    )
    parser.add_argument(
        "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have, not including the leading and trailing sections of each segment",
    )
    parser.add_argument(
        "--num_semitones",
        type=int,
        default=12,
        help="number of semitones to transpose",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "--build_sim",
        "-s",
        action="store_true",
        default=False,
        help="actually build table, vs. just uploading the files",
    )
    args = parser.parse_args()

    main(args)
