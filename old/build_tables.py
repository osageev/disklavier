import os
import time
import zipfile
from pretty_midi import PrettyMIDI
import numpy as np
from numpy.linalg import norm
from rich.pretty import pprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import redis
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser, Namespace
from utils import console

from typing import List


tag = "[green]main[/green]  :"


def calc_sims(
    redis_url: str,
    rows: List[str],
    all_rows: List[str],
    metric: str = "pitch_histogram",
    n_transpositions: int = 12,
    n_shifts: int = 8,
    index: int = 0,
) -> int:
    redis_conn = redis.from_url(redis_url)

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=0.1,
    )
    calc_sim_task = progress.add_task(
        f"subr{index:02d} calculating similarities", total=len(rows) * len(all_rows)
    )
    with progress:
        for row in rows:
            for other_row in all_rows:
                # Generate all permutations of the comparison filename
                permutations = [
                    f"{other_row}_t{T:02d}s{S:02d}"
                    for T, S in itertools.product(
                        range(n_transpositions), range(n_shifts)
                    )
                ]

                base_key = f"files:{row}_t00s00"
                base_vector = np.array(
                    redis_conn.json().get(base_key, f"$.{metric}")
                )  # (,12)
                if base_vector is None:
                    raise ValueError(
                        f"base vector not found in Redis for key: '{base_key}'"
                    )

                comparison_vectors = []
                keys = [f"files:{perm}" for perm in permutations]
                vectors = redis_conn.json().mget(keys, f"$.{metric}")
                for perm, vector in zip(permutations, vectors):
                    if vector:
                        comparison_vectors.append((perm, np.array(vector)))
                if not comparison_vectors:
                    raise ValueError(
                        f"No vectors found for any comparison permutations.\n\tLast search was '{keys[-1]}.{metric}'"
                    )

                vectors = (
                    np.stack([vec for _, vec in comparison_vectors]).squeeze().T
                )  # (12,96)
                similarities = np.dot(base_vector, vectors) / (
                    norm(base_vector) * norm(vectors)
                )
                best_perm = comparison_vectors[np.argmax(similarities)][0]
                best_transpose, best_shift = map(
                    int, best_perm.split("_t")[1].split("s")
                )
                transform_info = {
                    "transpose": int(best_transpose),
                    "shift": int(best_shift),
                }
                result_key = f"cmp:{base_key.split(':')[-1]}:{other_row}_t00s00"
                redis_conn.json().set(result_key, "$", transform_info)
                redis_conn.json().set(result_key, f"$.{metric}", similarities.max())
                progress.advance(calc_sim_task)

    progress.remove_task(calc_sim_task)
    console.log(f"[subr{index:02d}] subprocess complete")

    return index


def main(args: Namespace):
    r = redis.from_url(args.redis)
    # filenames
    train_dir = os.path.join(args.data_dir, "synthetic")
    all_filenames = [f[:-4] for f in os.listdir(train_dir) if f.endswith(".mid")]
    all_filenames.sort()
    play_dir = os.path.join(args.data_dir, "synthetic")
    base_filenames = [f[:-4] for f in os.listdir(play_dir) if f.endswith(".mid")]
    base_filenames.sort()
    split_keys = np.array_split(base_filenames, os.cpu_count())  # type: ignore

    console.log(f"{tag} verifying existence of files in redis before starting")
    all_keys_present = True
    for base_filename in base_filenames:
        redis_key = f"files:{base_filename}_t00s00"
        if not r.exists(redis_key):
            console.log(f"Warning: Key '{redis_key}' does not exist in Redis.")
            all_keys_present = False
    if not all_keys_present:
        exit()
    console.log(f"{tag} all required keys are present")

    if args.multithread:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    calc_sims,
                    args.redis,
                    chunk,
                    base_filenames,
                    metric=args.metric,
                    index=i,
                ): chunk
                for i, chunk in enumerate(split_keys)
            }

            for future in as_completed(futures):
                index = future.result()
                console.log(f"{tag} subprocess {index} returned")
    else:
        calc_sims(args.redis, base_filenames, base_filenames, metric=args.metric)

    console.log(f"{tag} similarities have been calculated")
    console.log(f"{tag} DONE")


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "--data_dir", default="data/datasets/test", help="location of MIDI files"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="test", help="name of dataset"
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="pitch_histogram",
        help="metric to use/calculate for similarity measurements",
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
        help="actually build sim table, vs. just uploading the files",
    )
    parser.add_argument(
        "--build_neighbor",
        "-n",
        action="store_true",
        default=False,
        help="actually build neighbor table",
    )
    parser.add_argument(
        "--build_transformation",
        "-r",
        action="store_true",
        default=False,
        help="actually build transformation table",
    )
    parser.add_argument(
        "-t",
        action="store_true",
        default=False,
        help="test dataset mode (don't expect transformations)",
    )
    parser.add_argument(
        "--redis",
        type=str,
        default="redis://localhost:6379/0",
        help="override default redis url",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        default=False,
        help="enable multithreading",
    )
    args = parser.parse_args()
    pprint(args)

    main(args)
