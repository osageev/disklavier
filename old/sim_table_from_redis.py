import os
import redis
import numpy as np
import pandas as pd
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List

DATASET = "20240621"

def insert_transformations(filename: str, transformations: List[int]=[0,0]) -> str:
    return f"{filename[:-4]}_t{transformations[0]:02d}s{transformations[1]:02d}{filename[-4:]}"

def calc_sims(rows: List[str], all_rows: List[str],  index: int):
    print(
        f"[SUBR{index:02d}] starting subprocess {index:02d} with {len(rows)}/{len(all_rows)} rows"
    )
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    metric_name="clamp"

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    sim_task = progress.add_task(
        f"[SUBR{index:02d}] calculating sims", total=len(all_rows)
    )
    with progress:
        for i, row_file in enumerate(rows):
            print(
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file '{row_file}'"
            )
            row_metric = r.json().get(f"files:{row_file}", f"$.{metric_name}")

            for col_file in all_rows:
                metric = r.json().get(f"files:{col_file}", f"$.{metric_name}")

            best_sim = -1
            best_shift = -1
            best_trans = -1

            # print(f"calculating sims")
            for col_file, metric in zip(rows, col_metrics_list):
                if r.exists(f'cmp:{row_file[:-7]}:{best_match[:-7]}') == 1:
                    continue
                if metric is not None:
                    col_metric = metric[0]
                    similarity = np.dot(row_metric, col_metric) / (
                        np.linalg.norm(row_metric) * np.linalg.norm(col_metric)
                    )
                    if similarity > best_sim:
                        best_match = col_file
                        best_sim = np.round(similarity, 5)[0]
                        ts = col_file.split('_')[-1]
                        best_trans = int(ts[1:3])
                        best_shift = int(ts[4:])
                else:
                    print(f"[SUBR{index:02d}] [yellow]WARNING: couldn't find metric for {col_file}")

            # print(f"best sim is ", {"transpose": best_trans, "shift": best_shift, metric_name: best_sim})
            r.json().set(f'cmp:{row_file[:-7]}:{best_match[:-7]}', "$", {"transpose": best_trans, "shift": best_shift, metric_name: best_sim})
            progress.advance(sim_task)

    progress.remove_task(sim_task)
    print(f"[SUBR{index:02d}] subprocess complete")

    return index

# def calc_sims(rows: List[str], all_rows: List[str],  index: int):
#     print(
#         f"[SUBR{index:02d}] starting subprocess {index:02d} with {len(rows)}/{len(all_rows)} rows"
#     )
#     r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
#     metric_name="clamp"

#     progress = Progress(
#         SpinnerColumn(),
#         *Progress.get_default_columns(),
#         TimeElapsedColumn(),
#         MofNCompleteColumn(),
#         refresh_per_second=1,
#     )
#     sim_task = progress.add_task(
#         f"[SUBR{index:02d}] calculating sims", total=len(all_rows)
#     )
#     with progress:
#         for i, row_file in enumerate(all_rows):
#             print(
#                 f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file '{row_file}'"
#             )
#             row_metric = r.json().get(f"files:{row_file}", f"$.{metric_name}")

#             # Prepare keys for mget
#             col_keys = [f"files:{col_file}" for col_file in rows]
#             # print(f"[SUBR{index:02d}] querying for {len(col_keys)} keys")
#             col_metrics_list = r.json().mget(col_keys, f"$.{metric_name}")
#             # print(f"[SUBR{index:02d}] got keys")

#             best_sim = -1
#             best_shift = -1
#             best_trans = -1

#             # print(f"calculating sims")
#             for col_file, metric in zip(rows, col_metrics_list):
#                 if r.exists(f'cmp:{row_file[:-7]}:{best_match[:-7]}') == 1:
#                     continue
#                 if metric is not None:
#                     col_metric = metric[0]
#                     similarity = np.dot(row_metric, col_metric) / (
#                         np.linalg.norm(row_metric) * np.linalg.norm(col_metric)
#                     )
#                     if similarity > best_sim:
#                         best_match = col_file
#                         best_sim = np.round(similarity, 5)[0]
#                         ts = col_file.split('_')[-1]
#                         best_trans = int(ts[1:3])
#                         best_shift = int(ts[4:])
#                 else:
#                     print(f"[SUBR{index:02d}] [yellow]WARNING: couldn't find metric for {col_file}")

#             # print(f"best sim is ", {"transpose": best_trans, "shift": best_shift, metric_name: best_sim})
#             r.json().set(f'cmp:{row_file[:-7]}:{best_match[:-7]}', "$", {"transpose": best_trans, "shift": best_shift, metric_name: best_sim})
#             progress.advance(sim_task)

#     progress.remove_task(sim_task)
#     print(f"[SUBR{index:02d}] subprocess complete")

#     return index

def main():
    # get files
    p_train = os.path.join("..", "..", "disklavier", "data", "datasets", DATASET, "train")
    train_filenames = [f[:-4] for f in os.listdir(p_train) if f.endswith(".mid")]
    train_filenames.sort()
    p_play = os.path.join("..", "..", "disklavier", "data", "datasets", DATASET, "play")
    play_filenames = [os.path.join(p_play, f) for f in os.listdir(p_play) if f.endswith(".mid")]
    play_filenames.sort()

    split_keys = np.array_split(play_filenames, os.cpu_count())  # type: ignore

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calc_sims, chunk, play_filenames, i): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            print(f"[MAIN]   subprocess {index} returned")




if __name__ == "__main__":
    main()

