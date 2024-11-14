import os
import redis
import numpy as np
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from itertools import product
from scipy.spatial.distance import cosine
from concurrent.futures import ProcessPoolExecutor, as_completed
from ml.clamp.model import Clamp

DATASET = "20240621"
N_BEATS = 8
N_TRANSPOSITIONS = 12


def gen_transformations(filename: str, options) -> list[str]:
    transformed_filenames = []
    for opt in options:
        transformed_filenames.append(
            f"{filename[:-4]}_t{opt[0]:02d}s{opt[1]:02d}{filename[-4:]}"
        )
    return transformed_filenames


def calc_sims(rows: list[str], all_rows: list[str], index: int):
    print(
        f"[SUBR{index:02d}] starting subprocess {index:02d} with {len(rows)}/{len(all_rows)} rows"
    )
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    model = Clamp(verbose=False)
    mod_table = list(product(range(N_TRANSPOSITIONS), range(N_BEATS)))

    progress = Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    sim_task = progress.add_task(
        f"[SUBR{index:02d}] calculating sims", total=len(all_rows)
    )
    with progress:
        for i, pf_row in enumerate(all_rows[:1]):
            print(
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file '{pf_row}'"
            )
            # only compare to un-transformed row
            row_embedding = model.encode([pf_row])  # (1, 768)
            row_embedding = row_embedding.squeeze()  # (768)
            row_name = os.path.basename(pf_row)[:-4]

            for pf_col in all_rows:
                # same file comparison
                if pf_col == pf_row:
                    r.json().set(
                        f"cmp:{row_name}:{row_name}",
                        "$",
                        {"transpose": 0, "shift": 0},
                        nx=True,
                    )
                    r.json().set(f"cmp:{row_name}:{row_name}", "$", {"clamp": 1.0})
                    continue

                transformed_columns = gen_transformations(
                    pf_col.replace("play", "train"), mod_table
                )
                col_embeddings = model.encode(transformed_columns)  # (96, 768)
                # col_embeddings = col_embeddings.T # (768, 96)
                # similarities = row_embedding @ col_embeddings / (
                #     np.linalg.norm(row_embedding) * np.linalg.norm(col_embeddings)
                # )
                similarities = [
                    1 - cosine(row_embedding, col_embedding)
                    for col_embedding in col_embeddings
                ]
                # for c, e in zip(transformed_columns, similarities):
                #     print(f"{os.path.basename(c)}\t{e:.05f}")
                best_match_index = np.argmax(similarities)
                best_match = os.path.basename(transformed_columns[best_match_index])
                best_sim = float(np.round(similarities[best_match_index], 5))
                ts = best_match.split("_")[-1]
                best_trans = int(ts[1:3])
                best_shift = int(ts[4:-4])

                # print(f"best sim for '{row_name}' is '{best_match[:-11]}'", {"transpose": best_trans, "shift": best_shift, "clamp": best_sim})
                r.json().set(
                    f"cmp:{row_name}:{best_match[:-7]}",
                    "$",
                    {"transpose": best_trans, "shift": best_shift, "clamp": best_sim},
                )
            progress.advance(sim_task)

    progress.remove_task(sim_task)
    print(f"[SUBR{index:02d}] subprocess complete")

    return index


def main():
    # get files
    p_train = os.path.join("data", "datasets", DATASET, "train")
    train_filenames = [f[:-4] for f in os.listdir(p_train) if f.endswith(".mid") or f.endswith(".midi")]
    train_filenames.sort()
    p_play = os.path.join("data", "datasets", DATASET, "play")
    play_filenames = [
        os.path.join(p_play, f) for f in os.listdir(p_play) if f.endswith(".mid") or f.endswith(".midi")
    ]
    play_filenames.sort()

    split_keys = np.array_split(play_filenames, os.cpu_count())  # type: ignore

    # calc_sims(play_filenames, play_filenames, 0)
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
