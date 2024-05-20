from multiprocessing import Process, cpu_count
import redis
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import json
import pandas as pd
import numpy as np

from typing import List


def build_similarity_dataframe(redis_url: str, all_files: List[str], indices: slice, batch_size=1000):
    r = redis.Redis(redis_url)
    # initialize an empty DataFrame with row_files as rows and col_files as columns
    df = pd.DataFrame(index=all_files[indices], columns=all_files, dtype=np.float16)

    # process keys in batches
    for i in range(0, indices.stop - indices.start, batch_size):
        row_batch = all_files[indices][i : i + batch_size]

        # use Redis pipeline to batch process the keys
        with r.pipeline() as pipe:
            for row_file in row_batch:
                key_pattern = f"cmp:{row_file}:*"
                keys = r.scan_iter(match=key_pattern)
                for key in keys:
                    pipe.get(key)
            results = pipe.execute()

        # parse the results and update the DataFrame
        for row_file in row_batch:
            key_pattern = f"cmp:{row_file}:*"
            keys = list(r.scan_iter(match=key_pattern))
            for key, value in zip(keys, results):
                if value:
                    data = json.loads(value)
                    col_file = data["col_file"]
                    sim = round(data["sim"], 3)  # limit precision to 3 decimal places
                    if col_file in col_files:
                        df.at[row_file, col_file] = sim

    return df


if __name__=="__main__":
    # Connect to Redis
    redis_url = "redis://localhost:6379"
    r = redis.Redis(redis_url)

    total_keys = r.dbsize()
    num_processes = cpu_count()
    keys_per_process = total_keys // num_processes  # type: ignore
    extra_keys = total_keys % num_processes  # type: ignore

    processes = []
    start_index = 0
    for i in range(num_processes):
        # Distribute the extra keys among the first few processes
        end_index = start_index + keys_per_process + (1 if i < extra_keys else 0)
        process = Process(
            target=build_similarity_dataframe, args=(redis_url, start_index, end_index, i)
        )
        processes.append(process)
        process.start()
        start_index = end_index

    # Wait for all processes to complete
    for process in processes:
        subr_i = process.join()
        print(f"[MAIN]   subroutine {subr_i} complete")

    # Example list of all possible row_files and col_files (for demonstration purposes)
    row_files = [f"20231220-80-01_0000-{i:04d}.mid" for i in range(4000)]
    col_files = [f"20240117-64-06_0080-{i:04d}.mid" for i in range(4000)]

    # Build the DataFrame
    similarity_df = build_similarity_dataframe(redis_url, row_files, batch_size=100)

    # Print the DataFrame
    print(similarity_df)
