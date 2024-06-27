import redis
import hashlib
import numpy as np

from utils.midi import split_filename

from typing import List


def get_track_id(track_name: str) -> str:
    """Generate a unique ID for a track name."""
    # return hashlib.md5(track_name.encode()).hexdigest()
    return track_name


def store_vector(
    redis_client: redis.Redis, filename: str, metric: str, vector: np.ndarray
):
    """Store a vector in Redis using an efficient key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        vector: The vector to store.
    """
    # generate key parts
    basename, transpose, shift = split_filename(filename)
    track_id = get_track_id(basename)
    # track_id = basename

    all_ids = redis_client.json().get("track_ids")

    if all_ids is None:
        redis_client.json().set("track_ids", "$", {})
        all_ids = redis_client.json().get("track_ids")

    if track_id not in all_ids.values():  # type: ignore
        redis_client.json().set("track_ids", f"$.{basename}", track_id)

    # store the vector in the hash
    redis_client.hset(f"file:{track_id}:{metric}", f"{transpose}{shift}", vector.tobytes())  # type: ignore


def load_vector(
    redis_client: redis.Redis, filename: str, metric: str, dtype=np.float64, shape=(12,)
):
    """Load a vector from Redis using an efficient key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        dtype: The data type of the vector.
        shape: The shape of the vector.

    Returns:
        The vector.
    """
    # generate key parts
    basename, transpose, shift = split_filename(filename)
    track_id = get_track_id(basename)

    # retrieve the vector from the hash
    vector_bytes = redis_client.hget(f"file:{track_id}:{metric}", f"{transpose}{shift}")

    # if vector_bytes is None:
    #     return None

    # convert bytes back to vector
    return np.frombuffer(vector_bytes, dtype=dtype).reshape(shape)  # type: ignore


def load_vectors(
    redis_client: redis.Redis,
    filename: str,
    metric: str,
    fields: List[str] | None = None,
    dtype=np.float64,
    shape=(12,),
):
    """Load vectors from Redis using an efficient key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        dtype: The data type of the vector.
        shape: The shape of the vector.

    Returns:
        The vector.
    """
    # generate key parts
    basename, _, _ = split_filename(filename)
    track_id = get_track_id(basename)

    # retrieve the vectors from the hash
    if fields is None:
        vector_dict = redis_client.hgetall(f"file:{track_id}:{metric}")
    else:
        vector_dict = redis_client.hmget(f"file:{track_id}:{metric}", fields)

    # if vector_bytes is None:
    #     return None

    # print(f"got vector for file {filename}", vector_dict)

    # convert bytes back to vector
    byte2arr = lambda b: np.frombuffer(b, dtype=dtype).reshape(shape)
    return {k.decode("utf-8"): byte2arr(v) for k, v in vector_dict.items()}  # type: ignore
