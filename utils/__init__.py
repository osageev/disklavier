import time
import random
import numpy as np
from pathlib import Path
import simpleaudio as sa
from threading import Event
from rich.console import Console


console = Console(record=True, log_time_format="%m-%d %H:%M:%S")


def tick(bpm: int, stop_event: Event, p: str = "[cyan]metro[/cyan] : ", do_print=True):
    """
    Plays a metronome tick at the specified BPM for a given duration in minutes.

    Args:
    bpm: Beats per minute, defines the tempo of the metronome.
    """
    tick = sa.WaveObject.from_wave_file("data/tick.wav")  # Load the tick sound
    seconds_per_beat = 60.0 / bpm  # Calculate the interval between beats

    while not stop_event.is_set():
        if do_print:
            console.log(f"{p} [grey50]tick!")
        tick.play()  # Play the tick sound
        time.sleep(seconds_per_beat)  # Wait for the next tick


def shift_array(arr, up=0, down=0):
    """Shift array vertically within bounds"""
    if up > 0:
        arr = np.roll(arr, -up, axis=0)
        arr[-up:] = 0
    elif down > 0:
        arr = np.roll(arr, down, axis=0)
        arr[:down] = 0
    return arr


def vertical_shift(array, name: str, num_iterations: int = 1, do_shuffle: bool = False):
    """vertically shift a matrix"""
    shifted_matrices = []

    rows_with_non_zero = np.where(array.any(axis=1))[0]
    maximum_up = array.shape[0] - rows_with_non_zero[-1] - 1
    maximum_down = rows_with_non_zero[0]

    # zipper up & down
    for i in range(1, num_iterations):
        if i < maximum_up:
            shifted_matrices.append(
                (f"{Path(name).stem}_u{i:02d}", np.copy(shift_array(array, down=i)))
            )
        else:
            shifted_matrices.append(
                (f"{Path(name).stem}_d{i:02d}", np.copy(shift_array(array, up=i)))
            )

        if i < maximum_down:
            shifted_matrices.append(
                (f"{Path(name).stem}_d{i:02d}", np.copy(shift_array(array, up=i)))
            )
        else:
            shifted_matrices.append(
                (f"{Path(name).stem}_u{i:02d}", np.copy(shift_array(array, down=i)))
            )

    if do_shuffle:
        random.shuffle(shifted_matrices)

    return shifted_matrices[:num_iterations]
