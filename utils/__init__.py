import time
import simpleaudio as sa
from threading import Event
from rich.console import Console


console = Console(record=True, log_time_format='%m-%d %H:%M:%S')


def tick(bpm: int, stop_event: Event):
    """
    Plays a metronome tick at the specified BPM for a given duration in minutes.
    
    Args:
    bpm: Beats per minute, defines the tempo of the metronome.
    """
    tick = sa.WaveObject.from_wave_file("data/tick.wav")  # Load the tick sound
    seconds_per_beat = 60.0 / bpm  # Calculate the interval between beats
    
    while not stop_event.is_set():        
        console.log(f"tick!")
        tick.play()  # Play the tick sound
        time.sleep(seconds_per_beat)  # Wait for the next tick