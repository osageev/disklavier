from datetime import datetime, timedelta
from queue import PriorityQueue
import mido

from utils import console
from .worker import Worker


class Scheduler(Worker):
    n_beats_per_segment: int = 8
    t_transitions: list[float] = []

    def __init__(self, params, bpm: int, log_path: str):
        self.tag = params.tag
        self.bpm = bpm
        self.n_beats_per_segment = params.n_beats_per_segment
        self.log_path = log_path

        console.log(f"{self.tag} initialization complete")

    def gen_transitions(self, n_stamps: int = 100, do_ticks: bool = False) -> list:
        t_interval = (self.n_beats_per_segment * 60) / self.bpm
        console.log(f"{self.tag} segment interval is {t_interval} seconds")
        self.t_transitions = [i * t_interval for i in range(n_stamps)]

        return [
            mido.MetaMessage(
                "text",
                text=f"transition {i} ({t_transition}s)",
                time=mido.second2tick(t_transition, 96, self.bpm),
            )
            for i, t_transition in enumerate(self.t_transitions)
        ]

    def gen_transitions_cgpt(self, n_stamps: int = 100, do_ticks: bool = False) -> list:
        t_interval = (self.n_beats_per_segment * 60) / self.bpm
        beat_interval = 60 / self.bpm  # time interval for each beat
        self.t_transitions = [i * t_interval for i in range(n_stamps)]
        console.log(
            f"{self.tag} segment interval is {t_interval} seconds", self.t_transitions
        )

        transitions = []

        for i, t_transition in enumerate(self.t_transitions):
            # Add the transition MetaMessage at the end of the segment
            transitions.append(
                mido.MetaMessage(
                    "text",
                    text=f"transition {i} ({t_transition}s)",
                    time=mido.second2tick(t_transition, 96, self.bpm),
                )
            )
            # Add tick messages for each beat in the segment if do_ticks is True
            if do_ticks:
                for beat in range(self.n_beats_per_segment):
                    tick_time = t_transition + (beat * beat_interval)
                    transitions.append(
                        mido.MetaMessage(
                            "text",
                            text=f"tick {beat + 1} ({tick_time}s)",
                            time=mido.second2tick(tick_time, 96, self.bpm),
                        )
                    )

        return transitions

    def add_midi_to_queue(
        self, midi_file_path: str, midi_queue: PriorityQueue
    ) -> float:
        console.log(f"{self.tag} adding file to queue '{midi_file_path}'")

        midi_file = mido.MidiFile(midi_file_path)
        t_abs = 0  # to keep track of the absolute time in ticks

        for msg in [m for m in midi_file if not m.is_meta]:  # skip meta messages
            t_abs += msg.time
            console.log(f"{self.tag} adding message to queue: {t_abs}, {msg}")
            midi_queue.put(
                (t_abs, msg)
            )  # add message with its absolute time to the queue

        return t_abs
