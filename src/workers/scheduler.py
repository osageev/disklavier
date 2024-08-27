from datetime import datetime, timedelta

from utils import console
from .worker import Worker


class Scheduler(Worker):
    n_beats_per_segment = 8
    t_transitions = []

    def __init__(self, params, bpm: int, log_path: str):
        self.tag = params.tag
        self.bpm = bpm
        self.n_beats_per_segment = params.n_beats_per_segment
        self.log_path = log_path

        console.log(f"{self.tag} initialization complete")

    def gen_transitions(self, t_start: datetime | int, n_stamps: int = 100) -> list:
        """Generate a list of timestamps marking the end of 8-beat segments.

        Args:
            t_start (datetime): The starting time.
            bpm (int): Beats per minute.
            n_stamps (int): number of timestamps to generate (default ).

        Returns:
            list: A list of datetime objects representing the end of each 8-beat segment.
        """
        # dt_interval = timedelta(seconds=(self.n_beats_per_segment * 60) / self.bpm)
        # console.log(f"{self.tag} interval {dt_interval}")
        # self.t_transitions = [t_start + i * dt_interval for i in range(n_stamps)]
        dt_interval = (self.n_beats_per_segment * 60) / self.bpm
        console.log(f"{self.tag} interval {dt_interval}")
        self.t_transitions = [t_start + i * dt_interval for i in range(n_stamps)]

        return self.t_transitions
