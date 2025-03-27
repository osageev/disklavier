import mido
import time
from queue import PriorityQueue
from threading import Thread, Event
from datetime import datetime, timedelta

from utils import console
from utils.midi import TICKS_PER_BEAT
from utils.udp import send_udp
from .worker import Worker


class Max(Worker):
    first_note = False
    active_notes = [0] * 88

    def __init__(self, params, bpm: int, td_start: datetime, pf_max: str):
        super().__init__(params, bpm=bpm)
        self.q_max = PriorityQueue()
        self.td_start = td_start
        self.pf_max = pf_max

    def play(self, queue: PriorityQueue):
        """
        Starts the thread which sends note data to Max

        Parameters
        ----------
        queue : PriorityQueue
            The queue to read the MIDI from

        Returns
        -------
        Event
            The stop event
        """
        self.q_max = queue
        self.stop_event = Event()
        self.max_thread = Thread(target=self.run, name="max", daemon=True)
        self.max_thread.start()

        return self.stop_event

    def stop(self):
        self.stop_event.set()
        if self.max_thread is not None:
            self.max_thread.join(0.1)

        if self.max_thread.is_alive():
            self.max_thread.join(0.1)
            console.log(
                f"{self.tag}[yellow bold] max thread is still running[/yellow bold]"
            )

    def run(self):
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

        while not self.stop_event.is_set():
            tt_abs, msg = self.q_max.get()
            ts_abs = mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)

            if self.verbose:
                console.log(
                    f"{self.tag} absolute time is {tt_abs} ticks (delta is {ts_abs:.03f} seconds)"
                )

            if self.active_notes[msg.note] == 0:
                self.active_notes[msg.note] = ts_abs
                continue

            note = f"{self.active_notes[msg.note]} {ts_abs} {msg.note} {msg.velocity}"
            self.active_notes[msg.note] = 0

            if self.verbose:
                console.log(f"{self.tag} sending message: {note}")
            send_udp(note, "/note")

        # turn off notes & close connection
        for note in range(128):
            note = f"0 0 {note} 0"
            send_udp(note, "/note")

        console.log(f"{self.tag}[green] playback finished")
