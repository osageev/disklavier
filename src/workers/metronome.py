import time
import pygame
from datetime import datetime, timedelta

from .worker import Worker
from utils import console

TS_DELAY_COMPENSATION = 0.1


class Metronome(Worker):
    def __init__(self, params, bpm: int, t_start: datetime):
        super().__init__(params, bpm=bpm)
        self.do_tick = params.do_tick if hasattr(params, "do_tick") else False
        self.wav_file_1 = params.tick_1
        self.wav_file_2 = params.tick_2
        self.td_start = t_start
        self.beat_interval = 60 / self.bpm
        self.running = False

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def tick(self):
        self.running = True
        console.log(f"{self.tag} initializing pygame")
        pygame.mixer.init()
        console.log(f"{self.tag} initialized pygame")
        next_tick = self.td_start  # + timedelta(seconds=TS_DELAY_COMPENSATION)
        try:
            while self.running:
                now = datetime.now()
                if now >= next_tick:
                    num_beat = (
                        int((now - self.td_start).total_seconds() / self.beat_interval)
                        % self.params.n_beats_per_segment
                    ) + 1
                    console.log(
                        f"{self.tag} [grey50]tick {num_beat}/{self.params.n_beats_per_segment}[/grey50]"
                    )
                    if self.do_tick:
                        # play tick, changing sample on first beat
                        pygame.mixer.Sound(
                            self.wav_file_1 if num_beat == 1 else self.wav_file_2
                        ).play()
                    next_tick += timedelta(seconds=self.beat_interval)
                time.sleep(0.01)  # Small sleep to prevent busy-waiting
        except KeyboardInterrupt:
            console.log(f"{self.tag}[yellow] CTRL + C detected, exiting...")

    def stop(self):
        self.running = False
