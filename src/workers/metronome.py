import time
import pygame
from datetime import datetime, timedelta

from .worker import Worker
from utils import console

TS_DELAY_COMPENSATION = 0.1


class Metronome(Worker):
    def __init__(self, params, bpm: int, t_start: datetime):
        super().__init__(params)
        self.bpm = bpm
        self.do_tick = params.do_tick if hasattr(params, "do_tick") else False
        self.wav_file_1 = params.tick_1
        self.wav_file_2 = params.tick_2
        self.td_start = t_start
        self.beat_interval = 60 / self.bpm

    def tick(self):
        pygame.mixer.init()
        next_tick = self.td_start  # + timedelta(seconds=TS_DELAY_COMPENSATION)
        while True:
            now = datetime.now()
            if now >= next_tick:
                beat_number = (
                    int((now - self.td_start).total_seconds() / self.beat_interval)
                    % self.params.n_beats_per_segment
                ) + 1
                console.log(
                    f"{self.tag} [grey50]tick {beat_number}/{self.params.n_beats_per_segment}[/grey50]"
                )
                if self.do_tick:
                    if beat_number == 1:
                        self.tick_sound = pygame.mixer.Sound(self.wav_file_1)
                    else:
                        self.tick_sound = pygame.mixer.Sound(self.wav_file_2)
                    self.tick_sound.play()
                next_tick += timedelta(seconds=self.beat_interval)
            time.sleep(0.01)  # Small sleep to prevent busy-waiting
