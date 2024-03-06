import os
import mido
import time
from mido import MidiFile
from threading import Event
from queue import Queue
import simpleaudio as sa

from utils import console

from typing import List

class Player():
    p = '[green]play[/green]  : '
    is_playing = False

    def __init__(self, params, record_dir: str, playback_event: Event, filename_queue: Queue) -> None:
        self.params = params
        self.record_dir = record_dir
        self.get_next = playback_event
        self.file_queue = filename_queue
        self.out_port = mido.open_output(self.params.out_port) # type: ignore


    def playback_loop(self, seed_file_path: str, recorded_ph: List):
        """"""
        console.log(f"{self.p}beginning playback from {recorded_ph}")
        
        self.playing_file_path = seed_file_path
        (next_file_path, similarity) = self.file_queue.get(block=True)
        next_file = os.path.basename(next_file_path)
        first_loop = True

        while next_file_path is not None:
            self.playing_file = os.path.basename(self.playing_file_path)
            if not first_loop: # bad code!
                console.log(f"{self.p}getting next file for {self.playing_file_path}")
                next_file_path, similarity = self.file_queue.get()
                next_file = os.path.basename(next_file_path)
            self.get_next.set()

            console.log(f"{self.p}playing {self.playing_file}\t(next up is {next_file})\tsim={similarity:.3f}")

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = next_file_path
            first_loop = False
        

    def play_midi_file(self, midi_path: str) -> None:
        playback_bpm = int(os.path.basename(midi_path).split('-')[1])
        console.log(f"{self.p}playback bpm is {playback_bpm}")

        tick_interval = 60./playback_bpm
        next_tick = tick_interval
        start_time = time.time()
        for msg in MidiFile(midi_path).play():
            self.out_port.send(msg)
            # current_time = time.time() - start_time
            # if current_time >= next_tick:
            #     self._tick()
            #     next_tick += tick_interval

        # Ensure the function runs for the total duration of the MIDI file
        while time.time() - start_time < MidiFile(midi_path).length:
            current_time = time.time() - start_time
            if current_time >= next_second:
                self._tick()
                next_second += tick_interval
            time.sleep(0.01)  # Sleep briefly to avoid a busy wait


    def _tick(self):
        tick = sa.WaveObject.from_wave_file("data/tick.wav")
        tick.play()

