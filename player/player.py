import os
import mido
from mido import MidiFile, MidiTrack, Message
from threading import Event
from queue import Queue
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

            console.log(f"{self.p}Playing {self.playing_file}\t(next up is {next_file})\tsim={similarity:.3f}")

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = next_file_path
            first_loop = False
        

    def play_midi_file(self, midi_path: str) -> None:
        midi_data = MidiFile(midi_path)
        with mido.open_output(self.params.out_port) as outport: # type: ignore
            t = 0.
            for msg in midi_data.play():
                t += msg.time # type: ignore
                if not msg.is_meta:
                    outport.send(msg)
