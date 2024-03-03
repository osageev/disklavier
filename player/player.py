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
        self.playback_begun = playback_event
        self.file_queue = filename_queue


    def playback_loop(self, recorded_ph: List):
        """"""
        console.log(f"{self.p}beginning playback from {recorded_ph}")
        time_elapsed = 0.
        self.playback_begun.set()
        (next_file, similarity) = self.file_queue.get(block=True)
        next_file_path = os.path.join(self.seeker.input_dir, next_file)
        first_loop = True

        while next_file is not None:
            self.playing_file = os.path.basename(self.playing_file_path)
            console.log(f"{self.p}{self.playing_file_path}\t{self.playing_file}")
            console.log(f"{self.p}{next_file_path}\t{next_file}")
            if not first_loop: # bad code!
                console.log(f"{self.p}getting next file for {self.playing_file_path}")
                self.seeker.metrics[self.playing_file]['played'] = 1
                next_file, seeker = self.seeker.get_most_similar_file(self.playing_file)
                next_file_path = os.path.join(self.seeker.input_dir, next_file)

            console.log(f"{self.p}[{int(time_elapsed):04d}]\tPlaying {self.playing_file}\t(next up is {next_file})\tsim={seeker:.3f}")
            time_elapsed += MidiFile(self.playing_file_path).length

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
