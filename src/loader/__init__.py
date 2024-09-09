import os
import mido

from typing import List
from dataclasses import dataclass, field
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from utils import console


@dataclass(order=True)
class MidiEvent:
    index: int  # to break ties when pulling from heap
    global_ticks: int  # ticks
    msg: mido.Message = field(compare=False)

    def __init__(self, index: int, global_ticks: int, msg: mido.Message):
        self.index = index
        self.global_ticks = global_ticks
        self.msg = msg


class PreloadedMidiFile:
    def __init__(self, filename: str, events: List[MidiEvent], ticks_per_beat: int):
        self.file_name = filename
        self.events = events
        self.ticks_per_beat = ticks_per_beat


class Loader:
    P = "[green4]load[/green4]  :"
    midi_files: List[PreloadedMidiFile] = []
    duration_t: int = 0
    ticks_per_beat: int = 0

    def __init__(self, directory: str, tempo: int, num_beats: int = 9):
        self.directory = directory
        self.bpm = tempo
        self.tempo = mido.bpm2tempo(self.bpm)
        self.num_beats = num_beats

    def load_midi_files(self) -> None:
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=1,
        )
        load_task = progress.add_task(
            "loading files",
            total=len([f for f in os.listdir(self.directory) if f.endswith(".mid")]),
        )
        with progress:
            for filename in sorted(os.listdir(self.directory)):
                if filename.endswith(".mid"):
                    file_path = os.path.join(self.directory, filename)

                    # all files will have same tpb, only need to calculate once
                    if self.duration_t == 0:
                        self.ticks_per_beat = mido.MidiFile(file_path).ticks_per_beat
                        self.duration_t = int(
                            self.num_beats * (60 / self.bpm) * self.ticks_per_beat
                        )
                        console.log(
                            f"{self.P} segments have {self.ticks_per_beat} tpb and duration {self.duration_t} ticks"
                        )

                    midi_file = self._load_single_midi_file(file_path)
                    self.midi_files.append(midi_file)
                    progress.advance(load_task)

        console.log(f"{self.P} loaded {len(self.midi_files)} files")

    def _load_single_midi_file(self, filepath: str) -> PreloadedMidiFile:
        mid = mido.MidiFile(filepath)
        events: List[MidiEvent] = []

        for track in mid.tracks:
            current_ticks = 0
            for i, msg in enumerate(track):
                current_ticks += msg.time
                if not msg.is_meta:
                    events.append(MidiEvent(i, current_ticks, msg))

        return PreloadedMidiFile(
            filepath,
            events,
            self.ticks_per_beat,
        )
