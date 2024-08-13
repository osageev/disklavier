import os
from mido import MidiFile

from typing import List


class MidiEvent:
    def __init__(self, global_time: float, relative_time: float, event_data):
        self.global_time = global_time
        self.relative_time = relative_time
        self.event_data = event_data


class PreloadedMidiFile:
    def __init__(self, filename: str, events: List[MidiEvent], duration: float):
        self.filename = filename
        self.events = events
        self.duration = duration  # in seconds


class MidiLoader:
    def __init__(self, directory: str, tempo: int, num_beats: int = 9):
        self.directory = directory
        self.midi_files: List[PreloadedMidiFile] = []
        self.tempo = tempo  # Tempo in beats per minute (BPM)
        self.num_beats = num_beats

    def load_midi_files(self):
        for filename in os.listdir(self.directory):
            if filename.endswith(".mid"):
                midi_file = self._load_single_midi_file(
                    os.path.join(self.directory, filename)
                )
                self.midi_files.append(midi_file)

        # unlikely this will ever be tripped but just in case
        first_dur = self.midi_files[0].duration
        for file in self.midi_files:
            if file.duration != first_dur:
                print(
                    f"WARN: different file durations detected: {first_dur} != {file.duration}"
                )
        print(f"done loading {len(self.midi_files)} files")

    def _load_single_midi_file(self, filepath: str):
        mid = MidiFile(filepath)
        events: List[MidiEvent] = []
        ticks_per_beat = mid.ticks_per_beat

        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if (
                    not msg.is_meta
                ):  # no meta messages since this is used for playback only
                    global_time = self._ticks_to_seconds(current_time, ticks_per_beat)
                    relative_time = msg.time
                    event_data = msg
                    events.append(MidiEvent(global_time, relative_time, event_data))
            segment_length = self._ticks_to_seconds(
                self.num_beats * ticks_per_beat, ticks_per_beat
            )

        return PreloadedMidiFile(filepath, events, segment_length)

    def _ticks_to_seconds(self, ticks: int, ticks_per_beat: int) -> float:
        # Convert ticks to seconds using the tempo and ticks_per_beat
        beats = ticks / ticks_per_beat
        seconds = (beats / self.tempo) * 60
        return seconds
