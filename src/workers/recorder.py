import os
import mido
from datetime import datetime, timedelta
import mido
from threading import Thread, Event

from .worker import Worker
from utils import console, tick
from .scheduler import N_TICKS_PER_BEAT

from typing import List


class Recorder(Worker):
    recorded_notes: List[mido.Message] = []
    is_recording: bool = False

    def __init__(
        self,
        params,
        bpm: int,
        recording_file_path: str,
    ):
        super().__init__(params, bpm=bpm)
        self.pf_midi_recording = recording_file_path

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def run(self) -> float:
        start_time = datetime.now()
        end_time = 0
        last_note_time = start_time

        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.append(
            mido.MetaMessage(
                type="set_tempo",
                tempo=self.tempo,
                time=0,
            )
        )

        with mido.open_input(self.params.midi_port) as inport:  # type: ignore
            console.log(
                f"{self.tag} listening on port '{self.params.midi_port}' at {midi.ticks_per_beat} tpb"
            )
            for msg in inport:
                # record pedal signal
                if msg.type == "control_change" and msg.control == self.params.record:
                    # record pedal released
                    if msg.value == 0:
                        end_time = datetime.now()
                        console.log(
                            f"{self.tag} recorded {(end_time - start_time).total_seconds():.02f} s"
                        )
                        self.is_recording = False
                        self.stop_tick_event.set()
                        self.metro_thread.join()

                        # have any notes been recorded?
                        if len(self.recorded_notes) > 0:
                            console.log(
                                f"{self.tag} saving recording '{os.path.basename(self.pf_midi_recording)}'"
                            )

                            # write out recording
                            midi.tracks.append(track)
                            midi.save(self.pf_midi_recording)

                            console.log(
                                f"{self.tag} saved recording '{self.pf_midi_recording}'"
                            )
                            return (end_time - start_time).total_seconds()
                        else:
                            # return to waiting for pedal press state
                            console.log(f"{self.tag} no notes recorded")

                    # record pedal not released, but not already recording
                    elif self.is_recording == False:
                        console.log(f"{self.tag} recording at {self.bpm} BPM")
                        self.is_recording = True

                        self.stop_tick_event = Event()
                        self.metro_thread = Thread(
                            target=tick,
                            args=(self.bpm, self.stop_tick_event, self.params.tag),
                            name="recorder metronome",
                        )
                        self.metro_thread.start()

                # record note on/off
                elif self.is_recording and msg.type in ["note_on", "note_off"]:
                    current_time = datetime.now()
                    if len(self.recorded_notes) == 0:
                        # set times to start from this point
                        start_time = datetime.now()
                        if self.verbose:
                            console.log(
                                f"{self.tag} first note received at {start_time.strftime('%H:%M:%S.%f')}"
                            )
                        msg.time = 0
                    else:
                        msg.time = int(
                            (current_time - last_note_time).total_seconds()
                            * N_TICKS_PER_BEAT
                            * self.bpm
                            / 60
                        )
                    track.append(msg)
                    self.recorded_notes.append(msg)
                    console.log(f"{self.tag} \t{msg}")
                    last_note_time = current_time
        return -1.0

    def save_midi(self) -> None:
        """Saves the recorded notes to a MIDI file."""
        console.log(
            f"{self.tag} saving recording '{os.path.basename(self.pf_midi_recording)}'"
        )

        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.append(
            mido.MetaMessage(
                type="set_tempo",
                tempo=self.bpm,
                time=0,
            )
        )
        for msg in self.recorded_notes:
            track.append(msg)
        midi.tracks.append(track)

        midi.save(self.pf_midi_recording)

        if os.path.exists(self.pf_midi_recording):
            console.log(
                f"{self.tag} successfully saved recording '{os.path.basename(self.pf_midi_recording)}'"
            )
            mid = mido.MidiFile(self.pf_midi_recording)
            # mid.print_tracks()
        else:
            console.log(
                f"{self.tag} failed to save recording '{os.path.basename(self.pf_midi_recording)}'"
            )
