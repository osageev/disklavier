import os
import mido
from datetime import datetime
import mido
from threading import Thread, Event
import time

from .worker import Worker
from utils import console, tick
from utils.midi import TICKS_PER_BEAT

from typing import List, Optional


class Recorder(Worker):
    """
    Records MIDI input and saves it to a file.
    """

    recorded_notes: List[mido.Message] = []
    is_recording: bool = False
    n_ticks: int = 0
    velocity_display = None

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
        """
        Records MIDI input and saves it to a file.

        Returns
        -------
        float
            The duration of the recording in seconds.
        """
        start_time = datetime.now()
        end_time = 0
        last_note_time = start_time

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
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
                f"{self.tag} listening on port '{self.params.midi_port}' at {midi.ticks_per_beat} ticks per beat"
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
                            args=(
                                self.bpm,
                                self.stop_tick_event,
                                self.params.tag,
                            ),
                            name="recorder metronome",
                        )
                        self.metro_thread.start()
                        self.n_ticks += 1

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
                            * TICKS_PER_BEAT
                            * self.bpm
                            / 60
                        )
                    track.append(msg)
                    self.recorded_notes.append(msg)
                    console.log(f"{self.tag} \t{msg}")
                    last_note_time = current_time
        return -1.0

    def passive_record(self, td_start: Optional[datetime] = None) -> None:
        """
        Passively records notes from the MIDI input.

        Parameters
        ----------
        td_start : Optional[datetime]
            Start time to calculate message times relative to.
        """

        if td_start is None:
            td_start = datetime.now()

        last_msg_time = td_start

        # initialize velocity tracking variables
        velocity_window = []  # list of (timestamp, velocity) tuples
        window_duration_seconds = 8 * 60.0 / self.bpm  # 8 beats in seconds

        # track the last time we printed stats
        last_stats_print_time = time.time()
        last_display_update_time = time.time()
        display_update_interval = 0.5  # Update display every half second

        if self.verbose:
            console.log(f"{self.tag} listening on port '{self.params.midi_port}'")
        self.is_recording = True

        with mido.open_input(self.params.midi_port) as inport:  # type: ignore
            for msg in inport:
                if not self.is_recording:
                    break

                current_time = datetime.now()
                # calculate time in ticks since last message
                time_diff = (current_time - last_msg_time).total_seconds()
                msg.time = int(time_diff * TICKS_PER_BEAT * self.bpm / 60)
                self.recorded_notes.append(msg)
                last_msg_time = current_time

                # track velocity for note_on messages in non-note-only mode too
                if msg.type == "note_on" and msg.velocity > 0:
                    # add current note to velocity window with timestamp
                    velocity_window.append((current_time.timestamp(), msg.velocity))

                    # remove velocities older than the window duration
                    current_timestamp = current_time.timestamp()
                    window_start_timestamp = current_timestamp - window_duration_seconds
                    velocity_window = [
                        item
                        for item in velocity_window
                        if item[0] >= window_start_timestamp
                    ]

                    # calculate velocity stats if we have data
                    if velocity_window:
                        velocities = [v[1] for v in velocity_window]
                        avg_velocity = sum(velocities) / len(velocities)
                        min_velocity = min(velocities)
                        max_velocity = max(velocities)

                        # Update the display occasionally to avoid flooding the terminal
                        current_time_seconds = time.time()
                        time_since_last_display = (
                            current_time_seconds - last_display_update_time
                        )

                    # print update if its time
                    current_time_seconds = time.time()
                    time_since_last_print = current_time_seconds - last_stats_print_time
                    if time_since_last_print >= window_duration_seconds:
                        # print velocity stats if we have data
                        if velocity_window:
                            console.log(
                                f"{self.tag} Velocity over last 8 beats: avg={avg_velocity:.2f}, min={min_velocity}, max={max_velocity}"
                            )
                        else:
                            console.log(
                                f"{self.tag} No notes played in the last 8 beats"
                            )

                        # update the last print time
                        last_stats_print_time = current_time_seconds

        console.log(f"{self.tag} stopped recording")

        # Close the display
        if self.velocity_display:
            self.velocity_display.close()

    def save_midi(self) -> bool:
        """Saves the recorded notes to a MIDI file."""
        # Close the velocity display if it exists
        if self.velocity_display:
            self.velocity_display.close()

        if self.verbose:
            console.log(
                f"{self.tag} saving recording '{os.path.basename(self.pf_midi_recording)}'"
            )

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        track = mido.MidiTrack()
        track.name = "player"
        track.append(
            mido.MetaMessage(
                type="set_tempo",
                tempo=self.tempo,
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
            if self.verbose:
                mido.MidiFile(self.pf_midi_recording).print_tracks()
        else:
            console.log(
                f"{self.tag} failed to save recording '{os.path.basename(self.pf_midi_recording)}'"
            )

        self.recorded_notes = []

        return os.path.exists(self.pf_midi_recording)
