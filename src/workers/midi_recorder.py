import os
import mido
import time
from datetime import datetime
from threading import Thread, Event

from .worker import Worker
from utils import console, tick
from utils.midi import TICKS_PER_BEAT

from typing import List, Optional


class MidiRecorder(Worker):
    recorded_notes: List[mido.Message] = []
    is_recording: bool = False
    n_ticks: int = 0
    stop_event: Optional[Event] = None
    midi_thread: Thread
    ts_window_duration: float = 1.0

    # published velocity statistics
    _avg_velocity: float = 0.0
    _min_velocity: int = 0
    _max_velocity: int = 0
    _velocity_window: List[tuple] = []

    @property
    def avg_velocity(self) -> float:
        """
        Get the average velocity of notes in the current window.

        Returns
        -------
        float
            Average velocity of notes.
        """
        return self._avg_velocity

    @property
    def min_velocity(self) -> int:
        """
        Get the minimum velocity of notes in the current window.

        Returns
        -------
        int
            Minimum velocity of notes.
        """
        return self._min_velocity

    @property
    def max_velocity(self) -> int:
        """
        Get the maximum velocity of notes in the current window.

        Returns
        -------
        int
            Maximum velocity of notes.
        """
        return self._max_velocity

    def __init__(
        self,
        params,
        bpm: int,
        recording_file_path: str,
    ):
        super().__init__(params, bpm=bpm)
        self.pf_midi_recording = recording_file_path
        self.ts_window_duration = 60.0 / self.bpm  # 1 beat in seconds

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def manual_record(self) -> float:
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
        track.append(mido.MetaMessage("track_name", name="player", time=0))
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
                            self.recorded_notes = []

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
                        msg = msg.copy(time=0)
                    else:
                        msg = msg.copy(
                            time=int(
                                (current_time - last_note_time).total_seconds()
                                * TICKS_PER_BEAT
                                * self.bpm
                                / 60
                            )
                        )
                    track.append(msg)
                    self.recorded_notes.append(msg)
                    console.log(f"{self.tag} \t{msg}")
                    last_note_time = current_time
        return -1.0

    def _passive_record(self, td_start: datetime, stop_event: Event) -> None:
        """
        Passively records notes from the MIDI input.

        Parameters
        ----------
        td_start : datetime
            Start time to calculate message times relative to.
        stop_event : Event
            Event that signals when to stop recording.
        """
        # wait until td_start
        wait_time = (td_start - datetime.now()).total_seconds()
        if wait_time > 0:
            console.log(
                f"{self.tag} waiting {wait_time:.2f}s until midi recording start"
            )
            time.sleep(wait_time)

        # initialize velocity tracking variables
        self._velocity_window = []  # list of (timestamp, velocity) tuples

        if self.verbose:
            console.log(f"{self.tag} listening on port '{self.params.midi_port}'")
        self.is_recording = True

        last_msg_time = td_start
        with mido.open_input(self.params.midi_port) as inport:  # type: ignore
            for msg in inport:
                # record note
                current_time = datetime.now()
                # calculate time in ticks since last message
                time_diff = (current_time - last_msg_time).total_seconds()
                msg = msg.copy(time=int(time_diff * TICKS_PER_BEAT * self.bpm / 60))
                self.recorded_notes.append(msg)
                last_msg_time = current_time

                # update velocity stats
                if msg.type == "note_on" and msg.velocity > 0:
                    self._update_velocity_stats(current_time.timestamp(), msg)

                # check stop conditions
                if not self.is_recording:
                    break
                if stop_event is not None and stop_event.is_set():
                    self.is_recording = False
                    break

        console.log(f"{self.tag} stopped recording ({len(self.recorded_notes)} notes)")

    def _update_velocity_stats(self, current_time: float, msg: mido.Message) -> None:
        """
        Update velocity statistics based on current velocity window.

        Parameters
        ----------
        current_time : datetime
            Current time of the recording.
        msg : mido.Message
            Current MIDI message.
        """

        # add current note to velocity window with timestamp
        v = msg.velocity  # type: ignore
        self._velocity_window.append((current_time, v))

        # remove velocities older than the window duration
        window_start_timestamp = current_time - self.ts_window_duration
        self._velocity_window = [
            item for item in self._velocity_window if item[0] >= window_start_timestamp
        ]
        if self._velocity_window:
            velocities = [v[1] for v in self._velocity_window]
            self._avg_velocity = sum(velocities) / len(velocities)
            self._min_velocity = min(velocities)
            self._max_velocity = max(velocities)
        else:
            # No data in the window
            self._avg_velocity = 0.0
            self._min_velocity = 0
            self._max_velocity = 0

        # print velocity stats if we have data
        if self._velocity_window:
            console.log(
                f"{self.tag} velocity stat updated: avg={self._avg_velocity:.2f}, min={self._min_velocity}, max={self._max_velocity}"
            )
        else:
            console.log(f"{self.tag} no notes played in the last beat")

    def save_midi(self) -> bool:
        """Saves the recorded notes to a MIDI file."""

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

    def start_recording(self, td_start: datetime) -> Event:
        """
        Starts midi recording in a separate thread.

        Parameters
        ----------
        td_start : datetime
            Start time for the recording.

        Returns
        -------
        Event
            The stop event that can be used to signal the recording to stop.
        """
        self.stop_event = Event()
        self.midi_thread = Thread(
            target=self._passive_record,
            args=(td_start, self.stop_event),
            name="midi recorder",
            daemon=True,
        )
        self.midi_thread.start()
        return self.stop_event

    def stop_recording(self) -> bool:
        """
        Stops the midi recording thread.

        Returns
        -------
        bool
            True if recording was successfully stopped, False otherwise.
        """
        if self.is_recording and self.stop_event is not None:
            console.log(f"{self.tag} stopping midi recording")
            self.stop_event.set()
            if self.midi_thread is not None:
                self.midi_thread.join(0.1)
            self.is_recording = False
            if self.midi_thread.is_alive():
                console.log(f"{self.tag} midi recording thread is still running")
            return True
        else:
            console.log(f"{self.tag} midi recording not active")
            console.log(f"{self.tag} \tis_recording: {self.is_recording}")
            console.log(f"{self.tag} \tstop_event: {self.stop_event}")
            return False
