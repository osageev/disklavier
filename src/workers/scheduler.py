import os
import mido
from shutil import copy2
from datetime import datetime
from queue import PriorityQueue

from utils import console
from .worker import Worker

N_TICKS_PER_BEAT: int = 96  # standard


class Scheduler(Worker):
    lead_bar: bool = True
    tt_offset: int = 0
    ts_transitions: list[float] = []
    tt_all_messages: list[int] = []
    n_files_played: int = 0
    n_beats_per_segment: int = 8

    def __init__(
        self,
        params,
        bpm: int,
        log_path: str,
        recording_file_path: str,
        playlist_path: str,
        t_start: datetime,
    ):
        self.tag = params.tag
        self.lead_bar = params.lead_bar
        self.bpm = bpm
        self.tempo = mido.bpm2tempo(self.bpm)
        self.n_beats_per_segment = params.n_beats_per_segment
        self.pf_log = log_path
        self.pf_midi_recording = recording_file_path
        self.p_playlist = playlist_path
        self.td_start = t_start

        console.log(f"{self.tag} initialization complete")

    def gen_transitions(
        self, ts_offset: float = 0, n_stamps: int = 100, do_ticks: bool = False
    ) -> list[mido.MetaMessage]:
        self.tt_offset = mido.second2tick(ts_offset, N_TICKS_PER_BEAT, self.tempo)
        # TODO: ts_offset will be used to have timing start from the end of the recording
        ts_interval = self.n_beats_per_segment * 60 / self.bpm
        ts_beat_length = 60 / self.bpm  # time interval for each beat
        self.ts_transitions = [ts_offset + i * ts_interval for i in range(n_stamps)]
        console.log(
            f"{self.tag} segment interval is {ts_interval} seconds", self.ts_transitions
        )

        transitions = []

        for i, ts_transition in enumerate(self.ts_transitions):
            # transition messages
            transitions.append(
                mido.MetaMessage(
                    "text",
                    text=f"transition {i} ({ts_transition:.02f}s)",
                    time=mido.second2tick(ts_transition, N_TICKS_PER_BEAT, self.tempo),
                )
            )
            # tick messages
            if do_ticks:
                for beat in range(self.n_beats_per_segment):
                    tick_time = ts_transition + (beat * ts_beat_length)
                    transitions.append(
                        mido.MetaMessage(
                            "text",
                            text=f"tick {i}-{beat + 1} ({tick_time:.02f}s)",
                            time=mido.second2tick(
                                ts_beat_length, N_TICKS_PER_BEAT, self.tempo
                            ),
                        )
                    )

        return transitions

    def add_midi_to_queue(self, pf_midi: str, q_midi: PriorityQueue) -> float:
        midi_file = mido.MidiFile(pf_midi)
        # number of seconds/ticks from the start of playback to start playing the file
        ts_offset, tt_offset = self._get_next_transition()
        tt_abs: int = tt_offset  # track the absolute time since system start

        console.log(
            f"{self.tag} adding file to queue '{pf_midi}' with offset {tt_offset} ({ts_offset}s)"
        )

        # add messages to queue first so that the player has access ASAP
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    tt_abs += msg.time
                    if tt_abs in self.tt_all_messages:
                        # find the nearest integer that doesn't exist in tt_all_messages
                        lower_bound = tt_abs - 1
                        upper_bound = tt_abs + 1
                        while (
                            lower_bound in self.tt_all_messages
                            or upper_bound in self.tt_all_messages
                        ):
                            lower_bound -= 1
                            upper_bound += 1

                        # select the nearest available integer
                        if lower_bound not in self.tt_all_messages:
                            tt_abs = lower_bound
                        else:
                            tt_abs = upper_bound
                    self.tt_all_messages.append(tt_abs)
                    # msg.time += tt_offset
                    console.log(
                        f"{self.tag} adding message to queue: ({tt_abs}, ({msg}))"
                    )
                    q_midi.put((tt_abs, msg))

        # update midi log file
        if self._log_midi(pf_midi):
            console.log(f"{self.tag} successfully updated recording file")
        else:
            console.log(f"{self.tag} [orange]error updating recording file")

        self.n_files_played += 1

        console.log(
            f"{self.tag} added {mido.tick2second(tt_abs, N_TICKS_PER_BEAT, self.tempo):.03f} seconds of music to queue"
        )

        return mido.tick2second(tt_abs, N_TICKS_PER_BEAT, self.tempo)

    def _get_next_transition(self) -> tuple[float, int]:
        ts_offset = self.ts_transitions[self.n_files_played]
        if self.lead_bar:
            ts_offset -= 60 / self.bpm
            ts_offset = (
                ts_offset if ts_offset > 0 else 0
            )  # prevent potential negative offset on first segment

        return ts_offset, mido.second2tick(ts_offset, N_TICKS_PER_BEAT, self.tempo)

    def _log_midi(self, pf_midi: str) -> bool:
        midi_in = mido.MidiFile(pf_midi)
        midi_out = mido.MidiFile(self.pf_midi_recording)
        _, tt_offset = self._get_next_transition()

        # create playback track if it doesn't already exist
        play_track = None
        for track in midi_out.tracks:
            if track.name == "playback":
                play_track = track
                break

        if play_track is None:
            play_track = midi_out.add_track("playback")

        # copy over midi to track
        for track in midi_in.tracks:
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    msg.time += tt_offset
                    play_track.append(msg)

        # clear out old MIDI file and rewrite (TODO: risky? what if recorder writes at same time?)
        # console.log(f"{self.tag} writing out MIDI to log:")
        # midi_out.print_tracks()
        os.remove(self.pf_midi_recording)
        midi_out.save(self.pf_midi_recording)

        # copy source file
        copy2(
            pf_midi,
            os.path.join(
                self.p_playlist,
                f"{self.n_files_played:02d} {os.path.basename(pf_midi)}",
            ),
        )

        return os.path.isfile(self.pf_midi_recording)
