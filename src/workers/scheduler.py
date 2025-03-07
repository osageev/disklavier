import os
import mido
from queue import PriorityQueue
from datetime import datetime, timedelta

from utils import console
from utils.midi import csv_to_midi
from .worker import Worker

N_TICKS_PER_BEAT: int = 220
N_BEATS_TRANSITION_OFFSET: int = 8


class Scheduler(Worker):
    lead_bar: bool = True
    tt_offset: int = 0
    ts_transitions: list[float] = []
    tt_all_messages: list[int] = []
    n_files_queued: int = 0
    n_beats_per_segment: int = 8

    def __init__(
        self,
        params,
        bpm: int,
        log_path: str,
        recording_file_path: str,
        playlist_path: str,
        start_time: datetime,
        n_transitions: int,
        recording_mode: bool,
    ):
        super().__init__(params, bpm=bpm)
        self.lead_bar = params.lead_bar
        self.n_beats_per_segment = params.n_beats_per_segment
        self.pf_log = log_path
        self.pf_midi_recording = recording_file_path
        self.p_playlist = playlist_path
        self.td_start = start_time
        self.n_transitions = n_transitions
        self.recording_mode = recording_mode

        # initialize queue file
        self.raw_notes_filepath = os.path.join(
            os.path.dirname(recording_file_path), "queue_dump.csv"
        )
        self.raw_notes_file = open(self.raw_notes_filepath, "w")
        self.raw_notes_file.write("file,type,note,velocity,time\n")

        console.log(f"{self.tag} initialization complete")
        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def enqueue_midi(self, pf_midi: str, q_midi: PriorityQueue) -> float:
        midi_in = mido.MidiFile(pf_midi)
        midi_track = os.path.basename(pf_midi).split("_")[0]
        # number of seconds/ticks from the start of playback to start playing the file
        if self.recording_mode and midi_track == "player-recording":
            ts_offset, tt_offset = 0, 0
        else:
            ts_offset, tt_offset = self._get_next_transition()
        tt_abs: int = tt_offset  # track the absolute time since system start
        tt_sum: int = 0  # track the sum of all notes in the segment

        if midi_in.ticks_per_beat != N_TICKS_PER_BEAT:
            console.log(
                f"{self.tag}[red] midi file ticks per beat mismatch!\n\tfile has {midi_in.ticks_per_beat} tpb but expected {N_TICKS_PER_BEAT}"
            )

        console.log(
            f"{self.tag} adding file {self.n_files_queued} to queue '{pf_midi}' with offset {tt_offset} ({ts_offset:.02f} s, so {str(self.td_start + timedelta(seconds=ts_offset))})"
        )

        # add messages to queue first so that the player has access ASAP
        for track in midi_in.tracks:
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    tt_abs += msg.time
                    tt_sum += msg.time
                    # occasionally need to shift the message to avoid priority conflicts
                    if tt_abs in self.tt_all_messages:
                        # find the nearest integer that doesn't exist in tt_all_messages
                        tt_lower_bound = tt_abs - 1
                        tt_upper_bound = tt_abs + 1
                        while (
                            tt_lower_bound in self.tt_all_messages
                            or tt_upper_bound in self.tt_all_messages
                        ):
                            tt_lower_bound -= 1
                            tt_upper_bound += 1

                        # select the nearest available integer
                        if tt_lower_bound not in self.tt_all_messages:
                            tt_abs = tt_lower_bound
                        else:
                            tt_abs = tt_upper_bound
                    self.tt_all_messages.append(tt_abs)
                    # if self.verbose:
                    #     console.log(
                    #         f"{self.tag} adding message to queue: ({tt_abs}, ({msg}))"
                    #     )
                    q_midi.put((tt_abs, msg))
                    self.raw_notes_file.write(
                        f"{os.path.basename(pf_midi)},{msg.type},{msg.note},{msg.velocity},{tt_abs}\n"
                    )

        if (
            mido.tick2second(tt_abs, N_TICKS_PER_BEAT, self.tempo)
            > self.ts_transitions[-1]
        ):

            _ = self._gen_transitions(self.ts_transitions[-1])

        self.n_files_queued += 1

        console.log(
            f"{self.tag} added {mido.tick2second(tt_sum, N_TICKS_PER_BEAT, self.tempo):.03f} seconds of music to queue"
        )

        return mido.tick2second(tt_sum, N_TICKS_PER_BEAT, self.tempo)

    def init_outfile(self, pf_midi: str, offset_s: float = 0) -> bool:
        """Initialize a MIDI file to hold a playback recording."""
        if self.verbose:
            console.log(f"{self.tag} initializing output file with offset {offset_s} s")
        midi = mido.MidiFile()
        tick_track = mido.MidiTrack()

        # default timing messages
        tick_track.append(
            mido.MetaMessage("track_name", name=os.path.basename(pf_midi), time=0)
        )
        tick_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=36,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
        tick_track.append(mido.MetaMessage("set_tempo", tempo=self.tempo, time=0))

        # transition messages
        mm_transitions = self._gen_transitions(
            ts_offset=offset_s, n_stamps=self.n_transitions
        )
        for mm_transition in mm_transitions:
            tick_track.append(mm_transition)
        tick_track.append(mido.MetaMessage("end_of_track", time=1))

        midi.tracks.append(tick_track)

        # write to file
        midi.save(pf_midi)

        return os.path.isfile(pf_midi)

    def _gen_transitions(
        self,
        ts_offset: float = 0,
        n_stamps: int = 1,
        do_ticks: bool = True,
    ) -> list[mido.MetaMessage]:
        """Generate transition times for 8-beat MIDI files.

        The function calculates transition times for segment changes in a MIDI playback sequence.
        If the offset is not zero, the first transition occurs at the next multiple of 8 beats.

        Args:
            ts_offset (float): Time offset in seconds.
            n_stamps (int): Number of transition timestamps to generate.
            do_ticks (bool): Whether to include tick messages.

        Returns:
            list[mido.MetaMessage]: List of MIDI meta messages representing transitions and ticks.
        """
        self.tt_offset = mido.second2tick(ts_offset, N_TICKS_PER_BEAT, self.tempo)
        ts_interval = self.n_beats_per_segment * 60 / self.bpm
        ts_beat_length = 60 / self.bpm  # time interval for each beat

        # Adjust ts_offset to the next interval
        if ts_offset % ts_interval < ts_beat_length * N_BEATS_TRANSITION_OFFSET:
            ts_offset = ((ts_offset // ts_interval) + 1) * ts_interval

        seg_range = range(n_stamps) if self.recording_mode else range(1, n_stamps + 1)
        self.ts_transitions.extend([ts_offset + i * ts_interval for i in seg_range])

        if self.verbose:
            console.log(
                f"{self.tag} segment interval is {ts_interval} seconds (from {self.td_start})",
                [
                    f"{t:07.03f}s -> {str(self.td_start + timedelta(seconds=t))}"
                    for t in self.ts_transitions
                ],
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

    def _get_next_transition(self) -> tuple[float, int]:
        if self.verbose:
            console.log(f"{self.tag} transition times:\n\t{self.ts_transitions[-5:]}")
        ts_offset = self.ts_transitions[
            self.n_files_queued - 1 if self.recording_mode else self.n_files_queued
        ]
        if self.lead_bar:
            ts_offset -= 60 / self.bpm
            ts_offset = (
                ts_offset if ts_offset > 0 else 0
            )  # prevent potential negative offset on first segment

        return ts_offset, mido.second2tick(ts_offset, N_TICKS_PER_BEAT, self.tempo)

    def _log_midi(self, pf_midi: str) -> bool:
        midi_in = mido.MidiFile(pf_midi)
        out_path = os.path.join(self.p_playlist, os.path.basename(pf_midi))

        console.log(f"{self.tag} copying midi to '{out_path}'")
        midi_in.save(out_path)

        return os.path.isfile(out_path)

    def queue_to_midi(self) -> bool:
        return csv_to_midi(
            self.raw_notes_filepath,
            os.path.join(os.path.dirname(self.raw_notes_filepath), "playback.mid"),
        )
