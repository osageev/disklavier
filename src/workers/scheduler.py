import os
import mido
from queue import PriorityQueue
from datetime import datetime, timedelta

from .worker import Worker
from utils import basename, console
from utils.midi import csv_to_midi
from utils.constants import TICKS_PER_BEAT

from typing import List, Optional, Tuple


class Scheduler(Worker):
    lead_bar: bool = True
    tt_offset: int = 0
    ts_transitions: List[List] = []
    tt_all_messages: List[int] = []
    n_files_queued: int = 0
    n_beats_per_segment: int = 8
    queued_files: List[str] = []
    first_file_avg_velocity: Optional[float] = None
    previous_track_name: Optional[str] = None

    def __init__(
        self,
        params,
        bpm: int,
        log_path: str,
        playlist_path: str,
        start_time: datetime,
        n_transitions: int,
        recording_mode: bool,
    ):
        super().__init__(params, bpm=bpm)
        self.lead_bar = params.lead_bar
        self.n_beats_per_segment = params.n_beats_per_segment
        self.pf_log = log_path
        self.p_playlist = playlist_path
        self.td_start = start_time
        self.n_transitions = n_transitions
        self.recording_mode = recording_mode

        # initialize queue file
        self.raw_notes_filepath = os.path.join(self.pf_log, "queue_dump.csv")
        self.raw_notes_file = open(self.raw_notes_filepath, "w")
        self.raw_notes_file.write("file,type,note,velocity,time\n")

        console.log(f"{self.tag} initialization complete")
        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def enqueue_midi(
        self,
        pf_midi: str,
        q_piano: PriorityQueue,
        q_gui: Optional[PriorityQueue] = None,
        similarity: Optional[float] = None,
    ) -> Tuple[float, float, int]:
        midi_in = mido.MidiFile(pf_midi)

        # --- calculate offset and track transition ---
        # (number of seconds/ticks from the start of playback to start playing the file)
        current_track_name = basename(pf_midi).split("_")[0]
        is_new_track_segment = False
        if self.n_files_queued == 0:
            # starting from player recording
            if self.recording_mode and "player" in basename(pf_midi):
                n_recorded_beats = None
                for msg in midi_in.tracks[1]:
                    if msg.type == "text" and msg.text.startswith("beat"):
                        n_recorded_beats = int(msg.text.split(" ")[1]) - 1

                # recording was broken for some reason
                if n_recorded_beats is None:
                    if self.verbose:
                        console.log(
                            f"{self.tag}[yellow] no beats found in {basename(pf_midi)}, defaulting to zero offset[/yellow]"
                        )
                    ts_offset, tt_offset = 0, 0
                else:
                    # offset = system start delay - segment length rounded up to nearest beat
                    # this ensures that the recording ends at the first beat
                    ts_offset = self.n_beats_per_segment * 60 / self.bpm - (
                        n_recorded_beats * 60 / self.bpm
                    )
                    tt_offset = mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)
                    if self.verbose:
                        console.log(
                            f"{self.tag} found {n_recorded_beats} beats in {basename(pf_midi)}, setting offset to {ts_offset:.02f} s so that end time is {self.td_start + timedelta(seconds=ts_offset) + timedelta(seconds=mido.tick2second(n_recorded_beats * TICKS_PER_BEAT, TICKS_PER_BEAT, self.tempo)):%H:%M:%S.%f}"
                        )
            else:
                # kickstarting from a file
                ts_offset, tt_offset = self._get_next_transition()
        else:
            # not first segment
            ts_offset, tt_offset = self._get_next_transition()
            if (
                self.previous_track_name is not None
                and current_track_name != self.previous_track_name
            ):
                is_new_track_segment = True

        console.log(f"{self.tag} {self.previous_track_name} -> {current_track_name}")
        self.previous_track_name = current_track_name
        console.log(f"{self.tag} {self.n_files_queued} is_new_track_segment: {is_new_track_segment}")
        self.ts_transitions[self.n_files_queued][1] = is_new_track_segment
        console.log(
            [
                f"[{t[1]}] {t[0]:02.01f}  -> {self.td_start + timedelta(seconds=t[0]):%H:%M:%S.%f}"
                for t in self.ts_transitions
                if self.td_start + timedelta(seconds=t[0]) < datetime.now() + timedelta(seconds=12)
            ],
        )

        tt_abs: int = tt_offset  # absolute time since system start
        tt_sum: int = 0  # time sum of all messages in the segment
        tt_max_abs_in_segment: int = (
            tt_offset  # maximum absolute tick time in the segment
        )

        if midi_in.ticks_per_beat != TICKS_PER_BEAT:
            raise ValueError(
                f"{self.tag}[red] midi file ticks per beat mismatch!\n\tfile has {midi_in.ticks_per_beat} tpb but expected {TICKS_PER_BEAT}"
            )

        # --- velocity normalization ---
        if self.params.scale_velocity:
            note_on_messages = []
            for track in midi_in.tracks:
                if track[0].type == "track_name" and track[0].name == "metronome":
                    continue
                for msg in track:
                    if msg.type == "note_on" and msg.velocity > 0:
                        note_on_messages.append(msg)

            if note_on_messages:
                velocities = [m.velocity for m in note_on_messages]
                current_avg_velocity = (
                    sum(velocities) / len(velocities) if velocities else 0.0
                )

                if self.first_file_avg_velocity is None:
                    if current_avg_velocity > 0:
                        self.first_file_avg_velocity = current_avg_velocity
                        console.log(
                            f"{self.tag} first file enqueued, average velocity set to {self.first_file_avg_velocity:.2f}"
                        )
                    else:
                        console.log(
                            f"{self.tag} [yellow]first file enqueued, but no valid notes with positive velocity to set average velocity.[/yellow]"
                        )
                elif self.first_file_avg_velocity is not None:
                    console.log(
                        f"{self.tag} normalizing to target average velocity: {self.first_file_avg_velocity:.2f}. Original avg: {current_avg_velocity:.2f}"
                    )

                    # 1. clamp velocity to below max
                    for msg in note_on_messages:
                        if msg.velocity > self.params.max_velocity:
                            console.log(
                                f"{self.tag}\t[grey50]clamping velocity: {msg.velocity} -> {self.params.max_velocity}[/grey50]"
                            )
                            msg.velocity = self.params.max_velocity

                    velocities_after_clamp = [m.velocity for m in note_on_messages]
                    avg_velocity_after_clamp = (
                        sum(velocities_after_clamp) / len(velocities_after_clamp)
                        if velocities_after_clamp
                        else 0.0
                    )

                    # 2. rescale velocity
                    scale_factor = (
                        self.first_file_avg_velocity / avg_velocity_after_clamp
                    )
                    console.log(f"{self.tag} scaling velocity by {scale_factor:.2f}")
                    for msg in note_on_messages:
                        new_velocity = int(round(msg.velocity * scale_factor))
                        msg.velocity = max(
                            0, min(self.params.max_velocity, new_velocity)
                        )

                    final_velocities = [m.velocity for m in note_on_messages]
                    final_avg_vel = sum(final_velocities) / len(final_velocities)
                    console.log(
                        f"{self.tag} velocities normalized. avg after clamp: {avg_velocity_after_clamp:.2f}, final avg: {final_avg_vel:.2f}"
                    )
            else:
                console.log(
                    f"{self.tag} no note_on messages found in {basename(pf_midi)} for velocity normalization."
                )

        console.log(
            f"{self.tag} adding file {self.n_files_queued} to queue '{pf_midi}' with offset {tt_offset} ({ts_offset:.02f} s -> {self.td_start + timedelta(seconds=ts_offset):%H:%M:%S.%f})"
        )

        # --- add messages to queue(s) ---
        # add messages to queue first so that the player has access ASAP
        for track in midi_in.tracks:
            if track[0].type == "track_name":
                if track[0].name == "metronome":
                    console.log(f"{self.tag} skipping metronome track")
                    continue
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    tt_abs += msg.time
                    tt_sum += msg.time
                    # occasionally need to shift the message to avoid priority collisions
                    current_tt_abs = tt_abs  # store original intended time
                    if current_tt_abs in self.tt_all_messages:
                        # find the nearest integer that doesn't exist in tt_all_messages
                        tt_lower_bound = current_tt_abs - 1
                        tt_upper_bound = current_tt_abs + 1
                        while (
                            tt_lower_bound in self.tt_all_messages
                            or tt_upper_bound in self.tt_all_messages
                        ):
                            tt_lower_bound -= 1
                            tt_upper_bound += 1

                        # select the nearest available integer
                        if tt_lower_bound not in self.tt_all_messages:
                            current_tt_abs = tt_lower_bound
                        else:
                            current_tt_abs = tt_upper_bound
                    self.tt_all_messages.append(current_tt_abs)
                    tt_max_abs_in_segment = max(
                        tt_max_abs_in_segment, current_tt_abs
                    )  # update max tick time

                    # console.log(f"{self.tag} adding message to queue: ({current_tt_abs}, ({msg}))")

                    q_piano.put((current_tt_abs, msg))

                    # --- add to gui queue ---
                    if q_gui is not None:
                        # TODO: make this 10 seconds a global parameter
                        tt_delay = mido.second2tick(10, TICKS_PER_BEAT, self.tempo)
                        q_gui.put(
                            (
                                current_tt_abs - tt_delay,
                                (similarity if similarity is not None else 1.0, msg),
                            )
                        )

                    # --- write to raw notes file ---
                    # edge case, but it does happen sometimes that multiple recorded notes start at 0, resulting in one note getting bumped to time -1
                    if current_tt_abs < 0:
                        current_tt_abs = 0
                    self.raw_notes_file.write(
                        f"{os.path.basename(pf_midi)},{msg.type},{msg.note},{msg.velocity},{current_tt_abs}\n"
                    )

        # --- generate transitions, update trackers ---
        if (
            mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)
            > self.ts_transitions[-1][0]
        ):

            _ = self._gen_transitions(self.ts_transitions[-1][0])

        self.n_files_queued += 1
        self.queued_files.append(basename(pf_midi))
        ts_seg_len = mido.tick2second(tt_sum, TICKS_PER_BEAT, self.tempo)

        console.log(
            f"{self.tag} added {ts_seg_len:.03f} seconds of music to queue ({self.n_files_queued} files in queue)"
        )

        if self._copy_midi(pf_midi):
            console.log(f"{self.tag} copied {basename(pf_midi)} to playlist folder")

        return ts_seg_len, ts_offset, tt_max_abs_in_segment

    def init_schedule(self, pf_midi: str, offset_s: float = 0):
        """Initialize a MIDI file to hold a playback recording."""
        if self.verbose:
            console.log(f"{self.tag} initializing output file with offset {offset_s} s")
        midi = mido.MidiFile()
        tick_track = mido.MidiTrack()

        # default timing messages
        tick_track.append(
            mido.MetaMessage("track_name", name=basename(pf_midi), time=0)
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

    def _gen_transitions(
        self,
        ts_offset: float = 0,
        n_stamps: int = 1,
        do_ticks: bool = True,
    ) -> list[mido.MetaMessage]:
        ts_offset = 0
        self.tt_offset = mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)
        ts_beat_length = 60 / self.bpm  # time duration of each beat
        ts_interval = self.n_beats_per_segment * ts_beat_length

        # --- adjust offset to next transition ---
        # INFO: leaving this here in case i ever need it again because the logic took some figuring out
        # if ts_offset % ts_interval < ts_beat_length * N_BEATS_TRANSITION_OFFSET:
        #     if self.verbose:
        #         console.log(
        #             f"{self.tag} adjusting ts_offset from {ts_offset:.02f} s to {((ts_offset // ts_interval) + 1) * ts_interval:.02f} s"
        #         )
        #     ts_offset = ((ts_offset // ts_interval) + 1) * ts_interval
        self.ts_transitions.extend(
            [[ts_offset + i * ts_interval, False] for i in range(n_stamps + 1)]
        )

        if self.verbose:
            console.log(
                f"{self.tag} segment interval is {ts_interval:.03f} seconds",
                # [
                #     f"{t:02.01f}  -> {self.td_start + timedelta(seconds=t):%H:%M:%S.%f}"
                #     for t in self.ts_transitions[:-5]
                # ],
            )

        transitions = []
        for i, (ts_transition, is_track_change) in enumerate(self.ts_transitions):
            # transition messages
            transitions.append(
                mido.MetaMessage(
                    "text",
                    text=f"transition {i+1} ({ts_transition:.02f}s)",
                    time=mido.second2tick(ts_transition, TICKS_PER_BEAT, self.tempo),
                )
            )
            # tick messages
            if do_ticks:
                transitions[-1].time = 0  # transition occurs at tick time
                for beat in range(self.n_beats_per_segment):
                    tick_time = ts_transition + (beat * ts_beat_length)
                    transitions.append(
                        mido.MetaMessage(
                            "text",
                            text=f"tick {i}-{beat + 1} ({tick_time:.02f}s)",
                            time=mido.second2tick(
                                ts_beat_length, TICKS_PER_BEAT, self.tempo
                            ),
                        )
                    )

        return transitions

    def _get_next_transition(self) -> Tuple[float, int]:
        ts_offset, is_track_change = self.ts_transitions[
            self.n_files_queued  # - 1 if self.recording_mode else self.n_files_queued
        ]
        if self.lead_bar:
            ts_offset -= 60 / self.bpm

        # print selected range from ts_transitions
        if self.verbose:
            selected_idx = (
                self.n_files_queued - 1 if self.recording_mode else self.n_files_queued
            )
            start_idx = max(0, selected_idx - 2)
            end_idx = min(len(self.ts_transitions) - 1, selected_idx + 2)
            transitions = []
            for i in range(start_idx, end_idx + 1):
                t_time = self.td_start + timedelta(seconds=self.ts_transitions[i][0])
                if i == selected_idx:
                    transitions.append(f"[bold]{t_time.strftime('%H:%M:%S.%f')}[/bold]")
                else:
                    transitions.append(t_time.strftime("%H:%M:%S.%f"))
            console.log(f"{self.tag} transitions: {transitions}")

        return ts_offset, mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)

    def _copy_midi(self, pf_midi: str) -> bool:
        """Copy the MIDI file to the playlist folder.

        Parameters
        ----------
        pf_midi : str
            Path to the MIDI file to copy.

        Returns
        -------
        bool
            True if the MIDI file was copied successfully, False otherwise.
        """
        midi = mido.MidiFile(pf_midi)
        out_path = os.path.join(self.p_playlist, os.path.basename(pf_midi))
        midi.save(out_path)

        return os.path.isfile(out_path)

    def queue_to_midi(self, out_path: str) -> bool:
        return csv_to_midi(
            self.raw_notes_filepath,
            out_path,
            verbose=self.verbose,
        )

    def get_current_file(self) -> Optional[tuple[str, int]]:
        if not self.queued_files:
            return None

        current_time = datetime.now()
        elapsed_seconds = (current_time - self.td_start).total_seconds()

        # we haven't started playing yet
        if elapsed_seconds < self.ts_transitions[0][0]:
            return None

        # find which segment we're in
        current_segment = 0
        for i in range(1, len(self.ts_transitions)):
            if elapsed_seconds < self.ts_transitions[i][0]:
                current_segment = i - 1
                break
            if i == len(self.ts_transitions) - 1:
                current_segment = i

        # make sure we have enough files
        if current_segment < len(self.queued_files):
            return self.queued_files[current_segment], current_segment

        return None

    def set_start_time(self, td_start: datetime):
        self.td_start = td_start
        console.log(f"{self.tag} start time set to {self.td_start}")
