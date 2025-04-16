import os
import time
from PySide6 import QtCore
from threading import Thread
from rich.table import Table
from queue import PriorityQueue
from datetime import datetime, timedelta
import numpy as np
import mido
import uuid

import workers
from workers import Staff
from utils import basename, console, midi, write_log, panther
from utils.midi import TICKS_PER_BEAT

from typing import Optional


class RunWorker(QtCore.QThread):
    """
    worker thread that handles the main application run loop.
    """

    tag = "[#008700]runner[/#008700]:"

    # session tracking
    ts_queue = 0
    n_files_queued = 0
    pf_augmentations = None

    # signals
    s_status = QtCore.Signal(str)
    s_start_time = QtCore.Signal(datetime)
    s_switch_to_pr = QtCore.Signal(object)
    s_transition_times = QtCore.Signal(list)

    # queues
    q_playback = PriorityQueue()
    q_gui = PriorityQueue()

    # Player Tracking State
    last_checked_beat: float = 0.0
    previous_player_embedding: Optional[np.ndarray] = None
    playback_cutoff_tick: float = float("inf")
    player_embedding_diff_threshold: float = 0.1  # TODO: Tune this value (Phase 4)
    player_segment_beat_interval: int = 8
    adjustment_lookahead_s: float = 3.0

    stop_requested: bool = False  # Flag to control the main loop

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.args = main_window.args
        self.params = main_window.params
        self.staff: Staff = main_window.workers
        self.td_system_start = main_window.td_system_start
        self.tempo = mido.bpm2tempo(self.params.bpm)
        self.player_segment_tick_interval = (
            self.player_segment_beat_interval * TICKS_PER_BEAT
        )

        # connect signals
        self.s_switch_to_pr.connect(self.main_window.switch_to_piano_roll)

        # file paths from main window
        self.p_log = main_window.p_log
        self.p_playlist = main_window.p_playlist
        self.pf_master_recording = main_window.pf_master_recording
        self.pf_system_recording = main_window.pf_system_recording
        self.pf_player_query = main_window.pf_player_query
        self.pf_player_accompaniment = main_window.pf_player_accompaniment
        self.pf_schedule = main_window.pf_schedule
        self.pf_playlist = (
            main_window.pf_playlist
        )  # This seems redundant, check main_window.py

        # metronome
        self.metronome = workers.Metronome(
            self.params.metronome, self.params.bpm, self.td_system_start
        )

    def run(self):
        # get seed
        match self.params.initialization:
            case "recording":  # collect user recording
                if self.args.replay:  # use old recording
                    from pretty_midi import PrettyMIDI

                    ts_recording_len = PrettyMIDI(self.pf_player_query).get_end_time()
                    console.log(
                        f"{self.tag} calculated time of last recording {ts_recording_len}"
                    )
                    if ts_recording_len == 0:
                        raise ValueError("no recording found")
                else:
                    ts_recording_len = self.staff.midi_recorder.run()
                self.pf_seed = self.pf_player_query

                # augment
                if self.params.seed_rearrange or self.params.seed_remove:
                    self.pf_augmentations = self.augment_midi(self.pf_seed)
                    console.log(
                        f"{self.tag} got {len(self.pf_augmentations)} augmentations:\n\t{self.pf_augmentations}"
                    )
            case "audio":
                self.pf_player_query = self.pf_player_query.replace(".mid", ".wav")
                ts_recording_len = self.staff.audio_recorder.record_query(
                    self.pf_player_query
                )
                embedding = self.staff.seeker.get_embedding(
                    self.pf_player_query, model="clap"
                )
                console.log(
                    f"{self.tag} got embedding {embedding.shape} from pantherino"
                )
                best_match, best_similarity = self.staff.seeker.get_match(
                    embedding, metric="clap-sgm"
                )
                console.log(
                    f"{self.tag} got best match '{best_match}' with similarity {best_similarity}"
                )
                self.pf_seed = best_match
            case "kickstart":  # use specified file as seed
                try:
                    if self.params.kickstart_path:
                        self.pf_seed = self.params.kickstart_path
                        console.log(
                            f"{self.tag} [cyan]KICKSTART[/cyan] - '{self.pf_seed}'"
                        )
                except AttributeError:
                    console.log(
                        f"{self.tag} no file specified to kickstart from, choosing randomly"
                    )
                    self.pf_seed = self.staff.seeker.get_random()
                    console.log(
                        f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'"
                    )
            case "random" | _:  # choose random file from library
                self.pf_seed = self.staff.seeker.get_random()
                console.log(f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'")

        if self.params.seeker.mode == "playlist":
            # TODO: implement playlist mode using generated csv's
            raise NotImplementedError("playlist mode not implemented")

        self.staff.scheduler.init_schedule(
            self.pf_schedule,
            ts_recording_len if self.params.initialization == "recording" else 0,
        )
        console.log(f"{self.tag} successfully initialized recording")

        # --- Start Timing and Workers ---
        # Calculate precise start time
        # self.td_start = datetime.now() + timedelta(seconds=self.params.startup_delay)
        td_start = datetime.now() + timedelta(seconds=self.params.startup_delay)
        self.td_start = td_start  # Ensure RunWorker uses the emitted start time
        console.log(
            f"{self.tag} Calculated start time: {td_start.strftime('%y-%m-%d %H:%M:%S.%f')[:-3]}"
        )
        # add seed to queue
        self._queue_file(self.pf_seed, None)

        # add first match to queue
        if self.pf_augmentations is None:
            pf_next_file, similarity = self.staff.seeker.get_next()
            self._queue_file(pf_next_file, similarity)

        try:
            self.s_start_time.emit(td_start)
            self.staff.scheduler.td_start = td_start
            self.metronome.td_start = td_start
            self.metronome.start()

            # Start audio recording
            self.e_audio_stop = self.staff.audio_recorder.start_recording(td_start)

            # Switch to piano roll view
            self.s_switch_to_pr.emit(self.q_gui)

            # Start player thread
            self.staff.player.td_start = td_start
            self.staff.player.td_last_note = td_start  # Reset last note time
            self.staff.player.set_runner_ref(self)  # <<< CONNECT PLAYER TO RUNNER
            self.thread_player = Thread(
                target=self.staff.player.play, name="player", args=(self.q_playback,)
            )
            self.thread_player.start()

            # Start midi recording thread
            self.midi_stop_event = self.staff.midi_recorder.start_recording(td_start)
            self.main_window.midi_stop_event = (
                self.midi_stop_event
            )  # Pass stop event to main window
            self.staff.player.set_recorder(
                self.staff.midi_recorder
            )  # Connect recorder for velocity

            # --- Queue Initial Augmentations ---
            if self.pf_augmentations is not None:
                for aug in self.pf_augmentations:
                    self._queue_file(aug, None)

            # --- Main Run Loop ---
            console.log(f"{self.tag} Starting main run loop...")
            current_file = ""
            last_time = time.time()
            self.stop_requested = False  # Reset stop flag

            while not self.stop_requested:
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time

                # Check if player thread is still alive
                if not self.thread_player.is_alive():
                    console.log(
                        f"{self.tag} Player thread finished or terminated. Exiting run loop."
                    )
                    break

                # Update status with current file
                playing_file = self.staff.scheduler.get_current_file()
                if playing_file is not None and current_file != playing_file:
                    current_file = playing_file
                    console.log(f"{self.tag} Now playing '{current_file}'")
                    self.s_status.emit(f"Now playing '{current_file}'")

                # Maintain playback queue buffer
                if (
                    self.ts_queue < self.params.startup_delay * 2
                ):  # TODO: Use a better threshold?
                    if self.n_files_queued < self.params.n_transitions:
                        pf_next_file, similarity = self.staff.seeker.get_next()
                        self._queue_file(pf_next_file, similarity)
                        if self.args.verbose:
                            console.log(
                                f"{self.tag} Queue buffer low ({self.ts_queue:.1f}s). Queued '{basename(pf_next_file)}'."
                            )
                    else:
                        if self.args.verbose:
                            console.log(
                                f"{self.tag} Reached transition limit ({self.params.n_transitions}), not queueing more files based on buffer time."
                            )

                # --- Player Tracking Logic ---
                current_elapsed_s = (datetime.now() - self.td_start).total_seconds()
                current_beat = current_elapsed_s * self.params.bpm / 60.0

                if (
                    current_beat // self.player_segment_beat_interval
                    > self.last_checked_beat // self.player_segment_beat_interval
                ):
                    if self.args.verbose:
                        console.log(
                            f"{self.tag} Checking player embedding at beat {current_beat:.2f}"
                        )
                    self.last_checked_beat = current_beat
                    try:
                        self._check_player_embedding()
                    except Exception as e:
                        console.print_exception(show_locals=True)
                        console.log(
                            f"{self.tag} [red]Error checking player embedding: {e}"
                        )
                # --- End Player Tracking ---

                # Update queue time estimate
                self.ts_queue -= elapsed

                # Check if playback queue is empty (player might finish early)
                if self.q_playback.empty() and not self.thread_player.is_alive():
                    console.log(
                        f"{self.tag} Playback queue empty and player thread stopped. Exiting loop."
                    )
                    break

                # Small sleep to prevent busy-waiting
                time.sleep(0.01)

            # --- Loop End ---
            console.log(f"{self.tag} Exited main run loop.")
            if self.stop_requested:
                console.log(f"{self.tag} Stop was requested.")

            # Wait for player thread to finish naturally if it hasn't already
            if self.thread_player.is_alive():
                console.log(f"{self.tag} Waiting for player thread to finish...")
                self.thread_player.join(timeout=5.0)  # Wait up to 5 seconds
                if self.thread_player.is_alive():
                    console.log(
                        f"{self.tag} [yellow]Player thread did not finish cleanly."
                    )

            console.log(f"{self.tag} Playback complete or stopped.")
            self.s_status.emit("Playback complete")

        except KeyboardInterrupt:
            console.log(f"{self.tag}[yellow] CTRL + C detected, initiating shutdown...")
            self.stop_requested = True  # Signal shutdown
        except Exception as e:
            console.print_exception(show_locals=True)
            console.log(
                f"{self.tag} [red bold]Unexpected error in run loop: {e}. Initiating shutdown..."
            )
            self.stop_requested = True  # Signal shutdown

        # Ensure shutdown happens even if loop exits unexpectedly
        self.shutdown()

    def shutdown(self):
        console.log(f"{self.tag}[yellow] Shutdown initiated...")
        self.stop_requested = True  # Ensure flag is set

        # --- Stop Threads and Workers ---
        # Stop Player by clearing queue and joining thread
        console.log(f"{self.tag} Clearing playback queue...")
        while not self.q_playback.empty():
            try:
                self.q_playback.get_nowait()
            except Exception:
                pass  # Ignore errors during queue clearing
        if hasattr(self, "thread_player") and self.thread_player.is_alive():
            console.log(f"{self.tag} Waiting for player thread to stop...")
            self.thread_player.join(timeout=1.0)
            if self.thread_player.is_alive():
                console.log(
                    f"{self.tag} [yellow]Player thread did not stop after clearing queue."
                )

        # Stop Metronome
        if hasattr(self, "metronome") and self.metronome.isRunning():
            console.log(f"{self.tag} Stopping metronome...")
            self.metronome.stop()
            self.metronome.wait(500)  # Wait for thread to finish

        # Stop Audio Recorder
        if hasattr(self, "e_audio_stop") and self.e_audio_stop is not None:
            console.log(f"{self.tag} Stopping audio recording...")
            self.staff.audio_recorder.stop_recording()  # Assumes this handles thread shutdown

        # Stop MIDI Recorder
        if hasattr(self, "midi_stop_event") and self.midi_stop_event is not None:
            console.log(f"{self.tag} Stopping MIDI recording...")
            stopped = self.staff.midi_recorder.stop_recording()
            if stopped:
                self.staff.midi_recorder.save_midi(self.pf_player_accompaniment)
            else:
                console.log(
                    f"{self.tag} [yellow]MIDI recorder was not active or failed to stop."
                )

        # --- Final Saving and Cleanup ---
        console.log(f"{self.tag} Finalizing recordings and logs...")
        # Close raw notes file if scheduler exists and file is open
        if (
            hasattr(self.staff, "scheduler")
            and hasattr(self.staff.scheduler, "raw_notes_file")
            and not self.staff.scheduler.raw_notes_file.closed
        ):
            self.staff.scheduler.raw_notes_file.close()

        # Convert queue to MIDI (might be empty if shutdown early)
        if os.path.exists(self.staff.scheduler.raw_notes_filepath):
            _ = self.staff.scheduler.queue_to_midi(self.pf_system_recording)
            _ = midi.combine_midi_files(
                [
                    self.pf_system_recording,
                    self.pf_player_accompaniment,
                    self.pf_schedule,
                ],
                self.pf_master_recording,
            )
        else:
            console.log(
                f"{self.tag} [yellow]Raw notes file not found, skipping system recording generation."
            )

        # Print playlist
        if os.path.exists(self.pf_playlist):
            table = Table(title="PLAYLIST")
            try:
                with open(self.pf_playlist, mode="r") as file:
                    headers = file.readline().strip().split(",")
                    for header in headers:
                        table.add_column(header)
                    for line in file:
                        row = line.strip().split(",")
                        if len(row) >= 3:
                            row[2] = os.path.basename(row[2])  # only print filenames
                        table.add_row(*row)
                console.print(table)
            except Exception as e:
                console.log(
                    f"{self.tag} [yellow]Could not read or print playlist file: {e}"
                )
        else:
            console.log(
                f"{self.tag} [yellow]Playlist file not found: {self.pf_playlist}"
            )

        # Generate piano roll visualization
        if os.path.exists(self.pf_master_recording):
            console.log(f"{self.tag} Generating piano roll visualization...")
            try:
                midi.generate_piano_roll(self.pf_master_recording)
            except Exception as e:
                console.log(f"{self.tag} [yellow]Failed to generate piano roll: {e}")
        else:
            console.log(
                f"{self.tag} [yellow]Master recording not found, skipping piano roll generation."
            )

        # Save console log
        try:
            console.save_text(
                os.path.join(
                    self.p_log, f"{self.td_system_start.strftime('%y%m%d-%H%M%S')}.log"
                )
            )
        except Exception as e:
            print(
                f"Error saving console log: {e}"
            )  # Use print as console might be broken

        console.log(f"{self.tag}[green bold] Shutdown complete.")
        # Optional: Signal main window about completion?
        # self.finished.emit() # If using QThread's finished signal

    def _queue_file(self, file_path: str, similarity: float | None) -> None:
        # Check if file exists before queueing
        if not os.path.exists(file_path):
            console.log(
                f"{self.tag} [red]Error: File not found, cannot queue: {file_path}"
            )
            return

        ts_seg_len, ts_seg_start = self.staff.scheduler.enqueue_midi(
            file_path, self.q_playback, self.q_gui, similarity
        )

        self.ts_queue += ts_seg_len
        # Avoid adding duplicates if path comes from seeker vs augment
        base_file_path = os.path.basename(file_path)
        if (
            not self.staff.seeker.played_files
            or os.path.basename(self.staff.seeker.played_files[-1]) != base_file_path
        ):
            self.staff.seeker.played_files.append(file_path)

        self.n_files_queued += 1
        self.s_transition_times.emit(self.staff.scheduler.ts_transitions)
        start_time = self.td_start + timedelta(seconds=ts_seg_start)

        # Ensure playlist path is valid before writing
        if hasattr(self, "pf_playlist") and self.pf_playlist:
            write_log(
                self.pf_playlist,
                self.n_files_queued,
                start_time.strftime("%y-%m-%d %H:%M:%S"),
                file_path,
                f"{similarity:.4f}" if similarity is not None else "----",
            )
        else:
            console.log(
                f"{self.tag} [yellow]Playlist path not set, skipping log write."
            )

    def _extract_player_segment(self) -> Optional[str]:
        """
        Extracts the last N ticks of recorded player MIDI into a temporary file.

        Returns
        -------
        Optional[str]
            Path to the temporary MIDI file, or None if insufficient notes.
        """
        if not self.staff.midi_recorder or not self.staff.midi_recorder.recorded_notes:
            return None

        segment_notes = []
        accumulated_ticks = 0
        # Iterate backwards through recorded notes
        for msg in reversed(self.staff.midi_recorder.recorded_notes):
            if msg.time > 0:
                accumulated_ticks += msg.time
            segment_notes.append(msg)
            if accumulated_ticks >= self.player_segment_tick_interval:
                break  # Collected enough ticks

        # Check if enough ticks were actually collected
        if accumulated_ticks < self.player_segment_tick_interval:
            if self.args.verbose:
                console.log(
                    f"{self.tag} Insufficient recorded notes for {self.player_segment_beat_interval}-beat segment ({accumulated_ticks}/{self.player_segment_tick_interval} ticks)."
                )
            return None

        # Reverse to chronological order
        segment_notes.reverse()

        # Create MIDI file
        midi_segment = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        track = mido.MidiTrack()
        midi_segment.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=self.tempo, time=0))

        # Add notes, adjusting time of the first note
        if segment_notes:
            first_note_time = segment_notes[0].time  # Store original delta time
            track.append(segment_notes[0].copy(time=0))  # First note starts at time 0
            # Add subsequent notes with their original delta times
            for i in range(1, len(segment_notes)):
                track.append(segment_notes[i])

        # Save to temporary file
        temp_filename = f"player_segment_{uuid.uuid4()}.mid"
        pf_temp_player_segment = os.path.join(
            self.p_log, temp_filename
        )  # Save in log dir
        try:
            midi_segment.save(pf_temp_player_segment)
            if self.args.verbose:
                console.log(
                    f"{self.tag} Saved temporary player segment: {pf_temp_player_segment}"
                )
            return pf_temp_player_segment
        except Exception as e:
            console.log(f"{self.tag} [red]Failed to save temporary MIDI segment: {e}")
            return None

    def _check_player_embedding(self):
        """
        Extracts player segment, gets embedding, checks diff, and potentially adjusts.
        """
        pf_temp_player_segment = self._extract_player_segment()
        if pf_temp_player_segment is None:
            return  # Not enough notes or error saving

        try:
            current_player_embedding = panther.send_embedding(
                pf_temp_player_segment,
                model=self.params.seeker.metric,
                mode=self.params.seeker.mode,
            )
        except Exception as e:
            console.log(f"{self.tag} [red]Error getting player embedding: {e}")
            os.remove(pf_temp_player_segment)  # Clean up temp file
            return
        finally:
            # Ensure temp file is deleted even if panther fails
            if os.path.exists(pf_temp_player_segment):
                os.remove(pf_temp_player_segment)

        if self.previous_player_embedding is not None:
            embedding_diff = current_player_embedding - self.previous_player_embedding
            diff_magnitude = np.linalg.norm(embedding_diff)
            if self.args.verbose:
                console.log(
                    f"{self.tag} Player embedding diff magnitude: {diff_magnitude:.4f}"
                )

            if diff_magnitude > self.player_embedding_diff_threshold:
                console.log(
                    f"{self.tag} [yellow]Embedding diff threshold exceeded ({diff_magnitude:.4f} > {self.player_embedding_diff_threshold}). Adjusting trajectory..."
                )
                try:
                    self._adjust_playback_trajectory(
                        embedding_diff, current_player_embedding
                    )
                except Exception as e:
                    console.print_exception(show_locals=True)
                    console.log(
                        f"{self.tag} [red]Error adjusting playback trajectory: {e}"
                    )
                    # Reset cutoff just in case it was set before error
                    self.playback_cutoff_tick = float("inf")
            else:
                if self.args.verbose:
                    console.log(
                        f"{self.tag} Player embedding difference below threshold."
                    )
        else:
            if self.args.verbose:
                console.log(
                    f"{self.tag} First player embedding calculated, storing for next check."
                )

        self.previous_player_embedding = current_player_embedding

    def _adjust_playback_trajectory(
        self, embedding_diff: np.ndarray, current_player_embedding: np.ndarray
    ):
        """
        Adjusts the upcoming playback based on the embedding difference.
        """
        # 1. Calculate Target Time & Tick
        target_dt = datetime.now() + timedelta(seconds=self.adjustment_lookahead_s)
        target_elapsed_s = (target_dt - self.td_start).total_seconds()
        target_tick = mido.second2tick(target_elapsed_s, TICKS_PER_BEAT, self.tempo)
        if self.args.verbose:
            console.log(
                f"{self.tag} Adjustment target time: {target_elapsed_s:.2f}s ({target_tick} ticks)"
            )

        # 2. Set Cutoff Tick for Player
        self.playback_cutoff_tick = target_tick
        console.log(
            f"{self.tag} Set playback cutoff tick to {self.playback_cutoff_tick}"
        )

        # --- Need to lock or safely modify scheduler state ---
        # This section needs careful handling if scheduler runs in parallel.
        # Assuming RunWorker directly manages scheduler state for now.
        # TODO: Add locking if Scheduler becomes multi-threaded.

        # 3. Wipe Scheduler Future
        try:
            # Find the index of the first transition *after* the target time
            transition_times = self.staff.scheduler.ts_transitions
            cutoff_index = -1
            for i, t in enumerate(transition_times):
                if t >= target_elapsed_s:
                    cutoff_index = i
                    break

            if cutoff_index == -1:
                console.log(
                    f"{self.tag} [yellow]Target time {target_elapsed_s:.2f}s is beyond all scheduled transitions. Cannot adjust."
                )
                self.playback_cutoff_tick = float("inf")  # Reset cutoff
                return

            if cutoff_index == 0:
                console.log(
                    f"{self.tag} [yellow]Target time {target_elapsed_s:.2f}s is before the first transition. Cannot adjust past."
                )
                self.playback_cutoff_tick = float("inf")  # Reset cutoff
                return

            # Keep transitions and files *before* the cutoff index
            num_files_before = cutoff_index  # Index corresponds to number of files
            if self.args.verbose:
                console.log(
                    f"{self.tag} Wiping schedule after index {cutoff_index} (keeping {num_files_before} files)"
                )
            original_queued_files = len(self.staff.scheduler.queued_files)

            self.staff.scheduler.ts_transitions = transition_times[:cutoff_index]
            self.staff.scheduler.queued_files = self.staff.scheduler.queued_files[
                :num_files_before
            ]
            self.staff.scheduler.n_files_queued = num_files_before
            self.n_files_queued = num_files_before  # Keep RunWorker's count synced

            # Recalculate ts_queue (approximate)
            # TODO: More accurate recalculation? This assumes segments have roughly equal length.
            avg_seg_len = self.player_segment_beat_interval * 60.0 / self.params.bpm
            self.ts_queue = max(0, (num_files_before * avg_seg_len) - target_elapsed_s)
            console.log(
                f"{self.tag} Wiped {original_queued_files - num_files_before} segments from scheduler. New queue time approx {self.ts_queue:.2f}s."
            )

        except Exception as e:
            console.log(f"{self.tag} [red]Error wiping scheduler future: {e}")
            self.playback_cutoff_tick = float("inf")  # Reset cutoff on error
            return

        # --- End lock/safe modification section ---

        # 4. Get Future Segment Embedding
        if not self.staff.scheduler.queued_files:
            console.log(
                f"{self.tag} [yellow]No files left in scheduler queue after wipe. Cannot get future segment."
            )
            self.playback_cutoff_tick = float("inf")
            return

        future_segment_file = self.staff.scheduler.queued_files[
            -1
        ]  # Last file remaining
        pf_future_segment = os.path.join(
            self.staff.seeker.p_dataset, basename(future_segment_file)
        )
        if not os.path.exists(pf_future_segment):
            console.log(
                f"{self.tag} [red]Future segment file not found: {pf_future_segment}"
            )
            self.playback_cutoff_tick = float("inf")
            return
        try:
            future_segment_embedding = self.staff.seeker.get_embedding(
                pf_future_segment,
                model=self.params.seeker.metric,
            )
        except Exception as e:
            console.log(f"{self.tag} [red]Error getting future segment embedding: {e}")
            self.playback_cutoff_tick = float("inf")
            return

        # 5. Calculate Target Embedding
        target_embedding = future_segment_embedding + embedding_diff
        target_embedding /= np.linalg.norm(target_embedding, axis=1, keepdims=True)
        if self.args.verbose:
            console.log(
                f"{self.tag} Calculated target embedding shape: {target_embedding.shape}"
            )

        # 6. Find Best Match
        try:
            best_match_filename, similarity = self.staff.seeker.get_match(
                target_embedding
            )
            console.log(
                f"{self.tag} Found best match for adjusted target: '{best_match_filename}' (Similarity: {similarity:.4f})"
            )
        except Exception as e:
            console.log(
                f"{self.tag} [red]Error finding match for target embedding: {e}"
            )
            self.playback_cutoff_tick = float("inf")
            return

        # 7. Enqueue Match
        pf_match = os.path.join(self.staff.seeker.p_dataset, best_match_filename)
        try:
            self._queue_file(pf_match, similarity)
            console.log(
                f"{self.tag} Successfully queued adjusted segment: '{best_match_filename}'"
            )
        except Exception as e:
            console.log(f"{self.tag} [red]Error queueing the adjusted match file: {e}")
            # Note: Cutoff is not reset here, as the queue might be in an inconsistent state.
            # Manual intervention or further error handling might be needed.
            return  # Avoid resetting cutoff if queueing failed

        # 8. Reset Cutoff Tick AFTER successful queueing
        console.log(f"{self.tag} Resetting playback cutoff tick.")
        self.playback_cutoff_tick = float("inf")

    def augment_midi(
        self,
        pf_midi: str,
        seed_rearrange: Optional[bool] = None,
        seed_remove: Optional[float] = None,
    ) -> list[str]:
        # generate augmentations
        console.log(f"{self.tag} augmenting '{basename(pf_midi)}'")
        import pretty_midi

        # load from parameters if not provided
        if not seed_rearrange:
            seed_rearrange = self.params.seed_rearrange
        if not seed_remove:
            seed_remove = self.params.seed_remove

        pf_augmentations = os.path.join(self.p_log, "augmentations")
        if not os.path.exists(pf_augmentations):
            os.makedirs(pf_augmentations)

        midi_paths = []
        if seed_rearrange:
            split_beats = midi.beat_split(pf_midi, self.params.bpm)
            console.log(
                f"{self.tag}\t\tsplit '{basename(pf_midi)}' into {len(split_beats)} beats"
            )
            ids = list(range(len(split_beats)))
            rearrangements: list[list[int]] = [
                ids,  # original
                # ids[: len(ids) // 2] * 2,  # first half twice
                # ids[len(ids) // 2 + 1 :] * 2,  # second half twice
                ids[0:4] * 2,  # first four twice
                ids[-4:] * 2,  # last four twice
                [ids[-2], ids[-1]] * 4,  # last two beats
                [ids[-1]] * 8,  # last beat
            ]
            for i, arrangement in enumerate(rearrangements):
                console.log(f"{self.tag}\t\trearranging seed:\t{arrangement}")
                joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                    split_beats, arrangement, self.params.bpm
                )

                console.log(f"{self.tag}\t\tjoined midi:\t{joined_midi.get_end_time()}")

                pf_joined_midi = os.path.join(
                    pf_augmentations, f"{basename(pf_midi)}_a{i:02d}.mid"
                )
                joined_midi.write(pf_joined_midi)
                midi_paths.append(pf_joined_midi)
        else:
            midi_paths.append(pf_midi)

        if seed_remove:
            joined_paths = midi_paths
            midi_paths = []  # TODO: stop overloading this
            num_options = 0
            for mid in joined_paths:
                stripped_paths = midi.remove_notes(mid, pf_augmentations, seed_remove)
                console.log(
                    f"{self.tag}\t\tstripped {seed_remove * 100 if isinstance(seed_remove, float) else seed_remove}{'%' if isinstance(seed_remove, float) else ''} notes from '{basename(mid)}' (+{len(stripped_paths)} versions)"
                )
                midi_paths.append(stripped_paths)
                num_options += len(stripped_paths)
            console.log(
                f"{self.tag}\t\taugmented '{basename(pf_midi)}' into {num_options} files"
            )

            best_aug = ""
            best_path = []
            best_match = ""
            best_similarity = 0.0
            for ps in midi_paths:
                console.log(f"{self.tag}\tps: {ps}")
                for m in ps:
                    embedding = self.staff.seeker.get_embedding(m)
                    match, similarity = self.staff.seeker.get_match(embedding)
                    if similarity > best_similarity:
                        best_aug = m
                        best_match = match
                        best_similarity = similarity
                        best_path = ps  # [(p, similarity) if p == m else p for p in ps]
                    console.log(
                        f"{self.tag}\t\tbest match for '{basename(m)}' is '{basename(match)}' with similarity {similarity}"
                    )
        else:
            best_aug = ""
            best_match = ""
            best_similarity = 0.0
            for ps in midi_paths:
                for m in ps:
                    console.log(f"{self.tag}\t\t{basename(m)}")

            # find best match for each augmentation
            for mid in midi_paths:
                embedding = self.staff.seeker.get_embedding(mid, model="clap")
                match, similarity = self.staff.seeker.get_match(embedding)
                if similarity > best_similarity:
                    best_aug = mid
                    best_match = match
                    best_similarity = similarity
                console.log(
                    f"{self.tag}\t\tbest match for '{basename(mid)}' is '{basename(match)}' with similarity {similarity}"
                )

        console.log(
            f"{self.tag}\tbest augmentation is '{basename(best_aug)}' with similarity {best_similarity} matches to {best_match}"
        )

        if seed_remove:
            # add em all up
            midi_paths = [
                *best_path[: best_path.index(best_aug) + 1],
                os.path.join(self.args.dataset_path, best_match + ".mid"),
            ]

        return midi_paths
