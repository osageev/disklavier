import os
import math
import mido
import time
from PySide6 import QtCore
from threading import Thread
from rich.table import Table
from queue import PriorityQueue
from datetime import datetime, timedelta

import workers
from workers import Staff
from utils import basename, console, midi, write_log
from utils.midi import TICKS_PER_BEAT, MidiAugmentationConfig


class RunWorker(QtCore.QThread):
    """
    worker thread that handles the main application run loop.
    """

    tag = "[#008700]runner[/#008700]:"

    # session tracking
    tt_queue_end_tick = 0
    n_files_queued = 0
    pf_augmentations = None
    playing_file = None
    ts_system_start = datetime.now()
    td_playback_start = datetime.now()

    # signals
    s_status = QtCore.Signal(str)
    s_start_time = QtCore.Signal(datetime)
    s_switch_to_pr = QtCore.Signal(object)
    s_transition_times = QtCore.Signal(list)
    s_segments_remaining = QtCore.Signal(int)
    s_augmentation_started = QtCore.Signal(int)
    s_embedding_processed = QtCore.Signal()

    # queues
    q_playback = PriorityQueue()
    q_gui = PriorityQueue()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.args = main_window.args
        self.params = main_window.params
        self.staff: Staff = main_window.workers
        self.td_system_start = main_window.td_system_start

        # connect signals
        self.s_switch_to_pr.connect(self.main_window.switch_to_piano_roll)
        self.staff.seeker.s_embedding_calculated.connect(self.s_embedding_processed)

        # file paths from main window
        self.p_log = main_window.p_log
        self.p_playlist = main_window.p_playlist
        self.pf_master_recording = main_window.pf_master_recording
        self.pf_system_recording = main_window.pf_system_recording
        self.pf_player_query = main_window.pf_player_query
        self.pf_player_accompaniment = main_window.pf_player_accompaniment
        self.pf_schedule = main_window.pf_schedule
        self.pf_playlist = main_window.pf_playlist

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
                aug_config = MidiAugmentationConfig(
                    rearrange=self.params.augmentation.rearrange,
                    remove_percentage=self.params.augmentation.remove,
                    target_notes_remaining=self.params.augmentation.target_notes_remaining,
                    notes_removed_per_step=self.params.augmentation.notes_removed_per_step,
                    num_variations_per_step=self.params.augmentation.num_variations_per_step,
                    num_plays_per_segment_version=self.params.augmentation.num_plays_per_segment_version,
                    total_segments_for_sequence=self.params.augmentation.total_segments_for_sequence,
                )

                if (
                    aug_config.rearrange or aug_config.remove_percentage is not None
                ):  # Check if any augmentation is enabled
                    self.pf_augmentations = self.augment_midi(self.pf_seed, aug_config)
                    console.log(
                        f"{self.tag} playing {len(self.pf_augmentations)} augmentations:\n\t{self.pf_augmentations}"
                    )
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

        try:
            # Get current time and add startup delay
            current_time = datetime.now()
            # Round up to the next even second
            current_seconds = current_time.second + current_time.microsecond / 1000000
            seconds_to_next_even = math.ceil(current_seconds / 2) * 2 - current_seconds

            # Add the startup delay and the time to next even second
            self.td_playback_start = current_time + timedelta(
                seconds=self.params.startup_delay + seconds_to_next_even
            )
            console.log(
                f"{self.tag} start time set to {self.td_playback_start.strftime('%y-%m-%d %H:%M:%S')}"
            )
            self.s_start_time.emit(self.td_playback_start)
            self.staff.scheduler.set_start_time(self.td_playback_start)
            self.staff.scheduler.init_schedule(
                self.pf_schedule,
                0,  # ts_recording_len if self.params.initialization == "recording" else 0,
            )
            console.log(f"{self.tag} successfully initialized recording")

            # add seed to queue
            self._queue_file(self.pf_seed, None)

            # start metronome
            self.metronome.td_start = self.td_playback_start + timedelta(
                seconds=self.staff.scheduler.ts_transitions[0]
            )
            self.metronome.start()

            # start audio recording in a separate thread
            # TODO: fix ~1-beat delay in audio recording startup
            self.e_audio_stop = self.staff.audio_recorder.start_recording(
                self.td_playback_start
            )

            # switch to piano roll view
            self.s_switch_to_pr.emit(self.q_gui)
            # start player
            self.staff.player.set_start_time(self.td_playback_start)
            self.staff.player.td_last_note = self.td_playback_start
            self.thread_player = Thread(
                target=self.staff.player.play, name="player", args=(self.q_playback,)
            )
            self.thread_player.start()

            # start midi recording
            self.midi_stop_event = self.staff.midi_recorder.start_recording(
                self.td_playback_start
            )
            self.main_window.midi_stop_event = self.midi_stop_event
            # connect recorder to player for velocity updates
            self.staff.player.set_recorder(self.staff.midi_recorder)

            if self.pf_augmentations is not None:
                # get average velocity of next match
                next_avg_velocity = midi.get_average_velocity(self.pf_augmentations[-1])
                console.log(
                    f"{self.tag} first match average velocity: {next_avg_velocity}"
                )
                # scale velocity of next match
                midi.ramp_vel(
                    self.pf_augmentations[:-1], next_avg_velocity, self.args.bpm
                )
                for aug in self.pf_augmentations:
                    self._queue_file(aug, None)

            # play for set number of transitions
            # TODO: move this to be managed by scheduler and track scheduler state instead
            tempo = mido.bpm2tempo(self.params.bpm)
            while self.n_files_queued < self.params.n_transitions:
                self._emit_current_file()

                # check amount of time remaining in queue
                remaining_seconds = 0
                if not self.q_playback.empty():
                    tt_next_note = self.q_playback.queue[0][0]
                    remaining_ticks = max(0, self.tt_queue_end_tick - tt_next_note)
                    remaining_seconds = mido.tick2second(
                        remaining_ticks, TICKS_PER_BEAT, tempo
                    )
                else:
                    pass

                if remaining_seconds < self.params.startup_delay + 1:
                    pf_next_file, similarity = self.staff.seeker.get_next()
                    self._queue_file(pf_next_file, similarity)

                time.sleep(0.1)
                if not self.thread_player.is_alive():
                    console.log(f"{self.tag} player ran out of notes, exiting")
                    self.thread_player.join(0.1)
                    break

            # all necessary files queued, wait for playback to finish
            console.log(f"{self.tag} waiting for playback to finish...")
            while self.q_playback.qsize() > 0:
                self._emit_current_file()
                time.sleep(0.1)
            self.thread_player.join(timeout=0.1)
            console.log(f"{self.tag} playback complete")
            self.s_status.emit("playback complete")
            self.s_segments_remaining.emit(0)
        except KeyboardInterrupt:
            console.log(f"{self.tag}[yellow] CTRL + C detected, saving and exiting...")

        self.shutdown()

    def shutdown(self):
        console.log(f"{self.tag}[yellow] shutdown called, saving and exiting...")
        # dump queue to stop player
        while self.q_playback.qsize() > 0:
            try:
                _ = self.q_playback.get()
            except:
                if self.args.verbose:
                    console.log(
                        f"{self.tag} [yellow]ouch! tried to dump queue but failed"
                    )
                pass

        if hasattr(self, "thread_player"):
            self.thread_player.join(timeout=0.1)

        console.log(f"{self.tag} stopping metronome")
        self.metronome.stop()
        console.log(f"{self.tag} metronome stopped")

        # run complete, save and exit
        # close raw notes file
        self.staff.scheduler.raw_notes_file.close()

        # Stop audio recording if it's running
        if self.e_audio_stop is not None:
            console.log(f"{self.tag} stopping audio recording")
            self.staff.audio_recorder.stop_recording()

        # Stop MIDI passive recording if its running
        if self.midi_stop_event is not None:
            console.log(f"{self.tag} stopping MIDI recording")
            self.staff.midi_recorder.stop_recording()
            self.staff.midi_recorder.save_midi(self.pf_player_accompaniment)

        # convert queue to midi
        _ = self.staff.scheduler.queue_to_midi(self.pf_system_recording)
        _ = midi.combine_midi_files(
            [self.pf_system_recording, self.pf_player_accompaniment, self.pf_schedule],
            self.pf_master_recording,
        )

        # print playlist
        table = Table(title="PLAYLIST")
        with open(self.pf_playlist, mode="r") as file:
            headers = file.readline().strip().split(",")
            for header in headers:
                table.add_column(header)

            for line in file:
                row = line.strip().split(",")
                row[2] = os.path.basename(row[2])  # only print filenames
                table.add_row(*row)
        console.print(table)

        # plot piano roll if master recording exists
        if os.path.exists(self.pf_master_recording):
            console.log(f"{self.tag} generating piano roll visualization")
            midi.generate_piano_roll(self.pf_master_recording)

        console.save_text(os.path.join(self.p_log, f"{self.td_system_start}.log"))
        console.log(f"{self.tag}[green bold] shutdown complete, exiting")

    def _queue_file(self, file_path: str, similarity: float | None) -> None:
        ts_seg_len, ts_seg_start, tt_end_tick = self.staff.scheduler.enqueue_midi(
            file_path, self.q_playback, self.q_gui, similarity
        )

        self.tt_queue_end_tick = max(self.tt_queue_end_tick, tt_end_tick)
        self.staff.seeker.played_files.append(file_path)
        self.n_files_queued += 1
        self.s_transition_times.emit(self.staff.scheduler.ts_transitions)
        start_time = self.td_playback_start + timedelta(seconds=ts_seg_start)

        write_log(
            self.pf_playlist,
            self.n_files_queued,
            start_time.strftime("%y-%m-%d %H:%M:%S"),
            file_path,
            f"{similarity:.5f}" if similarity is not None else "----",
        )

        remaining_seconds_log = 0
        if not self.q_playback.empty():
            tt_next_note_log = self.q_playback.queue[0][0]
            tempo_log = mido.bpm2tempo(self.params.bpm)
            remaining_ticks_log = max(0, self.tt_queue_end_tick - tt_next_note_log)
            remaining_seconds_log = mido.tick2second(
                remaining_ticks_log, TICKS_PER_BEAT, tempo_log
            )

        console.log(
            f"{self.tag} queue time is now {remaining_seconds_log:.01f} seconds (end tick: {self.tt_queue_end_tick})"
        )

    def augment_midi(self, pf_midi: str, config: MidiAugmentationConfig) -> list[str]:
        """
        augment a midi file based on the provided configuration.

        note: the best match (if applicable through seeker) is also returned as the last element of the list.

        Parameters
        ----------
        pf_midi : str
            path to the midi file to augment
        config : MidiAugmentationConfig
            configuration object for augmentations.

        Returns
        -------
        list[str]
            list of paths to the augmented files, potentially including a best match from the dataset.
        """
        # generate augmentations
        console.log(
            f"{self.tag} augmenting '{basename(pf_midi)}' with config: {config}"
        )
        import pretty_midi

        pf_augmentations_dir = os.path.join(self.p_log, "augmentations")
        if not os.path.exists(pf_augmentations_dir):
            os.makedirs(pf_augmentations_dir)

        midi_paths = []
        if config.rearrange:
            split_beats = midi.beat_split(pf_midi, self.params.bpm)
            console.log(
                f"{self.tag}\t\tsplit '{basename(pf_midi)}' into {len(split_beats)} beats"
            )
            ids = list(range(len(split_beats)))
            # TODO: make rearrangements configurable via MidiAugmentationConfig
            rearrangements: list[list[int]] = [
                ids,  # original
                ids[0:4],  # first four
                ids[0:4] * 2,  # first four twice
                ids[-4:],  # last four
                ids[-4:] * 2,  # last four twice
                [ids[-2], ids[-1]] * 4,  # last two beats
                [ids[-1]] * 8,  # last beat
            ]
            for i, arrangement in enumerate(rearrangements):
                console.log(f"{self.tag}\t\trearranging seed:\t{arrangement}")
                # Ensure beat_join can handle potentially empty beats if `beats` dict is sparse
                if (
                    not all(
                        idx in split_beats
                        for idx in arrangement
                        if idx < len(split_beats)
                    )
                    and len(arrangement) > 0
                ):  # check if all arrangement ids are in split_beats
                    # This check might be too strict or needs refinement based on how beat_join handles missing beat indices
                    # For now, if an arrangement requests a beat index that doesn't exist (e.g. from remove_empty=True in beat_split),
                    # it might cause issues or empty results.
                    # A more robust beat_join or filtering of arrangements might be needed.
                    console.log(
                        f"{self.tag}\t\tarrangement {arrangement} contains beat indices not found in split_beats (count: {len(split_beats)}), skipping rearrangement."
                    )
                    continue

                joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                    split_beats, arrangement, self.params.bpm
                )

                if joined_midi.get_end_time() < 1.0:  # basic check for near-empty MIDI
                    console.log(
                        f"\t\tjoined midi is empty (or near it), skipping ({basename(pf_midi)}_a{i:02d}.mid)"
                    )
                    continue
                else:
                    console.log(
                        f"{self.tag}\t\tjoined midi:\t{joined_midi.get_end_time()} s"
                    )

                # add lead-in bar for rearranged files
                beat_len_sec = 60 / self.params.bpm
                if (
                    i > 0
                ):  # assuming the first arrangement (original) doesn't need lead-in
                    for instrument in joined_midi.instruments:
                        for note in instrument.notes:
                            note.start += beat_len_sec
                            note.end += beat_len_sec

                pf_joined_midi = os.path.join(
                    pf_augmentations_dir, f"{basename(pf_midi)}_rearranged_a{i:02d}.mid"
                )
                joined_midi.write(pf_joined_midi)
                midi_paths.append(pf_joined_midi)
        else:
            # if no rearrangement, the original path is the base for potential note removal
            midi_paths.append(pf_midi)

        # note removal can occur on original or rearranged versions
        # if config.seed_remove_percentage or config.target_notes_remaining is specified
        paths_after_removal = []
        if (
            config.remove_percentage is not None
            or config.target_notes_remaining is not None
        ):
            # use the paths generated so far (original or rearranged) as input for note removal
            input_paths_for_removal = list(midi_paths)  # operate on a copy
            midi_paths = (
                []
            )  # reset midi_paths, will be populated by remove_notes results

            for mid_path_for_removal in input_paths_for_removal:
                # midi.remove_notes now takes the config object
                stripped_paths = midi.remove_notes(
                    mid_path_for_removal, pf_augmentations_dir, config
                )
                console.log(
                    f"\t\tapplied note removal config to '{basename(mid_path_for_removal)}' -> {len(stripped_paths)} versions"
                )
                paths_after_removal.extend(stripped_paths)

            # if removal was done, paths_after_removal contains all generated versions
            # if no removal criteria were met in remove_notes, it might return empty or original
            if (
                paths_after_removal
            ):  # only update midi_paths if removal actually produced files
                midi_paths.extend(
                    paths_after_removal
                )  # use extend to add all elements from paths_after_removal
            elif (
                not input_paths_for_removal and not midi_paths
            ):  # if no rearrangement and no removal, ensure original is still there
                midi_paths.append(pf_midi)
            elif (
                input_paths_for_removal
                and not paths_after_removal
                and not config.rearrange
            ):
                # This case implies removal was attempted on the original file but yielded no new files (e.g., no notes to remove).
                # midi_paths would still contain the original pf_midi from the earlier append.
                pass
            elif (
                input_paths_for_removal and not paths_after_removal and config.rearrange
            ):
                # This implies rearrangement happened, but subsequent removal on those rearranged files yielded nothing.
                # midi_paths should retain the rearranged files from the rearrangement step.
                # input_paths_for_removal still holds them.
                midi_paths = input_paths_for_removal

            console.log(
                f"{self.tag}\t\taugmented '{basename(pf_midi)}' considering note removal, resulting in {len(midi_paths)} candidate files before seeker matching."
            )

        # if midi_paths is empty here it means:
        # 1. No rearrangement AND
        # 2. No removal criteria met (or removal resulted in no files) AND
        # 3. Original pf_midi was not re-added. This should be handled.
        if not midi_paths and os.path.exists(pf_midi):
            console.log(
                f"{self.tag} no augmentations applied or resulted in files, using original: {basename(pf_midi)}"
            )
            midi_paths = [pf_midi]
        elif not midi_paths and not os.path.exists(pf_midi):
            console.log(
                f"{self.tag} [red]error: pf_midi {pf_midi} does not exist and no augmentations could be made.[/red]"
            )
            return []

        best_aug = ""
        best_match_from_dataset = ""  # Renamed to clarify it's from the dataset
        best_similarity = 0.0

        # make sure midi_paths is not empty and contains valid file paths
        valid_midi_paths = [
            p for p in midi_paths if os.path.exists(p) and os.path.getsize(p) > 0
        ]
        if not valid_midi_paths:
            console.log(
                f"{self.tag} [yellow]no valid midi paths found after augmentation steps for {basename(pf_midi)}.[/yellow]"
            )
            # if original pf_midi exists, return it as a fallback
            if os.path.exists(pf_midi):
                return [pf_midi]
            return []

        # Emit signal for augmentation start with total number of embeddings to calculate
        total_embeddings_to_calculate = len(valid_midi_paths)
        self.s_augmentation_started.emit(total_embeddings_to_calculate)

        for m_path in valid_midi_paths:
            embedding = self.staff.seeker.get_embedding(m_path)
            if embedding is None or embedding.sum() == 0:  # check for empty embedding
                console.log(
                    f"\t\t[orange italic]{basename(m_path)} has no notes or embedding failed, skipping[/orange italic]"
                )
                continue

            # get_match returns (matched_filename_stem, similarity_score)
            match_stem, similarity = self.staff.seeker.get_match(embedding)
            if similarity > best_similarity:
                best_aug = m_path  # this is the path to the best augmented version (local file)
                best_match_from_dataset = match_stem  # this is the filename stem of the best match in the dataset
                best_similarity = similarity
            console.log(
                f"\t\tmatch for '{basename(m_path)}' is '{basename(match_stem if match_stem else 'None')}' with similarity {similarity:.4f}"
            )

        console.log(
            f"{self.tag}\tbest augmentation is '{basename(best_aug)}' (sim: {best_similarity:.4f} to dataset file '{best_match_from_dataset}')"
        )

        final_paths_to_play = []

        if (
            config.remove_percentage is not None
            or config.target_notes_remaining is not None
        ):
            # if note removal was part of the process
            if best_aug:
                best_aug_basename = basename(best_aug)
                parts = best_aug_basename.split("_")

                key_prefix = ""
                if "_rearranged_a" in best_aug_basename and "_v" in best_aug_basename:
                    key_prefix = "_".join(
                        parts[:-1]
                    )  # everything before the last _rXX part
                elif "_v" in best_aug_basename:
                    key_prefix = "_".join(parts[:-1])
                else:
                    key_prefix = best_aug_basename.split(".")[0]

                for (
                    m_path
                ) in valid_midi_paths:  # iterate through all valid generated paths
                    if key_prefix in basename(m_path):
                        # This logic is a bit broad. If best_aug was 'file_v01_r05.mid',
                        # key_prefix is 'file_v01_r'. This will match 'file_v01_r02.mid', 'file_v01_r05.mid' etc.
                        # This seems to align with "play the augmentation which is missing X notes, Y notes"
                        final_paths_to_play.append(m_path)
                final_paths_to_play.sort()

                # if somehow the prefix logic failed but we have a best_aug
                if not final_paths_to_play and best_aug:
                    final_paths_to_play.append(best_aug)
            else:
                console.log(
                    "[yellow]warning: note removal active, but no 'best_aug' identified. returning original file or first valid path.[/yellow]"
                )
                # Fallback: if original pf_midi is valid, use it. Otherwise, first valid generated path.
                if os.path.exists(pf_midi) and pf_midi in valid_midi_paths:
                    final_paths_to_play.append(pf_midi)
                elif valid_midi_paths:
                    final_paths_to_play.append(valid_midi_paths[0])

        elif config.rearrange:  # only rearrangement, no removal
            if best_aug:  # best_aug would be one of the rearranged files
                final_paths_to_play.append(best_aug)  # Play the best rearranged one
            elif (
                valid_midi_paths
            ):  # no specific best (e.g. all zero sim), play the first valid rearranged
                final_paths_to_play.append(valid_midi_paths[0])
            elif os.path.exists(
                pf_midi
            ):  # if rearrangement produced nothing valid, fallback to original
                final_paths_to_play.append(pf_midi)

        else:  # no rearrangement, no removal attempted via config
            # best_aug would be the original pf_midi if it was processed by seeker
            if os.path.exists(pf_midi):
                final_paths_to_play.append(pf_midi)

        # ensure no duplicates and preserve order for initial simple case
        seen = set()
        unique_final_paths = []
        for p in final_paths_to_play:
            if p not in seen:
                unique_final_paths.append(p)
                seen.add(p)
        final_paths_to_play = unique_final_paths

        # Apply total_segments_for_sequence if specified
        num_segments_target = config.total_segments_for_sequence

        if (
            num_segments_target is not None
            and isinstance(num_segments_target, int)
            and num_segments_target > 0
            and len(final_paths_to_play) > num_segments_target
        ):

            N_current = len(final_paths_to_play)
            K_target = num_segments_target

            console.log(
                f"{self.tag} attempting to select {K_target} segments from {N_current} available augmentations for '{basename(pf_midi)}': {[basename(p) for p in final_paths_to_play]}"
            )

            selected_paths_subset = []
            if K_target == 1:
                # If only one segment is desired, pick the first from the current list
                if N_current > 0:
                    selected_paths_subset = [final_paths_to_play[0]]
            elif (
                K_target > 1
            ):  # K_target < N_current is guaranteed by the outer if, and N_current >= K_target
                # Build a new list with K_target elements, ensuring the first and last of the original
                # final_paths_to_play are included, with other elements spaced evenly.

                temp_indices = []
                for i in range(K_target):
                    # Calculate the index in the original list of N_current items
                    # The i-th item (0 to K_target-1) maps to an index from 0 to N_current-1
                    idx = round(i * (N_current - 1) / (K_target - 1))
                    temp_indices.append(int(idx))

                # Remove duplicates from indices (can happen if K_target is close to N_current due to rounding)
                # and maintain order.
                unique_selected_indices = []
                seen_indices_set = set()
                for idx_val in temp_indices:
                    if idx_val not in seen_indices_set:
                        unique_selected_indices.append(idx_val)
                        seen_indices_set.add(idx_val)

                # Create the subset of paths using the unique, sorted indices
                if unique_selected_indices:
                    selected_paths_subset = [
                        final_paths_to_play[i] for i in unique_selected_indices
                    ]

            if selected_paths_subset:
                final_paths_to_play = selected_paths_subset
                console.log(
                    f"{self.tag} selected {len(final_paths_to_play)} segments: {[basename(p) for p in final_paths_to_play]}"
                )
            elif N_current > 0:
                console.log(
                    f"{self.tag} [yellow]warning: failed to select subset (K_target={K_target}, N_current={N_current}), using original {len(final_paths_to_play)} segments for '{basename(pf_midi)}'.[/yellow]"
                )

        # Add the best matching original file from the dataset to cue the system for the next segment
        if best_match_from_dataset:
            pf_best_match = os.path.join(
                self.args.dataset_path, best_match_from_dataset + ".mid"
            )
            if os.path.exists(pf_best_match):
                if (
                    not final_paths_to_play or final_paths_to_play[-1] != pf_best_match
                ):  # avoid adding if it's already the last one
                    final_paths_to_play.append(pf_best_match)
            else:
                console.log(
                    f"[yellow]warning: best match from dataset '{pf_best_match}' not found.[/yellow]"
                )

        if not final_paths_to_play and os.path.exists(pf_midi):
            console.log(
                f"{self.tag} no augmentation paths selected, returning original MIDI: {basename(pf_midi)}"
            )
            return [pf_midi]
        elif not final_paths_to_play:
            console.log(
                f"{self.tag} [red]error: no paths to return from augment_midi for {basename(pf_midi)}.[/red]"
            )
            return []

        console.log(
            f"{self.tag} final augmentation sequence for '{basename(pf_midi)}': {[basename(p) for p in final_paths_to_play]}"
        )
        return final_paths_to_play

    def _emit_current_file(self):
        current_status = self.staff.scheduler.get_current_file()
        if current_status is not None and self.playing_file != current_status[0]:
            self.playing_file = current_status[0]
            console.log(f"{self.tag} now playing '{self.playing_file}'")
            self.s_status.emit(f"now playing '{self.playing_file}'")
            num_files_remaining = self.params.n_transitions - current_status[1]
            self.s_segments_remaining.emit(num_files_remaining)
