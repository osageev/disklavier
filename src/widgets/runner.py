import os
from PySide6 import QtCore
from utils import console
from datetime import datetime, timedelta
import csv
import time
from threading import Thread
from multiprocessing import Process
from queue import PriorityQueue

import workers
from utils.panther import send_embedding


class RunWorker(QtCore.QThread):
    """
    worker thread that handles the main application run loop.
    """

    status_update = QtCore.Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tag = main_window.tag
        self.args = main_window.args
        self.params = main_window.params
        self.workers = main_window.workers
        self.td_system_start = main_window.td_system_start

        # File paths from main window
        self.p_log = main_window.p_log
        self.p_playlist = main_window.p_playlist
        self.pf_master_recording = main_window.pf_master_recording
        self.pf_system_recording = main_window.pf_system_recording
        self.pf_player_query = main_window.pf_player_query
        self.pf_player_accompaniment = main_window.pf_player_accompaniment
        self.pf_schedule = main_window.pf_schedule
        self.pf_max = main_window.pf_max
        self.pf_playlist = main_window.pf_playlist

    def write_log(self, filename: str, *args):
        """
        write a row to a csv log file.

        parameters
        ----------
        filename : str
            path to the log file.
        *args : any
            data to write to the log.
        """
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(args)

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
                    ts_recording_len = self.workers.midi_recorder.run()
                self.pf_seed = self.pf_player_query
            case "audio":
                self.pf_player_query = self.pf_player_query.replace(".mid", ".wav")
                ts_recording_len = self.workers.audio_recorder.record_query(
                    self.pf_player_query
                )
                embedding = send_embedding(self.pf_player_query, model="clap")
                console.log(
                    f"{self.tag} got embedding {embedding.shape} from pantherino"
                )
                best_match, best_similarity = self.workers.seeker.get_match(embedding)
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
                    self.pf_seed = self.workers.seeker.get_random()
                    console.log(
                        f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'"
                    )
            case "random" | _:  # choose random file from library
                self.pf_seed = self.workers.seeker.get_random()
                console.log(f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'")

        if self.params.seeker.mode == "playlist":
            # TODO: implement playlist mode using generated csvs
            raise NotImplementedError("playlist mode not implemented")

        self.workers.seeker.played_files.append(self.pf_seed)

        q_playback = PriorityQueue()
        q_gui = PriorityQueue()
        td_start = datetime.now() + timedelta(seconds=self.params.startup_delay)
        ts_queue = (
            self.args.bpm * (self.params.n_beats_per_segment + 1) / 60
        )  # time in queue in seconds
        n_files = 1  # number of files played so far

        self.workers.scheduler.td_start = td_start

        # start audio recording in a separate thread
        self.e_audio_stop = self.workers.audio_recorder.start_recording(td_start)

        if self.workers.scheduler.init_schedule(
            self.pf_schedule,
            ts_recording_len if self.params.initialization == "recording" else 0,
        ):
            console.log(f"{self.tag} successfully initialized recording")
        else:
            console.log(f"{self.tag} [red]error initializing recording, exiting")
            if self.e_audio_stop is not None:
                self.workers.audio_recorder.stop_recording()
            raise FileExistsError("Couldn't initialize MIDI recording file")

        try:
            # add seed to queue
            ts_seg_len, ts_seg_start = self.workers.scheduler.enqueue_midi(
                self.pf_seed, q_playback, q_gui
            )
            ts_queue += ts_seg_len
            console.log(f"{self.tag} enqueued seed at {ts_seg_start}")
            console.log(
                f"{self.tag} sys start is {self.td_system_start.timestamp()}, reg start is {td_start.timestamp()}"
            )
            self.write_log(
                self.pf_playlist,
                n_files,
                datetime.fromtimestamp(ts_seg_start).strftime("%y-%m-%d %H:%M:%S"),
                self.pf_seed,
                "----",
            )

            # preload first match
            # TODO: check if this is necessary
            if (
                self.params.initialization == "recording"
                or self.params.seeker.mode == "graph"
            ):
                pf_next_file, similarity = self.workers.seeker.get_next()
                ts_seg_len, ts_seg_start = self.workers.scheduler.enqueue_midi(
                    pf_next_file, q_playback, q_gui
                )
                ts_queue += ts_seg_len
                n_files += 1
                self.write_log(
                    self.pf_playlist,
                    n_files,
                    datetime.fromtimestamp(
                        self.td_system_start.timestamp()
                        + timedelta(seconds=ts_seg_start).total_seconds()
                    ).strftime("%y-%m-%d %H:%M:%S"),
                    pf_next_file,
                    similarity,
                )

            # start player
            self.workers.player.td_start = td_start
            self.workers.player.td_last_note = td_start
            thread_player = Thread(
                target=self.workers.player.play, name="player", args=(q_playback,)
            )
            thread_player.start()

            # start midi recording
            self.midi_stop_event = self.workers.midi_recorder.start_recording(td_start)
            self.main_window.midi_stop_event = self.midi_stop_event
            # connect recorder to player for velocity updates
            self.workers.player.set_recorder(self.workers.midi_recorder)

            # start metronome
            metronome = workers.Metronome(
                self.params.metronome, self.args.bpm, td_start
            )
            process_metronome = Process(target=metronome.tick, name="metronome")
            process_metronome.start()

            # play for set number of transitions
            # TODO: move this to be managed by scheduler and track scheduler state instead
            current_file = ""
            while n_files < self.params.n_transitions:
                if current_file != self.workers.scheduler.get_current_file():
                    current_file = self.workers.scheduler.get_current_file()
                    console.log(f"{self.tag} now playing '{current_file}'")
                    self.status_update.emit(f"now playing '{current_file}'")

                if q_playback.qsize() < self.params.n_min_queue_length:
                    pf_next_file, similarity = self.workers.seeker.get_next()
                    ts_seg_len, ts_seg_start = self.workers.scheduler.enqueue_midi(
                        pf_next_file, q_playback, q_gui
                    )
                    ts_queue += ts_seg_len
                    console.log(f"{self.tag} queue time is now {ts_queue:.01f} seconds")
                    n_files += 1
                    self.write_log(
                        self.pf_playlist,
                        n_files,
                        datetime.fromtimestamp(
                            ts_seg_start - self.td_system_start.timestamp()
                        ).strftime("%y-%m-%d %H:%M:%S"),
                        pf_next_file,
                        similarity,
                    )

                time.sleep(0.1)
                ts_queue -= 0.1
                if not thread_player.is_alive():
                    console.log(f"{self.tag} player ran out of notes, exiting")
                    thread_player.join(0.1)
                    break

            metronome.stop()

            # all necessary files queued, wait for playback to finish
            console.log(f"{self.tag} waiting for playback to finish...")
            while q_playback.qsize() > 0:
                time.sleep(0.1)
            thread_player.join(timeout=0.1)
        except KeyboardInterrupt:
            console.log(f"{self.tag}[yellow] CTRL + C detected, saving and exiting...")
            # dump queue to stop player
            while q_playback.qsize() > 0:
                try:
                    _ = q_playback.get()
                except:
                    if self.args.verbose:
                        console.log(
                            f"{self.tag} [yellow]ouch! tried to dump queue but failed"
                        )
                    pass
            thread_player.join(timeout=0.1)
            console.log(f"{self.tag} exiting")
