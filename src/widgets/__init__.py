import os
import yaml
from omegaconf import OmegaConf
from PySide6 import QtWidgets, QtCore, QtGui
from utils import console
from datetime import datetime, timedelta
import csv
import time
from threading import Thread, Event
from multiprocessing import Process
from queue import PriorityQueue
import workers
from utils import midi
from utils.panther import send_embedding
from dataclasses import dataclass
from widgets.param_editor import ParameterEditorWidget


@dataclass
class Staff:
    def __init__(
        self,
        seeker: workers.Seeker,
        player: workers.Player,
        scheduler: workers.Scheduler,
        midi_recorder: workers.MidiRecorder,
        audio_recorder: workers.AudioRecorder,
    ):
        self.seeker = seeker
        self.player = player
        self.scheduler = scheduler
        self.midi_recorder = midi_recorder
        self.audio_recorder = audio_recorder


class MainWindow(QtWidgets.QMainWindow):
    tag = "[white]main[/white]  :"
    workers: Staff
    midi_stop_event: Event

    def __init__(self, args, params):
        self.args = args
        self.params = params

        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("disklavier")
        # toolbar
        self.toolbar = QtWidgets.QToolBar("Time Toolbar")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self._build_timer()

        # start by editing parameters
        self.param_editor = ParameterEditorWidget(self.params, self)
        self.setCentralWidget(self.param_editor)

        # status bar
        self.status = self.statusBar()
        self.status.showMessage("Parameter editor loaded")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(800, 600)
        self.resize(int(geometry.width() * 0.5), int(geometry.height() * 0.5))
        # self.setFixedSize(int(geometry.width() * 0.8), int(geometry.height() * 0.8))

    def _build_timer(self):
        # Create time label
        self.time_label = QtWidgets.QLabel()
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.time_label.setMinimumWidth(100)

        # Add spacer to push time label to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.toolbar.addWidget(spacer)
        self.toolbar.addWidget(self.time_label)

        # Timer to update the time
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)  # update every second

        # Initial time update
        self._update_time()

    def _update_time(self):
        """
        update the time display in the toolbar.
        """
        current_time = QtCore.QTime.currentTime()
        time_text = current_time.toString("hh:mm:ss")
        self.time_label.setText(time_text)

    def init_fs(self):
        """
        initialize the filesystem.
        """
        # filesystem setup
        # create session output directories
        if not os.path.exists(self.p_log) and self.args.verbose:
            console.log(f"{self.tag} creating new logging folder at '{self.p_log}'")
        self.p_playlist = os.path.join(self.p_log, "playlist")
        os.makedirs(self.p_playlist, exist_ok=True)

        # specify recording files
        self.pf_master_recording = os.path.join(self.p_log, f"master-recording.mid")
        self.pf_system_recording = os.path.join(self.p_log, f"system-recording.mid")
        self.pf_player_query = os.path.join(self.p_log, f"player-query.mid")
        self.pf_player_accompaniment = os.path.join(
            self.p_log, f"player-accompaniment.mid"
        )
        self.pf_schedule = os.path.join(self.p_log, f"schedule.mid")
        self.pf_max = os.path.join(self.p_log, f"max.mid")

        # copy old recording if replaying
        if self.args.replay and self.params.initialization == "recording":
            import shutil

            shutil.copy2(self.params.kickstart_path, self.pf_player_query)
            console.log(
                f"{self.tag} moved old recording to current folder '{self.pf_player_query}'"
            )
            self.params.seeker.pf_recording = self.pf_player_query

        # initialize playlist file
        self.pf_playlist = os.path.join(
            self.p_log, f"playlist_{self.td_system_start.strftime('%y%m%d-%H%M%S')}.csv"
        )
        self.write_log(self.pf_playlist, "position", "start time", "file path", "similarity")
        console.log(f"{self.tag} filesystem set up complete")

    def init_workers(self):
        """
        initialize the workers.
        """
        self.status.showMessage("initializing scheduler")
        scheduler = workers.Scheduler(
            self.params.scheduler,
            self.args.bpm,
            self.p_log,
            self.p_playlist,
            self.td_system_start,
            self.params.n_transitions,
            self.params.initialization == "recording",
        )
        self.status.showMessage("initializing seeker")
        seeker = workers.Seeker(
            self.params.seeker,
            self.args.tables,
            self.args.dataset_path,
            self.p_playlist,
            self.args.bpm,
        )
        self.status.showMessage("initializing player")
        player = workers.Player(self.params.player, self.args.bpm, self.td_system_start)
        self.status.showMessage("initializing midi recorder")
        midi_recorder = workers.MidiRecorder(
            self.params.recorder,
            self.args.bpm,
            self.pf_player_query,
        )
        self.status.showMessage("initializing audio recorder")
        audio_recorder = workers.AudioRecorder(
            self.params.audio, self.args.bpm, self.p_log
        )
        self.workers = Staff(seeker, player, scheduler, midi_recorder, audio_recorder)
        self.status.showMessage("all workers initialized")

    def save_and_start(self, params):
        """
        save parameters to yaml file and start the application.

        parameters
        ----------
        params : dict
            parameters to save.
        """
        self.td_system_start = datetime.now()
        ts_start = self.td_system_start.strftime("%y%m%d-%H%M%S")

        # filesystem setup
        # create session output directories
        self.p_log = os.path.join(
            self.args.output,
            f"{ts_start}_{self.params.seeker.metric}_{self.params.initialization}_{self.params.seeker.seed}",
        )

        # Create parameters file in p_log
        param_file = os.path.join(self.p_log, "parameters.yaml")

        try:
            with open(param_file, "w") as f:
                yaml.dump(OmegaConf.to_container(params), f, default_flow_style=False)

            self.status.showMessage(f"Parameters saved to {param_file}")
            console.log(f"{self.tag} parameters saved to '{param_file}'")
        except Exception as e:
            self.status.showMessage(f"Error saving parameters: {str(e)}")
            console.log(f"{self.tag} error saving parameters: {str(e)}")

        self.status.showMessage("initializing filesystem")
        self.init_fs()
        self.status.showMessage("initializing workers")
        self.init_workers()
        self.status.showMessage("system initialization complete")

        # Start the main processing loop in a separate thread
        # self.processing_thread = Thread(
        #     target=self.run_processing_loop,
        #     args=(
        #         seeker,
        #         scheduler,
        #         q_playback,
        #         thread_player,
        #         pf_playlist_file,
        #         n_files,
        #         params,
        #         ts_queue,
        #         td_system_start,
        #     ),
        # )
        # self.processing_thread.daemon = True
        # self.processing_thread.start()

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

    def write_log(self, filename: str, *args):
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(args)

    def cleanup_workers(self):
        """
        clean up all worker processes and threads.
        """
        if not hasattr(self, "workers"):
            return

        console.log(f"{self.tag} cleaning up workers...")

        # Stop metronome
        # if "process_metronome" in self.workers:
        #     self.workers["process_metronome"].kill()
        #     self.workers["process_metronome"].join(timeout=0.5)

        # Close raw notes file
        # if "scheduler" in self.workers:
        #     self.workers["scheduler"].raw_notes_file.close()

        # Stop audio recording
        # if (
        #     "audio_stop_event" in self.workers
        #     and self.workers["audio_stop_event"] is not None
        # ):
        #     self.workers["audio_recorder"].stop_recording()

        # Stop MIDI recording
        # if (
        #     hasattr(self, "midi_stop_event")
        #     and self.midi_stop_event is not None
        # ):
        #     self.workers.midi_recorder.stop_recording()
        #     if self.p_log:
        #         pf_player_accompaniment = os.path.join(
        #             self.p_log, "player-accompaniment.mid"
        #         )
        #         self.workers.midi_recorder.save_midi(pf_player_accompaniment)

        # Convert queue to MIDI
        # if self.p_log:
        #     pf_system_recording = os.path.join(self.p_log, "system-recording.mid")
        #     pf_player_accompaniment = os.path.join(
        #         self.p_log, "player-accompaniment.mid"
        #     )
        #     pf_schedule = os.path.join(self.p_log, "schedule.mid")
        #     pf_master_recording = os.path.join(self.p_log, "master-recording.mid")

        #     self.workers.scheduler.queue_to_midi(pf_system_recording)
        #     midi.combine_midi_files(
        #         [pf_system_recording, pf_player_accompaniment, pf_schedule],
        #         pf_master_recording,
        #     )

        console.log(f"{self.tag} workers cleaned up")

    def closeEvent(self, event):
        """
        handle the window close event.

        parameters
        ----------
        event : QCloseEvent
            close event.
        """
        self.cleanup_workers()
        super().closeEvent(event)

    def resizeEvent(self, event):
        """
        handle resize events and update dimensions.

        parameters
        ----------
        event : QResizeEvent
            resize event.
        """
        super().resizeEvent(event)
        self.window_height = self.height()
        self.window_width = self.width()

