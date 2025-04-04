import os
import csv
import yaml
from queue import Queue
from threading import Event
from omegaconf import OmegaConf
from PySide6 import QtWidgets, QtCore
from datetime import datetime

import workers
from workers import Staff
from utils import console
from widgets.runner import RunWorker
from widgets.param_editor import ParameterEditorWidget
from widgets.piano_roll import PianoRollWidget

from typing import Optional


class MainWindow(QtWidgets.QMainWindow):
    tag = "[white]main[/white]  :"
    workers: Staff
    midi_stop_event: Event
    run_thread: Optional[RunWorker] = None

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.params.bpm = self.args.bpm
        self.td_system_start = datetime.now()
        self.recording_offset = 0  # seconds

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
        self.status.showMessage("parameter editor loaded")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(800, 600)
        self.resize(int(geometry.width() * 0.5), int(geometry.height() * 0.5))
        # self.setFixedSize(int(geometry.width() * 0.8), int(geometry.height() * 0.8))

    def _build_timer(self):
        # Create velocity label (left side)
        self.velocity_label = QtWidgets.QLabel("Velocity: <b>0.0</b>")
        self.velocity_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.velocity_label.setMinimumWidth(120)
        font = self.velocity_label.font()
        font.setPointSize(font.pointSize() + 1)
        self.velocity_label.setFont(font)
        self.toolbar.addWidget(self.velocity_label)

        # Add spacer between velocity and time
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.toolbar.addWidget(spacer)

        # Create time label (right side)
        self.time_label = QtWidgets.QLabel()
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.time_label.setMinimumWidth(100)
        self.toolbar.addWidget(self.time_label)

        # Timer to update the time and velocity
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(100)  # update every 100ms for smoother velocity updates

        # Initial time update
        self._update_time()

    def _update_time(self):
        """
        update the time display in the toolbar.
        """
        current_time = QtCore.QTime.currentTime()
        time_text = current_time.toString("hh:mm:ss")
        delta = datetime.now() - self.td_system_start
        delta_text = f"{delta.seconds//3600:02d}:{(delta.seconds//60)%60:02d}:{delta.seconds%60:02d}"
        self.time_label.setText(time_text + " | " + delta_text)

        # Update velocity display if workers are initialized
        if hasattr(self, "workers") and hasattr(self.workers, "midi_recorder"):
            avg_vel = self.workers.midi_recorder.avg_velocity

            # Determine color based on velocity
            if avg_vel < 40:
                color = "light blue"
            elif avg_vel < 80:
                color = "orange"
            else:
                color = "red"

            self.velocity_label.setText(
                f"Velocity: <b><font color='{color}'>{avg_vel:.1f}</font></b>"
            )

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
        self.write_log(
            self.pf_playlist, "position", "start time", "file path", "similarity"
        )
        console.log(f"{self.tag} filesystem set up complete")

    def init_workers(self):
        """
        initialize the workers.
        """
        scheduler = workers.Scheduler(
            self.params.scheduler,
            self.params.bpm,
            self.p_log,
            self.p_playlist,
            self.td_system_start,
            self.params.n_transitions,
            self.params.initialization == "recording",
        )
        seeker = workers.Seeker(
            self.params.seeker,
            self.args.tables,
            self.args.dataset_path,
            self.p_playlist,
            self.params.bpm,
        )
        player = workers.Player(
            self.params.player, self.params.bpm, self.td_system_start
        )
        midi_recorder = workers.MidiRecorder(
            self.params.recorder,
            self.params.bpm,
            self.pf_player_query,
        )
        audio_recorder = workers.AudioRecorder(
            self.params.audio, self.params.bpm, self.p_log
        )
        self.workers = Staff(seeker, player, scheduler, midi_recorder, audio_recorder)

    def switch_to_piano_roll(self, q_gui: Queue):
        console.log(f"{self.tag} switching to piano roll view")
        self.piano_roll = PianoRollWidget(q_gui, self)
        self.setCentralWidget(self.piano_roll)
        if hasattr(self, "run_thread") and self.run_thread is not None:
            self.run_thread.s_transition_times.connect(
                self.piano_roll.pr_view.update_transitions
            )
        self.status.showMessage("piano roll view activated")

    def save_and_start(self, params):
        """
        save parameters to yaml file and start the application
        """
        ts_start = self.td_system_start.strftime("%y%m%d-%H%M%S")

        # filesystem setup
        # create session output directories
        self.p_log = os.path.join(
            self.args.output,
            f"{ts_start}_{self.params.seeker.metric}_{self.params.initialization}_{self.params.seeker.seed}",
        )
        os.makedirs(self.p_log, exist_ok=True)

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

        # Start the main processing in a QThread
        self.run_thread = RunWorker(self)
        self.run_thread.s_start_time.connect(self.update_start_time)
        self.run_thread.s_status.connect(self.status.showMessage)
        self.run_thread.start()

    def update_start_time(self, start_time):
        """
        update the system start time based on the signal from run thread.
        """
        self.td_start = start_time
        if hasattr(self, "piano_roll"):
            self.piano_roll.td_start = start_time

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

        # Stop the run thread if it exists
        if self.run_thread is not None and self.run_thread.isRunning():
            self.run_thread.requestInterruption()
            self.run_thread.wait(1000)  # Wait up to 1 second for thread to finish

        # Stop MIDI recording
        if hasattr(self, "midi_stop_event") and self.midi_stop_event is not None:
            self.workers.midi_recorder.stop_recording()
            if self.p_log:
                pf_player_accompaniment = os.path.join(
                    self.p_log, "player-accompaniment.mid"
                )
                self.workers.midi_recorder.save_midi(pf_player_accompaniment)

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
