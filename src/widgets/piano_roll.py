from datetime import datetime, timedelta
import mido
import time
from queue import Queue
from dataclasses import dataclass

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtCore import QThread, Signal, QTimer, QObject
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from utils import console
from utils.midi import TICKS_PER_BEAT

from typing import Optional


@dataclass
class Note:
    pitch: int
    velocity: int
    start_time: float
    end_time: Optional[float] = None
    similarity: float = 0.0
    is_active: bool = True
    is_playing: bool = False


class PianoRollBuilder(QThread):
    tag = "[#90EE90]pr bld[/#90EE90]:"
    note_on_signal = Signal(Note)
    note_off_signal = Signal(int)

    def __init__(self, midi_queue: Queue, bpm: int, td_start: datetime):
        super().__init__()
        self.queue = midi_queue
        self.bpm = bpm
        self.td_start = td_start
        self.running = True
        self.tempo = mido.bpm2tempo(bpm)

    def run(self):
        """
        process messages from the queue and emit signals for note events.
        """
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

        while self.running:
            if not self.queue.empty():
                tt_abs, sim, message = self.queue.get()
                ts_abs = mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)

                # calculate when to send the message
                note_start = self.td_start + timedelta(seconds=ts_abs)
                dt_sleep = note_start - datetime.now()

                # sleep until the correct time if needed
                if dt_sleep.total_seconds() > 0:
                    # console.log(
                    #     f"{self.tag} waiting {dt_sleep.total_seconds():.3f} s to send message: {message}"
                    # )
                    time.sleep(dt_sleep.total_seconds())

                # now emit the signal for the appropriate note event
                if message.type == "note_on" and message.velocity > 0:
                    note = Note(
                        message.note,
                        message.velocity,
                        note_start.timestamp(),
                        similarity=sim,
                    )
                    self.note_on_signal.emit(note)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    self.note_off_signal.emit(message.note)

            time.sleep(0.001)  # small sleep to prevent cpu hogging

    def stop(self):
        self.running = False
        self.wait()


class PianoRollView(QGraphicsView):
    """
    graphical view that displays a piano roll with scrolling notes.
    """

    # constants
    debug = True
    timestep_ms = 10  # ms
    roll_len_ms = 10000  # visible roll duration in ms # TODO: make this 10 seconds a global parameter
    max_note_dur_ms = 5000  # auto trim notes longer than this
    MIN_NOTE_HEIGHT = 6
    MAX_NOTE_HEIGHT = 20
    MIN_KEY_WIDTH = 30
    MAX_KEY_WIDTH = 80
    KEY_COLORS = {
        "white": QColor(230, 230, 230),
        "black": QColor(51, 51, 51),
        "playing": QColor(255, 128, 128),
        "grid": QColor(178, 178, 178, 128),
        "background": QColor(38, 38, 38),
        "transition": QColor(0, 200, 0, 128),
    }
    MIN_NOTE = 21  # A0
    MAX_NOTE = 108  # C8
    NOTE_RANGE = MAX_NOTE - MIN_NOTE + 1
    default_bpm = 60  # default tempo in BPM

    # variables
    window_height = 600
    window_width = 800
    key_width = 60
    note_height = 10
    white_key_width = 60
    black_key_width = 51  # 85% of white key width
    current_time = 0
    notes = []
    active_notes = {}
    playing_notes = [0] * NOTE_RANGE
    current_tempo = default_bpm
    tempo_scale = 1.0
    transition_times = []  # ms

    def __init__(self, parent=None):
        super().__init__(parent)
        if hasattr(parent, "td_start") and parent is not None:
            self.start_time = parent.td_start
            self.bpm = parent.bpm
        else:
            console.log(
                f"[orange bold] no start time found, using current time [/orange bold]"
            )
            self.start_time = datetime.now()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # appearance settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(self.KEY_COLORS["background"])

        # update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(self.timestep_ms)

        # calculate dimensions based on initial size
        self.resize(800, 1000)

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
        self.calculate_dimensions()
        self._scene.setSceneRect(0, 0, self.window_height, self.window_width)

    def calculate_dimensions(self):
        """
        calculate ui dimensions based on window size.
        """
        self.note_height = min(
            max(self.window_width / self.NOTE_RANGE, self.MIN_NOTE_HEIGHT),
            self.MAX_NOTE_HEIGHT,
        )

        self.key_width = min(
            max(self.window_height * 0.15, self.MIN_KEY_WIDTH), self.MAX_KEY_WIDTH
        )

        self.white_key_width = self.key_width
        self.black_key_width = self.key_width * 0.85

    def is_black_key(self, note: int) -> bool:
        n = note % 12
        return n in [1, 3, 6, 8, 10]

    def get_note_y(self, note: int) -> float:
        note = max(min(note, self.MAX_NOTE), self.MIN_NOTE)
        # position calculated from bottom of screen
        return (
            self.window_width
            - (note - self.MIN_NOTE) * self.note_height
            - self.note_height
        )

    def time_to_x(self, time_ms: float) -> float:
        """
        convert a time in milliseconds to an x-coordinate on the screen.

        parameters
        ----------
        time_ms : float
            time in milliseconds.

        returns
        -------
        float
            x-coordinate on screen.
        """
        # adjust time based on tempo scale and recording offset
        adjusted_time_ms = (time_ms - self.current_time) * self.tempo_scale
        return self.key_width + (
            (adjusted_time_ms + self.roll_len_ms) / self.roll_len_ms
        ) * (self.window_height - self.key_width)

    def is_note_at_keyboard(self, note: Note) -> bool:
        start_x = self.time_to_x(note.start_time)
        end_x = self.time_to_x(note.end_time if note.end_time else self.current_time)
        return start_x <= self.key_width and end_x >= self.key_width

    def update_playing_notes(self):
        # reset all playing notes
        self.playing_notes = [0] * self.NOTE_RANGE

        # update playing status for all notes
        for note in self.notes:
            if self.is_note_at_keyboard(note):
                note.is_playing = True

                # note index in the playing_notes array
                note_idx = note.pitch - self.MIN_NOTE
                if 0 <= note_idx < self.NOTE_RANGE:
                    self.playing_notes[note_idx] = 1
            else:
                note.is_playing = False

    def update_time(self):
        self.current_time += self.timestep_ms

        # check for stuck notes and end them if they've been active too long
        for note_num, note in list(self.active_notes.items()):
            if self.current_time - note.start_time > self.max_note_dur_ms:
                self.note_off(int(note_num))

        # cleanup old notes that are too far in the past
        cutoff_time = self.current_time - self.roll_len_ms * 2
        self.notes = [n for n in self.notes if n.is_active or n.end_time > cutoff_time]

        # update playing notes
        self.update_playing_notes()

        # redraw
        self._scene.update()

    def note_on(self, note: Note):
        # check if note already active and end it first
        if note.pitch in self.active_notes:
            if self.debug:
                print(
                    f"Warning: Note {note.pitch} already active, ending previous note"
                )
            self.note_off(note.pitch)

        note.start_time = self.current_time
        self.notes.append(note)
        self.active_notes[note.pitch] = note

    def note_off(self, pitch: int):
        if pitch in self.active_notes:
            self.active_notes[pitch].end_time = self.current_time
            self.active_notes[pitch].is_active = False

            del self.active_notes[pitch]
        elif self.debug:
            print(f"warning: received noteOff for inactive note: {pitch}")

    def set_tempo(self, bpm):
        if bpm > 0:
            if self.debug:
                print(f"Tempo changed to: {bpm} BPM")
            self.current_tempo = bpm
            # calculate tempo scale factor: 60bpm = 1.0, 120bpm = 2.0, etc.
            self.tempo_scale = self.current_tempo / self.default_bpm
        elif self.debug:
            print(f"Invalid tempo value: {bpm}")

    def cleanup(self):
        active_note_nums = list(self.active_notes.keys())
        if self.debug:
            print(f"Cleaning up {len(active_note_nums)} active notes")

        for note_num in active_note_nums:
            self.note_off(int(note_num))

    def update_transitions(self, transitions: list[float]):
        self.transition_times = [
            t * 1000 for t in transitions
        ]  # convert to milliseconds

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        self.draw_grid(painter)
        self.draw_transition_lines(painter)  # draw transition lines before notes
        self.draw_notes(painter)
        self.draw_keyboard(painter)

    def draw_transition_lines(self, painter):
        """
        draw vertical green lines at transition points.
        """
        painter.setPen(QPen(self.KEY_COLORS["transition"], 2))
        for transition_time in self.transition_times:
            x = self.time_to_x(transition_time)
            if (
                self.key_width <= x <= self.window_height
            ):  # only draw if in visible area
                painter.drawLine(x, 0, x, self.window_width)

    def draw_keyboard(self, painter):
        # keyboard background
        painter.fillRect(0, 0, self.key_width, self.window_width, QColor(25, 25, 25))

        # white keys
        for i in range(self.MIN_NOTE, self.MAX_NOTE + 1):
            if not self.is_black_key(i):
                is_playing = self.playing_notes[i - self.MIN_NOTE] == 1
                color = (
                    self.KEY_COLORS["playing"]
                    if is_playing
                    else self.KEY_COLORS["white"]
                )
                y = self.get_note_y(i)

                painter.fillRect(0, y, self.white_key_width, self.note_height, color)

                # border
                painter.setPen(QPen(QColor(128, 128, 128), 0.5))
                painter.drawRect(0, y, self.white_key_width, self.note_height)

        # black keys
        for i in range(self.MIN_NOTE, self.MAX_NOTE + 1):
            if self.is_black_key(i):
                is_playing = self.playing_notes[i - self.MIN_NOTE] == 1
                color = (
                    self.KEY_COLORS["playing"]
                    if is_playing
                    else self.KEY_COLORS["black"]
                )
                y = self.get_note_y(i)

                painter.fillRect(0, y, self.black_key_width, self.note_height, color)

    def draw_grid(self, painter):
        # draw main background
        painter.fillRect(
            self.key_width,
            0,
            self.window_height - self.key_width,
            self.window_width,
            self.KEY_COLORS["background"],
        )

        # draw horizontal grid lines for each octave
        painter.setPen(QPen(self.KEY_COLORS["grid"], 1))
        for octave in range(9):
            note_number = 12 * octave + 12  # C notes
            if self.MIN_NOTE <= note_number <= self.MAX_NOTE:
                y = self.get_note_y(note_number)
                painter.drawLine(self.key_width, y, self.window_height, y)

        # draw additional grid lines for F notes
        painter.setPen(QPen(QColor(102, 102, 102, 76), 0.5))
        for octave in range(9):
            note_number = 12 * octave + 5  # F notes
            if self.MIN_NOTE <= note_number <= self.MAX_NOTE:
                y = self.get_note_y(note_number)
                painter.drawLine(self.key_width, y, self.window_height, y)

        # draw vertical time markers every second
        for t in range(0, self.roll_len_ms + 1, 1000):
            # adjust t based on tempo scale
            adjusted_t = t / self.tempo_scale
            x = self.time_to_x(self.current_time + adjusted_t)

            # only draw if in visible area
            if self.key_width <= x <= self.window_height:
                # draw time marker line
                painter.setPen(QPen(QColor(102, 102, 102, 128), 1))
                painter.drawLine(x, 0, x, self.window_width)

                # draw time labels
                if t % 2000 == 0:
                    painter.setPen(QColor(178, 178, 178))
                    font = QFont("Arial", 10)
                    painter.setFont(font)

                    if x < self.window_height - 15:
                        text_x = x - 5
                    else:
                        text_x = x - 15

                    relative_time_seconds = round(-adjusted_t / 1000)
                    time_label = (
                        "0"
                        if relative_time_seconds == 0
                        else str(relative_time_seconds)
                    )
                    painter.drawText(text_x, 10, time_label)

        # current time marker
        current_x = self.time_to_x(self.current_time)
        painter.setPen(QPen(QColor(255, 77, 77, 204), 2))
        painter.drawLine(current_x, 0, current_x, self.window_width)

    def draw_notes(self, painter):
        visible_start_time = self.current_time - self.roll_len_ms

        for note in self.notes:
            # skip notes that are completely before the visible time window
            if note.end_time and note.end_time < visible_start_time:
                continue

            # calculate start and end positions
            start_x = max(self.time_to_x(note.start_time), self.key_width)
            end_x = self.time_to_x(
                note.end_time if note.end_time else self.current_time
            )

            # skip notes that are completely after the visible time window
            if start_x > self.window_height:
                continue

            # adjust end_x if it's off-screen
            end_x = min(end_x, self.window_height)

            # calculate note box dimensions
            y = self.get_note_y(note.pitch)
            note_width = max(end_x - start_x, 2)  # ensure a minimum width

            # note background scaled by velocity
            alpha = 0.25 + (note.velocity / 127) * 0.75

            if note.is_active:
                color = QColor(77, 77, 77, int(alpha * 255))
            elif note.is_playing:
                color = QColor(255, 128, 128, int(alpha * 255))
            else:
                if note.similarity < 0.8:
                    color = QColor(255, 255, 0, int(alpha * 255))
                else:
                    # interpolate between yellow and dark gray based on similarity
                    # when similarity is 1.0, we get the original dark gray (77,77,77)
                    # when similarity is 0.8, we get yellow (255,255,0)
                    t = (note.similarity - 0.8) / 0.2  # normalize to 0-1 range
                    r = int(128 + (255 - 128) * (1 - t))
                    g = int(204 + (255 - 204) * (1 - t))
                    b = int(255 + (0 - 255) * (1 - t))
                    color = QColor(r, g, b, int(alpha * 255))

            painter.fillRect(start_x, y, note_width, self.note_height, color)

            # note border
            painter.setPen(QPen(QColor(51, 51, 51, 204), 0.5))
            painter.drawRect(start_x, y, note_width, self.note_height)


class PianoRollWidget(QWidget):
    tag = "[#90FF00]pr wgt[/#90FF00]:"

    def __init__(self, message_queue: Queue, parent=None):
        super().__init__(parent)
        self.message_queue = message_queue
        if parent is not None:
            self.td_start = parent.td_start
            self.bpm = parent.params.bpm
            console.log(
                f"{self.tag} using start time: {self.td_start}, bpm: {self.bpm}"
            )
        else:
            self.td_start = datetime.now()
            self.bpm = 60
            console.log(
                f"{self.tag}[orange bold] no start time found, using current time: {self.td_start} [/orange bold]"
            )

        # setup layout
        self.setMinimumSize(800, 600)

        # create piano roll view
        self.pr_view = PianoRollView(self)

        # connect transition signal if parent has it
        if parent is not None and hasattr(parent, "s_transition_times"):
            console.log(f"{self.tag} connecting transition signal")
            parent.s_transition_times.connect(self.pr_view.update_transitions)
        else:
            console.log(f"{self.tag} [yellow]no transition signal found[/yellow]")

        # create worker thread
        self.pr_builder = PianoRollBuilder(message_queue, self.bpm, self.td_start)
        self.pr_builder.note_on_signal.connect(self.pr_view.note_on)
        self.pr_builder.note_off_signal.connect(self.pr_view.note_off)
        self.pr_builder.start()

    def update_start_time(self, start_time: datetime):
        """
        update the start time for both the widget and view.
        """
        self.td_start = start_time
        self.pr_view.start_time = start_time
        console.log(f"{self.tag} updated start time to {start_time}")

    def resizeEvent(self, event):
        self.pr_view.setGeometry(0, 0, self.width(), self.height())

    def closeEvent(self, event):
        self.pr_builder.stop()
        self.pr_view.cleanup()
        super().closeEvent(event)


###############################################################################
####################### NOT USED FOR LIVE SYSTEM ##############################
###############################################################################


class MidiListener(QObject):
    """
    Listens for MIDI input and places messages into a queue.
    """

    midi_message_received = Signal(object)

    def __init__(self, midi_port, queue):
        super().__init__()
        self.midi_port = midi_port
        self.queue = queue

    def run(self):
        with mido.open_input(self.midi_port) as port:  # type: ignore
            for message in port:
                self.queue.put(message)
                self.midi_message_received.emit(message)


class WorkerThread(QThread):
    """
    Worker thread to process MIDI messages.
    """

    midi_message = Signal(object)

    def __init__(self, midi_port, queue):
        super().__init__()
        self.midi_port = midi_port
        self.queue = queue

    def run(self):
        listener = MidiListener(self.midi_port, self.queue)
        listener.midi_message_received.connect(self.midi_message.emit)
        listener.run()
