
import sys
import time
import math
import mido
from queue import Queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF, QTimer, QObject
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QFont

from .worker import Worker

@dataclass
class Note:
    """
    represents a note in the piano roll.

    parameters
    ----------
    pitch : int
        midi note number.
    velocity : int
        note velocity (0-127).
    start_time : float
        time when the note started (in ms).
    end_time : Optional[float]
        time when the note ended (in ms), or none if still active.
    """

    pitch: int
    velocity: int
    start_time: float
    end_time: Optional[float] = None
    similarity: float = 0.0
    is_active: bool = True
    is_playing: bool = False

class PianoRollBuilder(QThread):
    note_on_signal = Signal(int, int)
    note_off_signal = Signal(int)

    def __init__(self, midi_queue: Queue):
        super().__init__()
        self.queue = midi_queue
        self.running = True

    def run(self):
        """
        process messages from the queue and emit signals for note events.
        """
        while self.running:
            if not self.queue.empty():
                message = self.queue.get()
                if message.type == "note_on" and message.velocity > 0:
                    self.note_on_signal.emit(message.note, message.velocity)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    self.note_off_signal.emit(message.note)
            time.sleep(0.001)  # Small sleep to prevent CPU hogging

    def stop(self):
        self.running = False
        self.wait()

class PianoRollView(QGraphicsView):
    """
    graphical view that displays a piano roll with scrolling notes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # constants
        self.DEBUG = False
        self.TIMESTEP = 10  # ms
        self.ROLL_LEN_MS = 10000  # visible roll duration in ms
        self.MAX_NOTE_DUR_MS = 5000  # auto trim notes longer than this

        # graphical constants
        self.MIN_NOTE_HEIGHT = 6
        self.MAX_NOTE_HEIGHT = 20
        self.MIN_KEY_WIDTH = 30
        self.MAX_KEY_WIDTH = 80
        self.KEY_COLORS = {
            "white": QColor(230, 230, 230),
            "black": QColor(51, 51, 51),
            "playing": QColor(255, 128, 128),
            "grid": QColor(178, 178, 178, 128),
            "background": QColor(38, 38, 38),
        }
        self.MIN_NOTE = 21  # A0
        self.MAX_NOTE = 108  # C8
        self.NOTE_RANGE = self.MAX_NOTE - self.MIN_NOTE + 1
        self.DEFAULT_TEMPO = 60  # default tempo in BPM

        # variables
        self.window_height = 600
        self.window_width = 800
        self.key_width = 60
        self.note_height = 10
        self.white_key_width = 60
        self.black_key_width = 51  # 85% of white key width
        self.current_time = 0
        self.notes = []
        self.active_notes = {}
        self.playing_notes = [0] * self.NOTE_RANGE
        self.current_tempo = self.DEFAULT_TEMPO
        self.tempo_scale = 1.0

        # appearance settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(self.KEY_COLORS["background"])

        # update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(self.TIMESTEP)

        # calculate dimensions based on initial size
        self.resize(800, 600)

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
        self.scene.setSceneRect(0, 0, self.window_height, self.window_width)

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
        # adjust time based on tempo scale
        adjusted_time_ms = (time_ms - self.current_time) * self.tempo_scale
        return self.key_width + (
            (adjusted_time_ms + self.ROLL_LEN_MS) / self.ROLL_LEN_MS
        ) * (self.window_height - self.key_width)

    def is_note_at_keyboard(self, note: Note) -> bool:
        start_x = self.time_to_x(note.start_time)
        end_x = self.time_to_x(note.end_time if note.end_time else self.current_time)
        return start_x <= self.key_width and end_x >= self.key_width

    def update_playing_notes(self):
        """
        update the state of which notes are currently playing.
        """
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
        """
        update the current time and manage notes.
        """
        self.current_time += self.TIMESTEP

        # check for stuck notes and end them if they've been active too long
        for note_num, note in list(self.active_notes.items()):
            if self.current_time - note.start_time > self.MAX_NOTE_DUR_MS:
                self.note_off(int(note_num))

        # cleanup old notes that are too far in the past
        cutoff_time = self.current_time - self.ROLL_LEN_MS * 2
        self.notes = [n for n in self.notes if n.is_active or n.end_time > cutoff_time]

        # update playing notes
        self.update_playing_notes()

        # redraw
        self.scene.update()

    def note_on(self, note, velocity):
        """
        handle note on event.

        parameters
        ----------
        note : int
            midi note number.
        velocity : int
            note velocity (0-127).
        """
        # check if note already active and end it first
        if note in self.active_notes:
            if self.DEBUG:
                print(f"Warning: Note {note} already active, ending previous note")
            self.note_off(note)

        new_note = Note(note, velocity, self.current_time)
        self.notes.append(new_note)
        self.active_notes[note] = new_note

        if self.DEBUG:
            print(f"Note ON: {note} velocity: {velocity} time: {self.current_time}")

    def note_off(self, note):
        """
        handle note off event.

        parameters
        ----------
        note : int
            midi note number.
        """
        if note in self.active_notes:
            self.active_notes[note].end_time = self.current_time
            self.active_notes[note].is_active = False

            if self.DEBUG:
                duration = self.current_time - self.active_notes[note].start_time
                print(f"Note OFF: {note} duration: {duration} ms")

            del self.active_notes[note]
        elif self.DEBUG:
            print(f"Warning: Received noteOff for inactive note: {note}")

    def set_tempo(self, bpm):
        """
        set the tempo for playback.

        parameters
        ----------
        bpm : float
            tempo in beats per minute.
        """
        if bpm > 0:
            if self.DEBUG:
                print(f"Tempo changed to: {bpm} BPM")
            self.current_tempo = bpm
            # calculate tempo scale factor: 60bpm = 1.0, 120bpm = 2.0, etc.
            self.tempo_scale = self.current_tempo / self.DEFAULT_TEMPO
        elif self.DEBUG:
            print(f"Invalid tempo value: {bpm}")

    def cleanup(self):
        """
        clean up any active notes.
        """
        active_note_nums = list(self.active_notes.keys())
        if self.DEBUG:
            print(f"Cleaning up {len(active_note_nums)} active notes")

        for note_num in active_note_nums:
            self.note_off(int(note_num))

    def drawBackground(self, painter, rect):
        """
        draw the piano roll background.

        parameters
        ----------
        painter : QPainter
            painter object.
        rect : QRectF
            rectangle to paint in.
        """
        super().drawBackground(painter, rect)

        # draw grid
        self.draw_grid(painter)

        # draw notes
        self.draw_notes(painter)

        # draw keyboard
        self.draw_keyboard(painter)

    def draw_keyboard(self, painter):
        """
        draw the piano keyboard.

        parameters
        ----------
        painter : QPainter
            painter object.
        """
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
        """
        draw the grid for the piano roll.

        parameters
        ----------
        painter : QPainter
            painter object.
        """
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
        for t in range(0, self.ROLL_LEN_MS + 1, 1000):
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
        """
        draw the notes on the piano roll.

        parameters
        ----------
        painter : QPainter
            painter object.
        """
        visible_start_time = self.current_time - self.ROLL_LEN_MS

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
            alpha = 0.5 + (note.velocity / 127) * 0.5

            if note.is_active:
                color = QColor(77, 77, 77, int(alpha * 255))
            elif note.is_playing:
                color = QColor(255, 128, 128, int(alpha * 255))
            else:
                color = QColor(128, 204, 255, int(alpha * 255))

            painter.fillRect(start_x, y, note_width, self.note_height, color)

            # note border
            painter.setPen(QPen(QColor(51, 51, 51, 204), 0.5))
            painter.drawRect(start_x, y, note_width, self.note_height)


class PianoRollWidget(QWidget):
    """
    widget that contains the piano roll view and handles midi input.

    parameters
    ----------
    message_queue : Queue
        queue containing midi messages.
    parent : QWidget, optional
        parent widget.
    """

    def __init__(self, message_queue, parent=None):
        super().__init__(parent)
        self.message_queue = message_queue

        # setup layout
        self.setMinimumSize(800, 600)

        # create piano roll view
        self.piano_roll = PianoRollView(self)

        # create worker thread
        self.worker = PianoRollBuilder(message_queue)
        self.worker.note_on_signal.connect(self.piano_roll.note_on)
        self.worker.note_off_signal.connect(self.piano_roll.note_off)
        self.worker.start()

    def resizeEvent(self, event):
        """
        handle resize events.

        parameters
        ----------
        event : QResizeEvent
            resize event.
        """
        self.piano_roll.setGeometry(0, 0, self.width(), self.height())

    def closeEvent(self, event):
        """
        handle close events and clean up resources.

        parameters
        ----------
        event : QCloseEvent
            close event.
        """
        self.worker.stop()
        self.piano_roll.cleanup()
        super().closeEvent(event)


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
        with mido.open_input(self.midi_port) as port:
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