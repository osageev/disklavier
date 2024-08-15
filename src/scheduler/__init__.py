import time
import mido
import heapq
import threading
from pathlib import Path

from loader import PreloadedMidiFile
from utils import console


class Scheduler:
    P = "[deep_pink]schdle[/deep_pink]:"
    event_queue = []
    scheduler_thread = None
    global_start_time = 0

    def __init__(self, tempo: int, queue_lock: threading.Lock):
        self.bpm = tempo
        self.tempo = mido.bpm2tempo(tempo)
        self.queue_lock = queue_lock
        self.midi_output = mido.open_output("Disklavier")  # type: ignore

    def schedule_events(self, midi_file: PreloadedMidiFile, start_ticks: int):
        """Schedules MIDI events from a midi_file to the event queue."""
        self.ticks_per_beat = mido.MidiFile(midi_file.file_name).ticks_per_beat
        with self.queue_lock:
            console.log(
                f"{self.P} scheduling {Path(midi_file.file_name).stem}\tat {mido.tick2second(start_ticks, self.ticks_per_beat, self.tempo):07.03f}s ({start_ticks} ticks)"
            )
            for event in midi_file.events:
                event_ticks = start_ticks + event.global_ticks
                heapq.heappush(self.event_queue, (event_ticks, event))

                # Check if we need to adjust the timing of subsequent events
                self._adjust_event_timing(event_ticks)

    def _adjust_event_timing(self, inserted_ticks: int):
        """Adjusts the timing of subsequent events to ensure correct playback."""
        temp_queue = []
        while self.event_queue:
            (event_ticks, event) = heapq.heappop(self.event_queue)
            if event_ticks >= inserted_ticks:
                # adjust timing for all events after the inserted event
                event_ticks += inserted_ticks - event_ticks
            heapq.heappush(temp_queue, (event_ticks, event))
        self.event_queue = temp_queue


class PlaybackScheduler:
    P = "[deep_pink]schdle[/deep_pink]:"
    event_queue = []
    playing = False
    scheduling = False
    playback_thread = None
    scheduler_thread = None
    queue_lock = threading.Lock()
    global_start_time = 0

    def __init__(self, params, tempo: int, midi_output_name: str):
        self.params = params
        self.bpm = tempo
        self.tempo = mido.bpm2tempo(tempo)
        self.midi_output = mido.open_output(midi_output_name)  # type: ignore

    def start_playback(self):
        self.playing = True
        self.global_start_time = time.time()  # initialize global time
        self.playback_thread = threading.Thread(
            target=self._playback_loop, name="player"
        )
        self.playback_thread.start()

    def stop_playback(self):
        self.playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
        self.midi_output.close()

    def start_scheduling(self, seed_file: PreloadedMidiFile):
        self.scheduling = True
        self.scheduler_thread = threading.Thread(
            target=self._schedule_loop, args=(self, seed_file), name="scheduler"
        )
        self.scheduler_thread.start()

    def stop_scheduling(self):
        self.scheduling = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join()

    def _adjust_event_timing(self, inserted_ticks: int):
        """Adjusts the timing of subsequent events to ensure correct playback."""
        temp_queue = []
        while self.event_queue:
            (event_ticks, event) = heapq.heappop(self.event_queue)
            if event_ticks >= inserted_ticks:
                # adjust timing for all events after the inserted event
                event_ticks += inserted_ticks - event_ticks
            heapq.heappush(temp_queue, (event_ticks, event))
        self.event_queue = temp_queue

    def _playback_loop(self):
        while self.playing:
            with self.queue_lock:
                if not self.event_queue:
                    continue
                (event_ticks, event) = heapq.heappop(self.event_queue)

            current_time = time.time() - self.global_start_time
            delay = (
                mido.tick2second(event_ticks, self.ticks_per_beat, self.tempo)
                - current_time
            )
            console.log(
                f"{self.P} sleeping for {delay:04.02f} seconds ({mido.tick2second(event_ticks, self.ticks_per_beat, self.tempo):.02f}s - {current_time:.02f}s)"
            )

            if delay > 0:
                time.sleep(delay)
            self.midi_output.send(event.msg)

    def _schedule_loop(self, seed_file: PreloadedMidiFile):
        start_time = 0.0
        time_buffer = 30.0  # 30 seconds buffer
        current_time_in_queue = 0.0

        # Add the first MIDI file to the queue
        midi_file_index = 0
        self._schedule_events(seed_file, 0)
        current_time_in_queue += self.params.duration_t - self.ticks_per_beat
        midi_file_index += 1

        while midi_file_index < len(midi_files):
            # Check if we need to add more files to maintain the 30 seconds buffer
            if current_time_in_queue < time_buffer:
                midi_file = midi_files[midi_file_index]
                self._schedule_events(midi_file, start_time + current_time_in_queue)
                current_time_in_queue += self.params.duration_t - self.ticks_per_beat
                midi_file_index += 1
            else:
                # Wait for a short time before checking again to avoid busy-waiting
                time.sleep(1)

    def _schedule_events(self, midi_file: PreloadedMidiFile, start_ticks: int):
        """Schedules MIDI events from a midi_file to the event queue."""
        self.ticks_per_beat = mido.MidiFile(midi_file.file_name).ticks_per_beat
        with self.queue_lock:
            console.log(
                f"{self.P} scheduling {Path(midi_file.file_name).stem}\tat {mido.tick2second(start_ticks, self.ticks_per_beat, self.tempo):07.03f}s ({start_ticks} ticks)"
            )
            for event in midi_file.events:
                event_ticks = start_ticks + event.global_ticks
                heapq.heappush(self.event_queue, (event_ticks, event))

                # Check if we need to adjust the timing of subsequent events
                self._adjust_event_timing(event_ticks)
