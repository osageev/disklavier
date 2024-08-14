import time
import mido
import heapq
import threading
from pathlib import Path

from midi_loader import PreloadedMidiFile


class PlaybackScheduler:
    event_queue = []
    playing = False
    playback_thread = None
    scheduler_thread = None
    queue_lock = threading.Lock()
    global_start_time = 0

    def __init__(self, tempo: int, midi_output_name: str):
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

    def schedule_events(self, midi_file: PreloadedMidiFile, start_ticks: int):
        """Schedules MIDI events from a midi_file to the event queue."""
        self.ticks_per_beat = mido.MidiFile(midi_file.file_name).ticks_per_beat
        with self.queue_lock:
            print(
                f"scheduling {Path(midi_file.file_name).stem}\tat {mido.tick2second(start_ticks, self.ticks_per_beat, self.tempo):07.03f}s ({start_ticks} ticks)"
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
            print(
                f"sleeping for {delay:04.02f} seconds ({mido.tick2second(event_ticks, self.ticks_per_beat, self.tempo):.02f}s - {current_time:.02f}s)"
            )

            if delay > 0:
                time.sleep(delay)
            self.midi_output.send(event.msg)
