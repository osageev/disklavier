import threading
import time
import heapq
import mido


class PlaybackScheduler:
    def __init__(self, tempo, midi_output_name):
        self.tempo = tempo  # Tempo in beats per minute (BPM)
        self.event_queue = (
            []
        )  # Priority queue for MIDI events (min-heap based on event time)
        self.playing = False
        self.playback_thread = None
        self.scheduler_thread = None
        self.queue_lock = threading.Lock()
        self.global_start_time = 0
        self.midi_output = mido.open_output(midi_output_name)  # type: ignore

    def start_playback(self):
        self.playing = True
        self.global_start_time = time.time()  # Initialize global time
        self.playback_thread = threading.Thread(target=self._playback_loop, name="player")
        self.playback_thread.start()

    def stop_playback(self):
        self.playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
        self.midi_output.close()

    def schedule_events(self, midi_file, start_time):
        """Schedules MIDI events from a midi_file to the event queue."""
        with self.queue_lock:
            for event in midi_file.events:
                event_time = start_time + event.global_time
                heapq.heappush(self.event_queue, (event_time, event))

                # Check if we need to adjust the timing of subsequent events
                self._adjust_event_timing(event_time)

    def _adjust_event_timing(self, inserted_time: float):
        """Adjusts the timing of subsequent events to ensure correct playback."""
        # This function re-orders the queue if necessary to handle overlaps
        temp_queue = []
        while self.event_queue:
            event_time, event = heapq.heappop(self.event_queue)
            print(f"comp {event_time} and {inserted_time}")
            if event_time >= inserted_time:
                # Adjust timing for all events after the inserted event
                event_time += inserted_time - event_time
            heapq.heappush(temp_queue, (event_time, event))
        self.event_queue = temp_queue

    def _playback_loop(self):
        while self.playing:
            with self.queue_lock:
                if not self.event_queue:
                    continue
                event_time, event = heapq.heappop(self.event_queue)

            current_time = time.time() - self.global_start_time
            delay = event_time - current_time

            if delay > 0:
                time.sleep(delay)
            self._send_midi_event(event.event_data)

    def _send_midi_event(self, event_data):
        try:
            self.midi_output.send(event_data)
            # print(f"sent MIDI event: {event_data}")
        except Exception as e:
            print(f"Error sending MIDI event: {e}")

    def _beat_duration(self):
        return 60 / self.tempo
