import os
import mido
from mido import MidiFile, Message
from threading import Thread, Event, Lock
from queue import Queue
import heapq
import time
import simpleaudio as sa

from utils import console, tick


class Player:
    P = "[blue]player[/blue]:"
    playing = False
    playing_file_path = ""
    playing_file = ""
    next_file_path = ""
    volume = 1.0  # volume scaling factor
    last_volume = 1.0  # memory

    def __init__(
        self,
        params,
        midi_port: str,
        lock: Lock,
        command_queue: Queue,
    ) -> None:
        self.params = params
        self.out_port = mido.open_output(midi_port)  # type: ignore
        self.queue_lock = lock
        self.commands = command_queue

        if self.params.P:
            self.P = self.params.P

    def start_playback(self):
        self.playing = True
        self.global_start_time = time.time()  # initialize global time
        self.playback_thread = Thread(target=self._playback_loop, name="player")
        self.playback_thread.start()

    def stop_playback(self):
        self.playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
        self.out_port.close()

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

    def play_loop(self):
        self.get_next.set()

        while not self.killed.is_set():
            # get next file from queue
            console.log(f"{self.p} waiting for file")
            while self.file_queue.empty():
                time.sleep(0.01)
                if self.killed.is_set():
                    console.log(f"{self.p}[bold orange1] shutting down")
                    return
            self.waiting.clear()
            self.playing_file_path, similarity, transformations = self.file_queue.get()
            self.playing_file = os.path.basename(self.playing_file_path)
            self.get_next.set()

            # console.log progress
            file_tempo = int(os.path.basename(self.playing_file).split("-")[1])
            found_tempo = -1
            for track in MidiFile(self.playing_file_path).tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        found_tempo = msg.tempo
            console.log(
                f"{self.p} loaded '{self.playing_file}' ({file_tempo}BPM -> {round(mido.tempo2bpm(found_tempo)):01d}BPM) sim = {similarity:.03f}"
            )

            # play file
            self.play_midi(self.playing_file_path)
            self.waiting.set()

        console.log(f"{self.p}[bold orange1] shutting down")

    def play_midi(self, midi_path: str) -> None:
        midi = MidiFile(midi_path)

        # open file and wait for play event
        with mido.open_output(self.params.out_port) as outport:  # type: ignore
            console.log(f"{self.p} waiting to play")
            while not self.play.is_set():
                time.sleep(0.00001)
                if self.killed.is_set():
                    console.log(f"{self.p}[bold orange1] shutting down")
                    return

            if not self.params.is_recording:
                self.play.clear()

            # play file
            last_beat = time.time()
            console.loged_speed = False
            for msg in midi.play(meta_messages=True):
                if not msg.is_meta:
                    if msg.type == "note_on":  # type: ignore
                        self.notes.put_nowait(msg)
                    outport.send(msg)
                else:
                    if msg.type == "set_tempo" and not console.loged_speed:  # type: ignore
                        console.log(f"{self.p} playing '{os.path.basename(midi_path)}' at {mido.tempo2bpm(msg.tempo):.01f} BPM")  # type: ignore
                        console.loged_speed = True
                    if msg.type == "text" and self.params.do_tick:  # type: ignore
                        beat = time.time()
                        console.log(f"{self.p} {msg.text} [grey50]({beat - last_beat:.05f}s)")  # type: ignore
                        last_beat = beat
                        tick(self.params.tick, self.p, False)

                # end active notes and return if killed
                if self.killed.is_set():
                    with mido.open_output(self.params.out_port) as outport:  # type: ignore
                        for note in range(128):
                            msg = Message("note_off", note=note, velocity=0, channel=0)
                            outport.send(msg)
                    return

        console.log(f"{self.p} finished playing '{os.path.basename(midi_path)}'s")
