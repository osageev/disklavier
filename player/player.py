import os
import mido
from mido import MidiFile, Message
from threading import Thread, Event
from queue import Queue
import time
import simpleaudio as sa

from utils import console, tick


class Player:
    p = "[blue]player[/blue]:"
    playing_file_path = ""
    playing_file = ""
    next_file_path = ""
    volume = 1.0  # volume scaling factor
    last_volume = 1.0  # memory

    def __init__(
        self,
        params,
        kill_event: Event,
        playback_event: Event,
        play_event: Event,
        filename_queue: Queue,
        command_queue: Queue,
    ) -> None:
        self.params = params
        self.killed = kill_event
        self.get_next = playback_event
        self.play = play_event
        self.file_queue = filename_queue
        self.commands = command_queue
        # self.out_port = mido.open_output(self.params.out_port)  # type: ignore

        # multiple player override
        if self.params.p:
            self.p = self.params.p

    def play_loop(self):
        self.get_next.set()

        while not self.killed.is_set():
            # get next file from queue
            console.log(f"{self.p} waiting for file")
            self.playing_file_path, similarity = self.file_queue.get()
            self.playing_file = os.path.basename(self.playing_file_path)
            self.get_next.set()

            # print progress
            file_tempo = int(os.path.basename(self.playing_file).split("-")[1])
            found_tempo = -1
            for track in MidiFile(self.playing_file_path).tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        found_tempo = msg.tempo
            console.log(
                f"{self.p} playing '{self.playing_file}' ({file_tempo}BPM --> {round(mido.tempo2bpm(found_tempo)):01d}BPM) sim = {similarity:.03f}"
            )

            # play file
            self.play_midi(self.playing_file_path)

        console.log(f"{self.p}[orange] shutting down")

    def play_midi(self, midi_path: str) -> None:
        midi = MidiFile(midi_path)

        # open file and wait for play event
        with mido.open_output(self.params.out_port) as outport:  # type: ignore
            console.log(f"{self.p} waiting to play")
            while not self.play.is_set():
                time.sleep(0.00001)

            self.play.clear()
            console.log(f"{self.p} playing")

            # play file
            last_beat = time.time()
            for msg in midi.play(meta_messages=True):
                if not msg.is_meta:
                    outport.send(msg)
                else:
                    if msg.type == "set_tempo":  # type: ignore
                        console.log(f"{self.p} playing at {mido.tempo2bpm(msg.tempo)} BPM", msg)  # type: ignore
                    if msg.type == "text":  # type: ignore
                        beat = time.time()
                        console.log(f"{self.p} {msg.text} [grey50]({beat - last_beat:.05f}s)")  # type: ignore
                        last_beat = beat
                        tick(self.params.tick, self.p, False)

                # end active notes and return if killed
                if self.killed.is_set():
                    with mido.open_output(self.out_port) as outport:  # type: ignore
                        for note in range(128):
                            msg = Message("note_off", note=note, velocity=0, channel=0)
                            outport.send(msg)
                    break

    # def playback_loop(self, seed_file_path: str, fh):
    #     """"""
    #     self.playing_file_path = seed_file_path
    #     (self.next_file_path, similarity) = self.file_queue.get()
    #     next_file = os.path.basename(self.next_file_path)
    #     first_loop = True

    #     while self.next_file_path is not None and not self.killed.is_set():
    #         self.playing_file = os.path.basename(self.playing_file_path)
    #         file_tempo = int(os.path.basename(self.playing_file).split("-")[1])
    #         found_tempo = -1
    #         for track in MidiFile(self.playing_file_path).tracks:
    #             for msg in track:
    #                 if msg.type == "set_tempo":
    #                     found_tempo = msg.tempo

    #         if not first_loop:  # bad code!
    #             self.next_file_path, similarity = self.file_queue.get()
    #             next_file = os.path.basename(self.next_file_path)
    #         self.get_next.set()

    #         console.log(
    #             f"{self.p} playing '{self.playing_file}' ({file_tempo}) -- {similarity:.3f} --> '{next_file}' ({round(mido.tempo2bpm(found_tempo)):01d})"
    #         )

    #         self.play_midi_file(self.playing_file_path)
    #         self.playing_file_path = self.next_file_path

    #         first_loop = False

    #     console.log(f"{self.p} [orange]shutting down")

    # def play_midi_file(self, midi_path: str) -> None:
    #     """"""
    #     midi = MidiFile(midi_path)
    #     found_tempo = -1
    #     runtime = 0
    #     printed_msg = False
    #     do_fade = False

    #     for track in midi.tracks:
    #         for msg in track:
    #             if msg.type == "set_tempo":
    #                 found_tempo = msg.tempo

    #     if self.params.do_tick:
    #         stop_tick = Event()
    #         tick_thread = Thread(
    #             target=tick,
    #             args=(int(mido.tempo2bpm(found_tempo)), stop_tick, self.p, False),
    #         )
    #     for msg in midi.play():
    #         if hasattr(msg, "time"):
    #             runtime += msg.time  # type: ignore

    #         # check for keypresses
    #         while not self.commands.qsize() == 0:
    #             try:
    #                 command = self.commands.get()
    #                 console.log(f"{self.p} got key command '{command}'")
    #                 match command:
    #                     case "FADE":
    #                         self.last_volume = self.volume
    #                         do_fade = not do_fade
    #                     case "MUTE":
    #                         if self.volume == 0.0:
    #                             self.volume = self.last_volume
    #                             self.last_volume = 0.0
    #                         else:
    #                             self.last_volume = self.volume
    #                             self.volume = 0.0
    #                     case "VOL DOWN":
    #                         self.last_volume = self.volume
    #                         self.volume = max(0, self.volume - 0.1)
    #                     case "VOL UP":
    #                         self.last_volume = self.volume
    #                         self.volume = min(2.0, self.volume + 0.1)
    #                     case _:
    #                         pass

    #                 console.log(
    #                     f"{self.p} volume change {self.last_volume:.02f} -> {self.volume:.02f}"
    #                 )

    #                 self.commands.task_done()
    #             except:
    #                 console.log(f"{self.p} [bold orange]whoops[/bold orange]")

    #         if self.params.do_tick and not tick_thread.is_alive():
    #             tick_thread.start()

    #         # handle volume changes
    #         scaled_msg = msg
    #         if hasattr(msg, "velocity"):
    #             if do_fade:
    #                 self.last_volume = self.volume
    #                 # self.volume = self.fade(self.volume, runtime, midi.length)

    #             scaled_msg.velocity = max(0, min(127, round(scaled_msg.velocity * self.volume)))  # type: ignore

    #         self.out_port.send(scaled_msg)

    #         self.playback_progress.put(msg.time)  # type: ignore

    #         if self.killed.is_set() and not printed_msg:
    #             self.volume = 1.0
    #             self.last_volume = 1.0
    #             with mido.open_output(self.params.out_port) as outport:  # type: ignore
    #                 for note in range(128):
    #                     msg = Message("note_off", note=note, velocity=0, channel=0)
    #                     outport.send(msg)
    #             break
    #             # console.log(f"{self.p} [yellow]finishing playback of the current file")
    #             # do_fade = True
    #             # printed_msg = True

    #     if self.params.do_tick:
    #         stop_tick.set()
    #         tick_thread.join()

    #     # do_fade = False

    # def fade(self, current_value, current_time, end_time):
    #     """
    #     Scales the current value to zero by the end time.

    #     Parameters:
    #     - current_value: The current value to scale.
    #     - current_time: The current time in ticks
    #     - end_time: The end time by which the value should reach 0.

    #     Returns:
    #     - The scaled value, which linearly decreases to 0 by the end time. If the current time is past the end time, it returns 0.
    #     """

    #     scaling_factor = 1 - current_time / (end_time * 2.0)
    #     scaled_value = current_value * scaling_factor

    #     return scaled_value
