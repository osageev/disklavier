import os
from datetime import datetime
import mido
from mido import MidiFile, MidiTrack
from queue import Queue
from threading import Thread, Event
import time

from player.player import Player
from listener.listener import Listener
from seeker.seeker import Seeker

from utils import console


class Overseer:
    p = '[yellow]ovrsee[/yellow]: '
    playing_file = ''
    
    def __init__(self, params, data_dir: str, output_dir: str, record_dir: str):
        console.log(f"{self.p}initializing")

        self.params = params
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.record_dir = record_dir

        self._init_midi() # make sure MIDI port is available first

        # set up events & queues
        self.recording_ready_event = Event()
        self.give_next_event = Event()
        self.kill_event = Event()
        self.player_queue = Queue()

        self.seeker = Seeker(self.data_dir, self.output_dir, self.params.seeker)
        self.seeker.build_metrics()
        self.seeker.build_similarity_table()

        self.listener = Listener(self.params.listener, self.record_dir, self.recording_ready_event, self.kill_event)
        self.player = Player(self.params.player, self.record_dir, self.give_next_event, self.player_queue)

        
    def start(self):
        if not self.input_port or not self.output_port:
            return
        
        listen_thread = Thread(target=self.listener.listen, args=(), name="listener")

        listen_thread.start()

        try:
            next_file_path = ''
            while True:
                # check for recordings
                if self.recording_ready_event.is_set():
                    first_path = os.path.join(self.record_dir, self.listener.outfile)

                    console.log(f"{self.p}triggering playback from recording '{first_path}'")

                    recorded_ph = self.seeker.midi_to_ph(first_path)
                    first_link = self.seeker.find_most_similar_vector(recorded_ph)
                    next_file_path = os.path.join(self.data_dir, str(first_link[0]))

                    playback_thread = Thread(target=self.player.playback_loop, args=(first_path, recorded_ph), name="player")
                    self.player_queue.put((next_file_path, float(first_link[1])))

                    playback_thread.start()
                    self.listener.outfile = ''
                    self.recording_ready_event.clear()

                    console.log(f"{self.p}finished kicking off playback")

                # check for next file requests
                if self.give_next_event.is_set():
                    # console.log(f"{self.p}player is playing {self.player.playing_file}\t(next up is {next_file_path})")
                    if self.player.playing_file.split('_')[0] != 'recording':
                        self.seeker.metrics[self.player.playing_file]['played'] = 1
                    next_file, similarity = self.seeker.get_most_similar_file(os.path.basename(next_file_path))
                    next_file_path = os.path.join(self.data_dir, str(next_file))
                    self.player_queue.put((next_file_path, similarity))

                    console.log(f"{self.p}added next file '{next_file}' to queue with similarity {similarity}")
                    self.give_next_event.clear()

        except KeyboardInterrupt:
            # stop on Ctrl+C
            console.log(f"{self.p}shutting down")
            self.kill_event.set()

        listen_thread.join()
        playback_thread.join()


    def _init_midi(self):
        console.log(f"{self.p}connecting to MIDI")
        available_inputs = mido.get_input_names() # type: ignore
        available_outputs = mido.get_output_names() # type: ignore

        console.log(f"{self.p}found input ports: {available_inputs}")
        console.log(f"{self.p}found output ports: {available_outputs}")

        if len(available_inputs) == 0 or len(available_outputs) == 0:
            console.log(f"{self.p}no MIDI device detected")

        if self.params.in_port in available_inputs:
            self.input_port = mido.open_input(self.params.in_port) # type: ignore
        elif len(available_inputs) > 0:
            console.log(f"{self.p}[orange]unable to find MIDI device[/orange] '{self.params.in_port}' [orange]defaulting to[/orange]'{available_inputs[0]}'")
            self.input_port = mido.open_input(available_inputs[0]) # type: ignore
            self.params.player.in_port = mido.open_input(available_inputs[0]) # type: ignore
            self.params.listener.in_port = mido.open_input(available_inputs[0]) # type: ignore
        else:
            console.log(f"{self.p}[orange]no MIDI input devices available")

        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port) # type: ignore
        elif len(available_outputs) > 0:
            console.log(f"{self.p}[orange]unable to find MIDI device[/orange] '{self.params.out_port}' [orange]defaulting to[/orange]'{available_outputs[0]}'")
            self.output_port = mido.open_output(available_outputs[0]) # type: ignore
            self.params.player.out_port = mido.open_input(available_outputs[0]) # type: ignore
            self.params.listener.out_port = mido.open_input(available_outputs[0]) # type: ignore
        else:
            console.log(f"{self.p}[orange]no MIDI output devices available")
