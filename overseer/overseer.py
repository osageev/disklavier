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
        self.playback_begun = Event()
        self.kill_event = Event()
        self.player_queue = Queue()

        self.seeker = Seeker(self.data_dir, self.output_dir, self.params.seeker)
        self.seeker.build_metrics()
        self.seeker.build_similarity_table()

        self.listener = Listener(self.params.listener, self.record_dir, self.recording_ready_event, self.kill_event)
        self.player = Player(self.params.player, self.record_dir, self.playback_begun, self.player_queue)

        
    def start(self):
        listen_thread = Thread(target=self.listener.listen, args=(), name="listener")

        listen_thread.start()

        try:
            while True:
                # check for recordings
                if self.recording_ready_event.is_set():
                    console.log(f"{self.p}triggering playback from recording '{self.listener.outfile}'")

                    recorded_ph = self.seeker.midi_to_ph(os.path.join(self.record_dir, self.listener.outfile))
                    first_link = self.seeker.find_most_similar_vector(recorded_ph)

                    playback_thread = Thread(target=self.player.playback_loop, args=(recorded_ph,), name="player")
                    self.player_queue.put((os.path.join(self.data_dir, str(first_link[0])), float(first_link[1])))

                    playback_thread.start()
                    self.listener.outfile = ''
                    self.recording_ready_event.clear()

                # check for next file requests
                if self.playback_begun:
                    pass

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
        else:
            console.log(f"{self.p}[orange]unable to find MIDI device[/orange] '{self.params.in_port}' [orange]defaulting to[/orange]'{available_inputs[0]}'")
            self.input_port = mido.open_input(available_inputs[0]) # type: ignore
            self.params.player.in_port = mido.open_input(available_inputs[0]) # type: ignore
            self.params.listener.in_port = mido.open_input(available_inputs[0]) # type: ignore

        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port) # type: ignore
        else:
            console.log(f"{self.p}[orange]unable to find MIDI device[/orange] '{self.params.out_port}' [orange]defaulting to[/orange]'{available_outputs[0]}'")
            self.output_port = mido.open_output(available_outputs[0]) # type: ignore
            self.params.player.out_port = mido.open_input(available_outputs[0]) # type: ignore
            self.params.listener.out_port = mido.open_input(available_outputs[0]) # type: ignore


    def save_midi(self):
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.mid"
        console.log(f"{self.p}saving recording '{self.outfile}'")
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for msg in self.recorded_notes:
            track.append(msg)
        mid.save(os.path.join(self.params.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.params.record_dir, self.outfile)):
            console.log(f"{self.p}successfully saved recording '{self.outfile}'")
        else:
            console.log(f"{self.p}failed to save recording '{self.outfile}'")


    def _stretch_midi_file(self, new_duration_seconds):
        """"""
        console.log(f"{self.p}rescaling file from {self.active_midi_file.length:.02f}s to {new_duration_seconds:.02f}s ({new_duration_seconds / self.active_midi_file.length:.03f})")
        # Calculate stretch factor based on the original duration
        stretch_factor = new_duration_seconds / self.active_midi_file.length
        
        # Scale the time attribute of each message by the stretch factor
        for track in self.active_midi_file.tracks:
            for msg in track:
                msg.time = int(msg.time * stretch_factor)