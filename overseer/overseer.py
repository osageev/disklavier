import os
import mido
from queue import Queue
from threading import Thread, Event
from rich.progress import Progress

from player.player import Player
from listener.listener import Listener
from seeker.seeker import Seeker

from utils import console


class Overseer:
    p = '[yellow]ovrsee[/yellow]: '
    playing_file = ''
    
    def __init__(self, params, data_dir: str, output_dir: str, record_dir: str, force_rebuild:bool=False, do_kickstart: bool=False):
        console.log(f"{self.p}initializing")

        self.params = params
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.record_dir = record_dir

        self._init_midi() # make sure MIDI port is available first
        if len(os.listdir(self.data_dir)) < 10:
            console.log(f"{self.p}[red]less than 10 files in input folder. are you sure you didnt screw something up?")

        # set up events & queues
        self.recording_ready_event = Event()
        self.give_next_event = Event()
        self.kill_event = Event()
        self.player_queue = Queue()

        # initialize objects to be overseen
        self.seeker = Seeker(self.params.seeker, self.data_dir, self.output_dir, force_rebuild)
        self.seeker.build_metrics()
        self.seeker.build_similarity_table()
        self.listener = Listener(self.params.listener, self.record_dir, self.recording_ready_event, self.kill_event)
        self.player = Player(self.params.player, self.record_dir, self.give_next_event, self.player_queue)

        
    def start(self):
        """start the system"""
        if not self.input_port or not self.output_port:
            return
        
        listen_thread = Thread(target=self.listener.listen, args=(), name="listener")
        listen_thread.start()

        next_file_path = ''
        num_played = 0
        while True:
            # check for recordings
            if self.recording_ready_event.is_set():
                first_path = os.path.join(self.record_dir, self.listener.outfile)

                console.log(f"{self.p}triggering playback from recording '{first_path}'")

                recorded_ph = self.seeker.midi_to_ph(first_path)
                first_link = self.seeker.find_most_similar_vector(recorded_ph)
                next_file_path = os.path.join(self.data_dir, str(first_link[0]))
                # self._set_tempo(next_file_path)

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
                self.change_tempo(next_file_path)

                self.player_queue.put((next_file_path, similarity))

                console.log(f"{self.p}added next file '{next_file}' to queue with similarity {similarity:.03f}")
                
                self.give_next_event.clear()
                num_played += 1

        listen_thread.join()
        playback_thread.join()


    def _set_tempo(self, midi_file_path):
        midi_file = mido.MidiFile(midi_file_path)
        bpm = float(os.path.basename(midi_file_path).split('-')[1])
        console.log(f"{self.p}found bpm {bpm} for file {os.path.basename(midi_file_path)}")
    
        for track in midi_file.tracks:
            for i, msg in enumerate(track):
                if msg.type == "set_tempo":
                    console.log(f"{self.p}update bpm message found: {msg}")
                    del track[i]

        console.log(f"{self.p}updating bpm")
        midi_file.tracks[0].append(mido.Message(
            type = "set_tempo",
            tempo = mido.bpm2tempo(bpm),
            time = 0
        ))

        self._set_tempo(midi_file_path)

        midi_file.save(midi_file_path)


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
            console.log(f"{self.p}unable to find MIDI device '{self.params.in_port}' falling back on '{available_inputs[0]}'")
            self.input_port = mido.open_input(available_inputs[0]) # type: ignore
            self.params.player.in_port = available_inputs[0]
            self.params.listener.in_port = available_inputs[0]
        else:
            console.log(f"{self.p}no MIDI input devices available")

        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port) # type: ignore
        elif len(available_outputs) > 0:
            console.log(f"{self.p}unable to find MIDI device '{self.params.out_port}' falling back on '{available_outputs[0]}'")
            self.output_port = mido.open_output(available_outputs[0]) # type: ignore
            self.params.player.out_port = available_outputs[0]
            self.params.listener.out_port = available_outputs[0]
        else:
            console.log(f"{self.p}no MIDI output devices available")


    def change_tempo(self, midi_file_path):
        midi_file = mido.MidiFile(midi_file_path)
        new_tempo_bpm = int(os.path.basename(midi_file_path).split('-')[1])
        new_tempo = mido.bpm2tempo(new_tempo_bpm)  # Convert BPM to microseconds per beat
        console.log(f"{self.p}new tempo is {new_tempo} (from {new_tempo_bpm})")
        new_message = mido.MetaMessage('set_tempo', tempo=new_tempo, time=0)
        tempo_added = False
        
        for track in midi_file.tracks:
            # Remove existing set_tempo messages
            for msg in track:
                if msg.type == 'set_tempo':
                    console.log(f"{self.p}removing set tempo message {msg}")
                    track.remove(msg)
            
            # Add new set_tempo message to the first track
            if not tempo_added:
                console.log(f"{self.p}adding message {new_message}")
                track.insert(0, new_message)
                tempo_added = True
        
        # If no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
        if not tempo_added:
            new_track = mido.MidiTrack()
            console.log(f"{self.p}adding message to new track {new_message}")
            new_track.append(new_message)
            midi_file.tracks.append(new_track)
        
        # Save the modified MIDI file
        console.log(f"{self.p}saved modified MIDI file with new tempo {new_tempo_bpm} BPM to {midi_file_path}")