import os
import mido
from mido import MidiFile, MidiTrack, Message
from threading import Thread
from datetime import datetime
import time
from queue import LifoQueue
from utils import console


class Player():
    p = '[green]play[/green]  : '
    is_recording = False
    recorded_notes = []
    ctrl = 67  # soft pedal
    active_midi_file = None
    is_playing = False

    def __init__(self, params, record_dir: str) -> None:
        self.params = params
        self.record_dir = record_dir


    def start(self) -> None:
        """Starts monitoring MIDI input and recording when pedal is pressed."""
        console.log(f"{self.p}listening on port {self.input_port}")
        Thread(target=self.listener.listen(), name="listening").start()


    def _record_loop(self) -> None:
        """Monitors MIDI input for pedal and note events."""
        # timing
        start_time = 0
        end_time = 0
        last_note_time = start_time

        dtpb = 480

        with mido.open_input(self.input_port) as inport: # type: ignore
            console.log(f"{self.p}recording at {dtpb}tpb... Press Ctrl+C to stop.")
            try:
                for msg in inport:
                    # record delta time of input message 
                    # mido doesn't do this by default for some reason
                    current_time = time.time()
                    msg.time = int((current_time - last_note_time) * dtpb)
                    console.log(f"{self.p}\t{msg}")
                    last_note_time = current_time 

                    if msg.type == 'control_change' and msg.control == self.params.ctrl:
                        if msg.value == 0:
                            end_time = time.time()
                            console.log(f"{self.p}stopping recording at {end_time - start_time:.02f}s...")
                            self.is_recording = False

                            # start playback
                            self._stretch_midi_file(end_time - start_time)
                            self.save_midi()
                            threading.Thread(target=self.playback_loop(), name="playing").start()
                            console.log(f"{self.p}playback thread started")
                            self.active_midi_file = None
                        elif self.is_recording == False:
                            console.log(f"{self.p}recording...")
                            self.is_recording = True
                    elif self.is_recording and msg.type in ['note_on', 'note_off']:
                        if self.active_midi_file is None:
                            self._new_midi_obj()
                            start_time = time.time()
                            msg.time = 0
                        self.active_midi_file.tracks[0].append(msg)
            except KeyboardInterrupt:
                # Stop recording on Ctrl+C
                end_time = time.time()
                console.log(f"{self.p}stopping recording at {end_time}...")


    def playback_loop(self):
        """"""
        time_elapsed = 0.
        self.playing = True
        self.playing_file_path = self.last_recorded_file
        recorded_ph = self.seeker.midi_to_ph(self.last_recorded_file)
        next_file, seeker = self.seeker.find_most_similar_vector(recorded_ph)
        next_file_path = os.path.join(self.seeker.input_dir, next_file)
        first_loop = True

        while next_file is not None:
            self.playing_file = os.path.basename(self.playing_file_path)
            console.log(f"{self.p}{self.playing_file_path}\t{self.playing_file}")
            console.log(f"{self.p}{next_file_path}\t{next_file}")
            if not first_loop: # bad code!
                console.log(f"{self.p}getting next file for {self.playing_file_path}")
                self.seeker.metrics[self.playing_file]['played'] = 1
                next_file, seeker = self.seeker.get_most_similar_file(self.playing_file)
                next_file_path = os.path.join(self.seeker.input_dir, next_file)

            console.log(f"{self.p}[{int(time_elapsed):04d}]\tPlaying {self.playing_file}\t(next up is {next_file})\tsim={seeker:.3f}")
            time_elapsed += MidiFile(self.playing_file_path).length

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = next_file_path
            first_loop = False
        

    def play_midi_file(self, midi_path: str) -> None:
        midi_data = MidiFile(midi_path)
        with mido.open_output(self.params.out_port) as outport: # type: ignore
            t = 0.
            for msg in midi_data.play():
                self._printProgressBar(t, midi_data.length, suffix='s')
                t += msg.time # type: ignore
                if not msg.is_meta:
                    outport.send(msg)
            self._printProgressBar(midi_data.length, midi_data.length, suffix='s')


    def save_midi(self) -> None:
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.mid"
        console.log(f"{self.p}saving recording '{self.outfile}'")
        for note in self.active_midi_file.tracks[0]:
            console.log(f"{self.p}\t{note}")
        
        self.active_midi_file.save(os.path.join(self.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.record_dir, self.outfile)):
            console.log(f"{self.p}successfully saved recording '{self.outfile}'")
            self.last_recorded_file = os.path.join(self.record_dir, self.outfile)
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


    # def wait_for_msg(self) -> None:
