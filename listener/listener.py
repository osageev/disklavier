import os
import mido
from mido import MidiFile, MidiTrack, Message
from threading import Thread
from queue import Queue
from datetime import datetime
import time

from utils import console

class Listener:
    p = '[purple]listen[/purple]: '
    is_recording = False # always either recording or listening
    recorded_notes = []
    active_recording = None

    def __init__(self, params, record_dir) -> None:
        self.params = params
        self.record_dir = record_dir

    
    def listen(self):
        start_time = time.time()
        end_time = 0
        last_note_time = start_time

        dtpb = 480

        with mido.open_input(self.params.in_port) as inport: # type: ignore
            console.log(f"{self.p}listening at {dtpb} ticks per beat")
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
                            console.log(f"{self.p}stopping recording after {end_time - start_time:.02f}s")
                            self.is_recording = False

                            # start playback
                            self._stretch_midi_file(end_time - start_time)
                            self.save_midi()
                            
                            # Thread(target=self.playback_loop(), name="playing").start()
                            # console.log(f"{self.p}playback thread started")
                            self.active_recording = None
                        elif self.is_recording == False:
                            console.log(f"{self.p}recording at {time.time() - start_time:.02f}s")
                            self.is_recording = True
                    elif self.is_recording and msg.type in ['note_on', 'note_off']:
                        if self.active_recording is None:
                            self._new_midi_obj()
                            # set times to start from now
                            start_time = time.time()
                            msg.time = 0
                        self.active_recording.tracks[0].append(msg)
            except KeyboardInterrupt:
                # Stop recording on Ctrl+C
                end_time = time.time()
                console.log(f"{self.p}stopping recording at {end_time}...")


    def start_recording(self, queue: Queue) -> Thread:
        """Starts monitoring MIDI input and recording when pedal is pressed."""
        console.log(f"{self.p}listening on port {self.params.in_port}")
        # record_thread = Thread(target=self._record_loop, args=(queue)).start()
        record_thread = Thread(target=self.spin, args=([queue]))
        record_thread.start()
        return record_thread


    def _record_loop(self, queue) -> None:
        """Monitors MIDI input for pedal and note events."""
        # timing
        start_time = 0
        end_time = 0
        last_note_time = start_time

        dtpb = 480

        with mido.open_input(self.params.in_port) as inport: # type: ignore
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
                            
                            Thread(target=self.playback_loop(), name="playing").start()
                            console.log(f"{self.p}playback thread started")
                            self.active_recording = None
                        elif self.is_recording == False:
                            console.log(f"{self.p}recording...")
                            self.is_recording = True
                    elif self.is_recording and msg.type in ['note_on', 'note_off']:
                        if self.active_recording is None:
                            self._new_midi_obj()
                            start_time = time.time()
                            msg.time = 0
                        self.active_recording.tracks[0].append(msg)
            except KeyboardInterrupt:
                # Stop recording on Ctrl+C
                end_time = time.time()
                console.log(f"{self.p}stopping recording at {end_time}...")


    def save_midi(self):
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.mid"
        console.log(f"{self.p}saving recording '{self.outfile}'")

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for msg in self.recorded_notes:
            track.append(msg)

        mid.save(os.path.join(self.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.record_dir, self.outfile)):
            console.log(f"{self.p}successfully saved recording '{self.outfile}'")
        else:
            console.log(f"{self.p}failed to save recording '{self.outfile}'")

    def _stretch_midi_file(self, new_duration_seconds):
        """"""
        console.log(f"{self.p}rescaling file from {self.active_recording.length:.02f}s to {new_duration_seconds:.02f}s ({new_duration_seconds / self.active_recording.length:.03f})")
        # Calculate stretch factor based on the original duration
        stretch_factor = new_duration_seconds / self.active_recording.length
        
        # Scale the time attribute of each message by the stretch factor
        for track in self.active_recording.tracks:
            for msg in track:
                msg.time = int(msg.time * stretch_factor)

    def _new_midi_obj(self) -> None:
        self.active_recording = MidiFile()
        track = MidiTrack()
        self.active_recording.tracks.append(track)
        track.append(Message('program_change', program=12)) # dunno what this does tbh


