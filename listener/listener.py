import os
import mido
from mido import MidiFile, MidiTrack, Message
from threading import Event
from queue import Queue
from datetime import datetime
import time
from rich.json import JSON

from utils import console

class Listener:
    p = '[purple]listen[/purple]: '
    is_recording = False # always either recording or listening
    recorded_notes = []
    outfile = ''

    def __init__(self, params, record_dir, rec_event: Event, kill_event: Event) -> None:
        self.params = params
        self.record_dir = record_dir
        self.ready_event = rec_event
        self.kill_event = kill_event
    
    def listen(self):
        start_time = time.time()
        end_time = 0
        last_note_time = start_time

        dtpb = 480

        with mido.open_input(self.params.in_port) as inport: # type: ignore
            console.log(f"{self.p}listening at {dtpb} ticks per beat")
            for msg in inport:
                # record delta time of input message 
                # mido doesn't do this by default for some reason
                current_time = time.time()
                msg.time = int((current_time - last_note_time) * dtpb)
                console.log(f"{self.p}\t{msg.dict()}")
                last_note_time = current_time 

                if msg.type == 'control_change' and msg.control == self.params.ctrl:
                    if msg.value == 0:
                        end_time = time.time()
                        console.log(f"{self.p}stopping recording after {end_time - start_time:.02f}s")
                        self.is_recording = False
                        if len(self.recorded_notes) == 0:
                            console.log(f"{self.p}no notes recorded")
                            continue

                        # save file and notify overseer that its ready
                        self.save_midi(end_time - start_time)
                        self.ready_event.set()
                    elif self.is_recording == False:
                        console.log(f"{self.p}recording at {time.time() - start_time:.02f}s")
                        self.is_recording = True
                elif self.is_recording and msg.type in ['note_on', 'note_off']:
                    if len(self.recorded_notes) == 0:
                        # set times to start from now
                        start_time = time.time()
                        msg.time = 0
                    self.recorded_notes.append(msg)
                
            if self.kill_event.is_set():
                console.log(f"{self.p}shutting down")
                return


    def save_midi(self, dt):
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.mid"
        console.log(f"{self.p}saving recording '{self.outfile}'")

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(Message('program_change', program=12)) # dunno what this does tbh
        for msg in self.recorded_notes:
            track.append(msg)

        mid = self._stretch_midi_file(mid, dt)

        mid.save(os.path.join(self.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.record_dir, self.outfile)):
            console.log(f"{self.p}successfully saved recording '{self.outfile}'")
        else:
            console.log(f"{self.p}failed to save recording '{self.outfile}'")


    def _stretch_midi_file(self, file: MidiFile, new_duration_seconds):
        """"""
        console.log(f"{self.p}rescaling file from {file.length:.02f}s to {new_duration_seconds:.02f}s ({new_duration_seconds / file.length:.03f})")
        # Calculate stretch factor based on the original duration
        stretch_factor = new_duration_seconds / file.length
        
        # Scale the time attribute of each message by the stretch factor
        for track in file.tracks:
            for msg in track:
                msg.time = int(msg.time * stretch_factor)

        return file


