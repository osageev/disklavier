import os
import mido
from mido import MidiFile, MidiTrack, Message
import threading
from datetime import datetime

class Listener:
    is_recording = False
    recorded_notes = []
    ctrl = 64  # CC 64 is the standard sustain pedal
    sustain_pedal_value = 127  # Value might need adjustment

    def __init__(self, params) -> None:
        self.params = params
        self.input_port = self.params.in_port
        self.output_port = self.params.out_port

        if self.params.ctrl:
            self.ctrl = self.params.ctrl

        if not os.path.exists(self.params.record_dir):
            print(f"creating new recording folder: '{self.params.record_dir}'")
            os.mkdir(self.params.record_dir)


    def start_recording(self) -> None:
        """Starts monitoring MIDI input and recording when pedal is pressed."""
        print(f"listening on port {self.input_port}")
        threading.Thread(target=self._record_loop).start()


    def _record_loop(self) -> None:
        """Monitors MIDI input for pedal and note events."""
        with mido.open_input(self.input_port) as inport:
            for msg in inport:
                print(msg)
                if msg.type == 'control_change' and msg.control == self.ctrl:
                    if msg.value == self.sustain_pedal_value:
                        self.is_recording = True
                        self.recorded_notes.clear()
                    else:
                        self.is_recording = False
                        self.save_midi()
                elif self.is_recording and msg.type in ['note_on', 'note_off']:
                    self.recorded_notes.append(msg)


    def save_midi(self):
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.mid"
        print(f"saving recording '{self.outfile}'")
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for msg in self.recorded_notes:
            track.append(msg)
        mid.save(os.path.join(self.params.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.params.record_dir, self.outfile)):
            print(f"successfully saved recording '{self.outfile}'")
        else:
            print(f"failed to save recording '{self.outfile}'")


