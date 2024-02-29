import os
import mido
from mido import MidiFile, MidiTrack, Message
import threading
from datetime import datetime

from similarity import Similarity
# from playback.listener import Listener


class Player():
    is_recording = False
    recorded_notes = []
    ctrl = 64  # CC 64 is the standard sustain pedal
    sustain_pedal_value = 127  # Value might need adjustment

    def __init__(self, params) -> None:
        print("player initializing")
        print(f"found input ports: {mido.get_input_names()}")
        print(f"found output ports: {mido.get_output_names()}")

        self.params = params
        self.listener = Listener(self.params.listener)
        # self.logger = logging.getLogger(name="main.player")
        self.input_port = self.params.in_port
        self.output_port = self.params.out_port


    def start_recording(self) -> None:
        """Starts monitoring MIDI input and recording when pedal is pressed."""
        print(f"listening on port {self.input_port}")
        threading.Thread(target=self._record_loop).start()


    def _record_loop(self) -> None:
        """Monitors MIDI input for pedal and note events."""
        with mido.open_input(self.input_port) as inport:
            for msg in inport:
                print(msg)
                if msg.type == 'control_change' and msg.control == self.params.ctrl:
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


    def play_midi_file(self, midi_data: mido.MidiFile, midi_port) -> None:
        t = 0
        for msg in midi_data.play():
            self._printProgressBar(t, midi_data.length, suffix='s')
            t += msg.time # type: ignore
            midi_port.send(msg)
        self._printProgressBar(midi_data.length, midi_data.length, suffix='s')


    def _printProgressBar (self, iteration, total, prefix = '', suffix = '', fill = '█', printEnd = "\r") -> None:
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

        TODO: replace with tqdm
        """
        bar_length = 100
        # decimals = 1
        # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        percent = f"{iteration:.2f}/{total:.2f}"
        filledLength = int(bar_length * iteration // total)
        bar = fill * filledLength + '-' * (bar_length - filledLength)
        print(f'\r{prefix} |{bar}| {percent} {suffix}', end = printEnd)

        if iteration == total: 
            print()