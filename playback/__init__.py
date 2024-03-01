import os
import mido
from mido import MidiFile, MidiTrack, Message
import threading
from datetime import datetime
import time

from similarity import Similarity


class Player():
    is_recording = False
    recorded_notes = []
    ctrl = 67  # soft pedal
    active_midi_file = None
    is_playing = False

    def __init__(self, similarity: Similarity, record_dir: str, params) -> None:
        print("player initializing")
        print(f"found input ports: {mido.get_input_names()}") # type: ignore
        print(f"found output ports: {mido.get_output_names()}") # type: ignore

        self.params = params
        self.record_dir = record_dir
        self.similarity = similarity
        self.input_port = self.params.in_port
        self.output_port = self.params.out_port
        # self.listener = Listener(self.params.listener)
        # self.logger = logging.getLogger(name="main.player")


    def start_recording(self) -> None:
        """Starts monitoring MIDI input and recording when pedal is pressed."""
        print(f"listening on port {self.input_port}")
        threading.Thread(target=self._record_loop, name="recording").start()


    def _record_loop(self) -> None:
        """Monitors MIDI input for pedal and note events."""
        # timing
        start_time = 0
        end_time = 0
        last_note_time = start_time

        dtpb = 480

        with mido.open_input(self.input_port) as inport: # type: ignore
            print(f"recording at {dtpb}tpb... Press Ctrl+C to stop.")
            try:
                for msg in inport:
                    # record delta time of input message 
                    # mido doesn't do this by default for some reason
                    current_time = time.time()
                    msg.time = int((current_time - last_note_time) * dtpb)
                    print(f"\t{msg}")
                    last_note_time = current_time 

                    if msg.type == 'control_change' and msg.control == self.params.ctrl:
                        if msg.value == 0:
                            end_time = time.time()
                            print(f"stopping recording at {end_time - start_time:.02f}s...")
                            self.is_recording = False

                            # start playback
                            self._stretch_midi_file(end_time - start_time)
                            self.save_midi()
                            threading.Thread(target=self.playback_loop(), name="playing").start()
                            print(f"playback thread started")
                            self.active_midi_file = None
                        elif self.is_recording == False:
                            print(f"recording...")
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
                print(f"stopping recording at {end_time}...")


    def playback_loop(self):
        """"""
        time_elapsed = 0.
        self.playing = True
        self.playing_file_path = self.last_recorded_file
        recorded_ph = self.similarity.midi_to_ph(self.last_recorded_file)
        next_file, similarity = self.similarity.find_most_similar_vector(recorded_ph)
        next_file_path = os.path.join(self.similarity.input_dir, next_file)
        first_loop = True

        while next_file is not None:
            self.playing_file = os.path.basename(self.playing_file_path)
            print(f"{self.playing_file_path}\t{self.playing_file}")
            print(f"{next_file_path}\t{next_file}")
            if not first_loop: # bad code!
                print(f"getting next file for {self.playing_file_path}")
                self.similarity.metrics[self.playing_file]['played'] = 1
                next_file, similarity = self.similarity.get_most_similar_file(self.playing_file)
                next_file_path = os.path.join(self.similarity.input_dir, next_file)

            print(f"[{int(time_elapsed):04d}]\tPlaying {self.playing_file}\t(next up is {next_file})\tsim={similarity:.3f}")
            time_elapsed += MidiFile(self.playing_file_path).length

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = next_file_path
            first_loop = False
        

    def play_midi_file(self, midi_path: str) -> None:
        midi_data = MidiFile(midi_path)
        with mido.open_output(self.output_port) as outport: # type: ignore
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
        print(f"saving recording '{self.outfile}'")
        for note in self.active_midi_file.tracks[0]:
            print(f"\t{note}")
        
        self.active_midi_file.save(os.path.join(self.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.record_dir, self.outfile)):
            print(f"successfully saved recording '{self.outfile}'")
            self.last_recorded_file = os.path.join(self.record_dir, self.outfile)
        else:
            print(f"failed to save recording '{self.outfile}'")


    def _stretch_midi_file(self, new_duration_seconds):
        """"""
        print(f"rescaling file from {self.active_midi_file.length:.02f}s to {new_duration_seconds:.02f}s ({new_duration_seconds / self.active_midi_file.length:.03f})")
        # Calculate stretch factor based on the original duration
        stretch_factor = new_duration_seconds / self.active_midi_file.length
        
        # Scale the time attribute of each message by the stretch factor
        for track in self.active_midi_file.tracks:
            for msg in track:
                msg.time = int(msg.time * stretch_factor)


    def _new_midi_obj(self) -> None:
        self.active_midi_file = MidiFile()
        track = MidiTrack()
        self.active_midi_file.tracks.append(track)
        track.append(Message('program_change', program=12)) # dunno what this does tbh


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