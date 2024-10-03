import os
from pathlib import Path
import mido
from mido import bpm2tempo, MidiFile, MetaMessage, MidiTrack
from pretty_midi import PrettyMIDI, Instrument, Note

from typing import Dict


def change_tempo(in_path: str, out_path: str, tempo: int):
    midi = MidiFile(in_path)
    new_tempo = bpm2tempo(tempo)
    new_message = MetaMessage("set_tempo", tempo=new_tempo, time=0)
    tempo_added = False

    for i, track in enumerate(midi.tracks):
        # remove existing set_tempo messages
        tempo_messages = []
        for j, msg in enumerate(track):
            if msg.type == "set_tempo":
                tempo_messages.append(j)

        for index in tempo_messages:
            midi.tracks[i][index] = new_message

        # add new set_tempo message to the first track
        if not tempo_added:
            track.insert(0, new_message)
            tempo_added = True

    # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
    if not tempo_added:
        new_track = MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(out_path)


def transform(file_path: str, out_dir: str, bpm: int, transformations: Dict, num_beats: int = 8) -> str:
    f_output = f"{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}.mid"
    pf_out = os.path.join(out_dir, f_output)
    MidiFile(file_path).save(pf_out) # in case transpose is 0

    if transformations["transpose"] != 0:
        midi_transformed = PrettyMIDI(initial_tempo=bpm)
        for instrument in PrettyMIDI(pf_out).instruments:
            transposed_instrument = Instrument(program=instrument.program, name=f_output[:-4])
            for note in instrument.notes:
                transposed_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )
            midi_transformed.instruments.append(transposed_instrument)
        midi_transformed.write(pf_out)
        
    if transformations["shift"] != 0:
        midi_shifted = PrettyMIDI(initial_tempo=bpm)
        seconds_per_beat = 60 / bpm
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 1) * seconds_per_beat

        for instrument in PrettyMIDI(pf_out).instruments:
            shifted_instrument = Instrument(
                program=instrument.program, name=f_output[:-4]
            )
            for note in instrument.notes:
                dur = note.end - note.start
                shifted_start = (note.start + shift_seconds) % loop_point
                shifted_end = shifted_start + dur

                if note.start + shift_seconds >= loop_point:
                    shifted_start += seconds_per_beat
                    shifted_end += seconds_per_beat

                shifted_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=shifted_start,
                        end=shifted_end
                    )
                )

            midi_shifted.instruments.append(shifted_instrument)

        midi_shifted.write(pf_out)

    change_tempo(pf_out, pf_out, bpm)

    return pf_out

