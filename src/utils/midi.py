import os
from pathlib import Path
from pretty_midi import PrettyMIDI, Instrument, Note
from mido import bpm2tempo, MidiFile, MetaMessage, MidiTrack, Message

from typing import Dict

from utils import basename, console

TICKS_PER_BEAT = 220


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


def transform(
    file_path: str, out_dir: str, bpm: int, transformations: Dict, num_beats: int = 8
) -> str:
    f_output = f"{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}.mid"
    pf_out = os.path.join(out_dir, f_output)
    MidiFile(file_path).save(pf_out)  # in case transpose is 0

    if transformations["transpose"] != 0:
        midi_transformed = PrettyMIDI(initial_tempo=bpm)
        for instrument in PrettyMIDI(pf_out).instruments:
            transposed_instrument = Instrument(
                program=instrument.program, name=f_output[:-4]
            )
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
                        end=shifted_end,
                    )
                )

            midi_shifted.instruments.append(shifted_instrument)

        midi_shifted.write(pf_out)

    change_tempo(pf_out, pf_out, bpm)

    return pf_out


def csv_to_midi(csv_path: str, midi_output_path: str, verbose: bool = False) -> bool:
    """
    Convert a CSV file containing piano note data to a MIDI file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with piano note data.
    midi_output_path : str
        Path where the MIDI file will be saved.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    bool
        True if the MIDI file was created successfully, False otherwise.
    """
    messages = []
    with open(csv_path, "r") as file:
        next(file)
        for line in file:
            _, msg_type, msg_pitch, msg_velocity, abs_time = line.strip().split(",")
            messages.append(
                Message(
                    type=msg_type,
                    note=int(msg_pitch),
                    velocity=int(msg_velocity),
                    time=int(abs_time),
                )
            )

    if not messages:
        console.log("[yellow]Warning: No notes found in CSV file")
        return False

    messages.sort(key=lambda msg: msg.time)
    if verbose:
        console.log(messages[:5])
        console.log(messages[-5:])

    # convert absolute timing to relative timing
    for i in range(len(messages) - 1, 0, -1):
        messages[i].time = messages[i].time - messages[i - 1].time
    # messages[0].time = 0

    midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    track.name = basename(midi_output_path)

    # add messages to track
    for msg in messages:
        track.append(msg)
    midi.tracks.append(track)

    midi.save(midi_output_path)

    console.log(f"[green]MIDI file created: '{midi_output_path}'")
    if verbose:
        console.log("[bold]MIDI File Contents:[/bold]")
        midi.print_tracks()

    return os.path.isfile(midi_output_path)


def combine_midi_files(input_files: list[str], output_path: str) -> bool:
    """
    Combine multiple MIDI files into a single MIDI file, preserving separate tracks.

    Parameters
    ----------
    input_files : list[str]
        List of paths to input MIDI files.
    output_path : str
        Path where the combined MIDI file will be saved.

    Returns
    -------
    bool
        True if the combined MIDI file was created successfully, False otherwise.
    """
    combined = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # add all tracks from all files
    for file_path in input_files:
        midi = MidiFile(file_path)
        for track in midi.tracks:
            combined.tracks.append(track)

    combined.save(output_path)
    console.log(f"[green]combined MIDI file created: '{output_path}'")

    return os.path.isfile(output_path)
