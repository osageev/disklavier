import os
import mido
import pretty_midi
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from typing import Dict

from . import basename, console

TICKS_PER_BEAT = 220


def change_tempo(in_path: str, out_path: str, tempo: int):
    midi = mido.MidiFile(in_path)
    new_tempo = mido.bpm2tempo(tempo)
    new_message = mido.MetaMessage("set_tempo", tempo=new_tempo, time=0)
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
        new_track = mido.MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(out_path)


def transform(
    file_path: str, out_dir: str, bpm: int, transformations: Dict, num_beats: int = 8
) -> str:
    f_output = f"{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}.mid"
    pf_out = os.path.join(out_dir, f_output)
    mido.MidiFile(file_path).save(pf_out)  # in case transpose is 0

    if transformations["transpose"] != 0:
        midi_transformed = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for instrument in pretty_midi.PrettyMIDI(pf_out).instruments:
            transposed_instrument = pretty_midi.Instrument(
                program=instrument.program, name=f_output[:-4]
            )
            for note in instrument.notes:
                transposed_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )
            midi_transformed.instruments.append(transposed_instrument)
        midi_transformed.write(pf_out)

    if transformations["shift"] != 0:
        midi_shifted = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        seconds_per_beat = 60 / bpm
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 1) * seconds_per_beat

        for instrument in pretty_midi.PrettyMIDI(pf_out).instruments:
            shifted_instrument = pretty_midi.Instrument(
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
                    pretty_midi.Note(
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
                mido.Message(
                    type=msg_type,
                    note=int(msg_pitch),
                    velocity=int(msg_velocity),
                    time=int(abs_time),
                )
            )

    if not messages:
        console.log("[orange]Warning: No notes found in CSV file")
        return False

    messages.sort(key=lambda msg: msg.time)
    if verbose:
        console.log(messages[:5])
        console.log(messages[-5:])

    # convert absolute timing to relative timing
    for i in range(len(messages) - 1, 0, -1):
        messages[i].time = messages[i].time - messages[i - 1].time
    # messages[0].time = 0

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    track.name = basename(midi_output_path)

    # add messages to track
    for msg in messages:
        track.append(msg)
    midi.tracks.append(track)

    midi.save(midi_output_path)

    console.log(f"[green]MIDI file created: '{midi_output_path}'")
    # if verbose:
    #     console.log("[bold]MIDI File Contents:[/bold]")
    #     midi.print_tracks()

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
    combined = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # add all tracks from all files
    for file_path in input_files:
        midi = mido.MidiFile(file_path)
        for track in midi.tracks:
            # remove all set_tempo messages (one is set in player recording file)
            track[:] = [msg for msg in track if msg.type != "set_tempo"]
            combined.tracks.append(track)

    combined.save(output_path)
    console.log(f"[green]combined MIDI file created: '{output_path}'")

    return os.path.isfile(output_path)


def generate_piano_roll(midi_path, output_path=None, figsize=(12, 8), dpi=100):
    """
    Generate a piano roll visualization from a MIDI file.
    TODO: have live velocity changes be reflected here

    Parameters
    ----------
    midi_path : str
        Path to the MIDI file.
    output_path : str, optional
        Path to save the piano roll image. If None, will save to the same directory
        as the MIDI file with the same name but with .png extension.
    figsize : tuple, optional
        Figure size in inches (width, height).
    dpi : int, optional
        DPI for the output image.

    Returns
    -------
    str
        Path to the generated piano roll image.
    """
    if output_path is None:
        output_path = os.path.splitext(midi_path)[0] + "-pianoroll.png"

    try:
        if not os.path.exists(midi_path):
            console.log(
                f"\t[orange]midi file '{os.path.basename(midi_path)}' not found[/orange]"
            )
            return None

        midi_data = pretty_midi.PrettyMIDI(midi_path)

        plt.figure(figsize=figsize)
        legend_elements = []
        colors = ["red", "blue"]  # player is red, system is blue
        for i, instrument in enumerate(midi_data.instruments):
            if not instrument.notes:
                continue

            for note in instrument.notes:
                plt.fill_betweenx(
                    [note.pitch, note.pitch + 0.8],
                    [note.start, note.start],
                    [note.end, note.end],
                    color=colors[i],
                    alpha=note.velocity / 127.0,
                )

            # add to legend
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[i],
                    lw=4,
                    label=instrument.name.split("-")[0],
                )
            )

        # add legend, labels and title
        plt.legend(handles=legend_elements, loc="upper right")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Pitch")
        plt.title(f"Piano Roll - {os.path.basename(midi_path)}")
        # set y limits to show only the piano range
        plt.ylim(20, 108)
        plt.xlim(0, midi_data.get_end_time())
        # display octaves
        octaves = list(range(24, 108, 12))
        octave_names = ["C" + str(i) for i in range(1, 8)]
        plt.yticks(octaves, octave_names)
        plt.grid(alpha=0.3)
        # save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()

        console.log(
            f"\t[green italic]saved piano roll to '{os.path.basename(output_path)}'[/green italic]"
        )
        return output_path

    except Exception as e:
        console.log(f"\t[red]error generating piano roll: {e}[/red]")
        return None
