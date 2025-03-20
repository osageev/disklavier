import os
import mido
import pretty_midi
from matplotlib import pyplot as plt
import numpy as np

from utils import basename
from utils.midi import trim_piano_roll


def add_metronome(midi_file_path: str, num_beats: int) -> None:
    """
    adds a clave on each beat of the segment.

    Parameters
    ----------
    midi_file_path : str
        path to the midi file.
    num_beats : int
        number of beats in the segment.

    Returns
    -------
    None
    """
    midi = mido.MidiFile(midi_file_path)

    # create a new track for the clave
    clave_track = mido.MidiTrack()
    clave_track.append(mido.MetaMessage("track_name", name="metronome", time=0))

    # add clave notes on each beat
    # +1 because we want to include the last beat
    for beat in range(num_beats + 1):
        time_ticks = midi.ticks_per_beat - midi.ticks_per_beat // 8 if beat > 0 else 0
        clave_track.append(
            mido.Message("note_on", note=76, velocity=100, time=time_ticks, channel=9)
        )

        # note off - make it short (1/8 of a beat)
        clave_track.append(
            mido.Message(
                "note_off",
                note=76,
                velocity=0,
                time=midi.ticks_per_beat // 8,
                channel=9,
            )
        )

    clave_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(clave_track)
    midi.save(midi_file_path)


def add_novelty(
    midi_file_path: str,
    novelty: np.ndarray,
    num_beats: int,
    times: tuple[float, float],
    pic_dir: str,
) -> None:
    """
    Adds a novelty track to the MIDI file.
    TODO: properly convert time from PR to ticks (100 -> 220 and only log every 16th note?)
            i dunno but do this carefully and check

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file.
    novelty : np.ndarray
        The novelty curve.
    num_beats : int
        The number of beats in the segment.
    times : tuple[float, float]
        The start and end times of the segment.
    pic_dir : str
        The directory to save the novelty curve.

    Returns
    -------
    None
    """
    midi = mido.MidiFile(midi_file_path)
    piano_roll = pretty_midi.PrettyMIDI(midi_file_path).get_piano_roll()
    novelty_track = mido.MidiTrack()
    novelty_track.append(mido.MetaMessage("track_name", name="novelty", time=0))

    # find the index of the start and end times in the novelty curve
    start_index = int(times[0] * 100)
    end_index = int(times[1] * 100)
    novelty = novelty[start_index:end_index]

    # add the novelty curve to the track
    n_msgs = [
        mido.MetaMessage("text", text=f"{n:.03f}", time=i)
        for i, n in enumerate(novelty)
    ]
    novelty_track.extend(n_msgs)

    novelty_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(novelty_track)
    midi.save(midi_file_path)

    # save novelty curve
    plt.figure(figsize=(8, 4))
    plt.imshow(
        trim_piano_roll(piano_roll),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    plt.plot(
        (1 - novelty / novelty.max()) * 17,
        "g",
        linewidth=1.0,
        alpha=0.7,
    )
    plt.axis("off")
    plt.savefig(os.path.join(pic_dir, f"{basename(midi_file_path)}_novelty.png"))
    plt.close()


def modify_end_of_track(midi_file_path: str, new_end_time: float, bpm: int) -> None:
    """
    Modifies the 'end_of_track' message in a MIDI file to match the new end time.

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file.
    new_end_time : float
        The new end time.
    bpm : int
        The BPM of the MIDI file.
    """
    midi = mido.MidiFile(midi_file_path)
    new_end_time_t = mido.second2tick(new_end_time, 220, mido.bpm2tempo(bpm))

    for _, track in enumerate(midi.tracks):
        total_time_t = 0
        # Remove existing 'end_of_track' messages and calculate last note time
        for msg in track:
            if msg.type == "note_on":
                total_time_t += msg.time
            if msg.type == "end_of_track":
                track.remove(msg)
                # Add a new 'end_of_track' message at the calculated offset time
                offset = (
                    new_end_time_t - total_time_t
                    if new_end_time_t > total_time_t
                    else 0
                )
                track.append(mido.MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    midi.save(midi_file_path)
