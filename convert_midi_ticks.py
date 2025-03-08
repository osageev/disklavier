import os
import glob
import mido


def convert_midi_ticks(input_path, output_path, source_ticks, target_ticks):
    """
    Convert a MIDI file from one tick resolution to another.

    Parameters
    ----------
    input_path : str
        Path to the input MIDI file.
    output_path : str
        Path to save the converted MIDI file.
    source_ticks : int
        Source ticks per beat.
    target_ticks : int
        Target ticks per beat.

    Returns
    -------
    None
    """
    # load the midi file
    midi = mido.MidiFile(input_path)

    # check if the file already has the target ticks per beat
    if midi.ticks_per_beat == target_ticks:
        print(f"File {input_path} already has {target_ticks} ticks per beat. Skipping.")
        return

    # create a new midi file with the target ticks per beat
    new_midi = mido.MidiFile(ticks_per_beat=target_ticks)

    # conversion ratio
    ratio = target_ticks / source_ticks

    # copy all tracks with adjusted times
    for track in midi.tracks:
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)

        for msg in track:
            # create a copy of the message
            new_msg = msg.copy()

            # adjust time attribute
            if hasattr(new_msg, "time"):
                new_msg.time = int(new_msg.time * ratio)

            new_track.append(new_msg)

    # save the new midi file
    new_midi.save(output_path)
    print(
        f"Converted {input_path} from {source_ticks} to {target_ticks} ticks per beat. Saved to {output_path}"
    )


def batch_convert(directory, source_ticks=96, target_ticks=220):
    """
    Convert all MIDI files in a directory from one tick resolution to another.

    Parameters
    ----------
    directory : str
        Directory containing MIDI files.
    source_ticks : int
        Source ticks per beat.
    target_ticks : int
        Target ticks per beat.

    Returns
    -------
    None
    """
    # create an output directory if it doesn't exist
    output_dir = os.path.join(directory, "converted")
    os.makedirs(output_dir, exist_ok=True)

    # find all midi files
    midi_files = glob.glob(os.path.join(directory, "*.mid")) + glob.glob(
        os.path.join(directory, "*.midi")
    )

    print(f"Found {len(midi_files)} MIDI files to convert.")

    # convert each file
    for midi_file in midi_files:
        filename = os.path.basename(midi_file)
        output_path = os.path.join(output_dir, filename)
        convert_midi_ticks(midi_file, output_path, source_ticks, target_ticks)

    print(f"Conversion complete. Converted files are in {output_dir}.")


if __name__ == "__main__":
    # set the directory containing midi files
    midi_directory = "data/datasets/test"

    # run the batch conversion from 96 to 220 ticks per beat
    batch_convert(midi_directory, 96, 220)
