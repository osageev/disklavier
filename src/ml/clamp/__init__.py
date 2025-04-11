import os
import mido

def basename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode("unicode_escape").decode("utf-8")


def load_midi(filename: str, m3_compatible: bool = True) -> str:
    """
    Load a MIDI file and convert it to MTF format.
    """
    mid = mido.MidiFile(filename)
    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

    # Traverse the MIDI file
    for msg in mid.merged_track:
        if m3_compatible:
            if msg.is_meta:
                if msg.type in [
                    "text",
                    "copyright",
                    "track_name",
                    "instrument_name",
                    "lyrics",
                    "marker",
                    "cue_marker",
                    "device_name",
                ]:
                    continue
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)

    return "\n".join(msg_list)


def convert_midi2mtf(
    pf_midi_in: str, pf_midi_out: str, m3_compatible: bool = True
) -> None:
    """
    Converts MIDI files to MTF format.
    """
    try:
        output = load_midi(pf_midi_in, m3_compatible)

        if not output:
            with open("logs/midi2mtf_error_log.txt", "a", encoding="utf-8") as f:
                f.write(pf_midi_in + "\n")
            return
        else:
            output_file_path = os.path.join(pf_midi_out, basename(pf_midi_in) + ".mtf")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(output)
    except Exception as e:
        with open("logs/midi2mtf_error_log.txt", "a", encoding="utf-8") as f:
            f.write(pf_midi_in + " " + str(e) + "\n")
        pass
