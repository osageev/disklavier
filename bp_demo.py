import mido
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from mido import MidiFile
import mido


def record_audio(duration=5, samplerate=44100):
    """Record audio from the user's microphone.

    Args:
        duration (int): Duration of the recording in seconds. Defaults to 5.
        samplerate (int): Sampling rate in Hz. Defaults to 44100.

    Returns:
        str: Path to the recorded WAV file.
    """
    # record audio
    print("Recording...")
    audio_data = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()
    print("Recording finished.")

    # save to a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, samplerate, audio_data)
    return temp_wav.name


def process_audio_to_midi(wav_path):
    """Process the recorded WAV file to extract MIDI data using Basic Pitch.

    Args:
        wav_path (str): Path to the WAV file.

    Returns:
        str: Path to the generated MIDI file.
    """
    # run Basic Pitch prediction
    _, midi_data, _ = predict(wav_path)

    # save MIDI data to a temporary file
    temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    midi_data.write(temp_midi.name)
    return temp_midi.name


def play_midi(midi_path):
    with mido.open_output("to Max 1") as output:
        # load and play the MIDI file
        mid = MidiFile(midi_path)
        for msg in mid.play():
            output.send(msg)


def main():
    # record audio from the microphone
    wav_path = record_audio(duration=5)

    # process the recorded audio to MIDI
    midi_path = process_audio_to_midi(wav_path)

    # play the MIDI file using the specified SoundFont
    play_midi(midi_path)

    # clean up temporary files
    os.unlink(wav_path)
    os.unlink(midi_path)


if __name__ == "__main__":
    main()
