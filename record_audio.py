import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import datetime
import time


def record_audio(
    output_dir: str = "data",
    duration: int = 5,
    sample_rate: int = 44100,
    channels: int = 1,
):
    """
    record audio from the default microphone using an input stream and save it to a .wav file.

    parameters
    ----------
    output_dir : str, optional
        directory to save the recording in, by default "data".
    duration : int, optional
        recording duration in seconds, by default 5.
    sample_rate : int, optional
        audio sample rate in hz, by default 44100.
    channels : int, optional
        number of audio channels, by default 1 (mono).

    returns
    -------
    str
        path to the saved audio file, or none if an error occurred.
    """
    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"created directory: {output_dir}")

    # generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d_%h%m%s")
    filename = f"recording_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)

    print(f"recording for {duration} seconds...")

    recorded_data = []

    def callback(indata, frames, time_info, status):
        """this is called (from a separate thread) for each audio block."""
        if status:
            print(f"stream status: {status}")
        recorded_data.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=channels, callback=callback
        ):
            time.sleep(duration)  # record for the specified duration

    except Exception as e:
        print(f"error during recording stream: {e}")
        return None

    print("recording complete.")

    if not recorded_data:
        print("no audio data recorded.")
        return None

    # combine recorded blocks and save
    try:
        recording = np.concatenate(recorded_data, axis=0)
        sf.write(filepath, recording, sample_rate)
        print(f"recording saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"error saving file: {e}")
        return None


if __name__ == "__main__":
    # example usage: record 5 seconds of audio
    record_audio(duration=5)
