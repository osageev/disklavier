import os
import zipfile
from rich import print
from glob import glob
import soundfile as sf
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from midi2audio import FluidSynth

# sf = "src/ml/clap/Yamaha_C7__Normalized_.sf2"
sf = "src/ml/clap/alex_gm.sf2"
sr = 48000
fs = FluidSynth(sf, sr)

basename = lambda x: os.path.splitext(os.path.basename(x))[0]

def midi_to_wav(
    input_folder: str,
    output_folder: str,
) -> int:
    midi_files = glob(os.path.join(input_folder, "*.mid"))
    if not midi_files:
        print(f"[red]Warning: No MIDI files found in input folder '{input_folder}'")
        return -1

    os.makedirs(output_folder, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    wav_task = progress.add_task(f"synthesizing wav files", total=len(midi_files))
    n_files = 0
    with progress:
        for midi_path in midi_files:
            try:
                filename = basename(midi_path) + ".wav"
                output_path = os.path.join(output_folder, filename)
                fs.midi_to_audio(midi_path, output_path)
                n_files += 1
            except Exception as e:
                print(f"[yellow]Error converting '{midi_path}' to WAV: {e}")

            progress.advance(wav_task)

    print(f"[green]converted {n_files}/{len(midi_files)} MIDI files to WAV files")

    return n_files


in_path = "/media/nova/Datasets/sageev-midi/20250320/segmented"
out_path = "/media/scratch/sageev-midi/20250320/wavs-alex_gm"

midi_to_wav(in_path, out_path)

print(f"zipping...")
zip_path = os.path.join(os.path.dirname(out_path), "wavs-alex_gm.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(out_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(out_path))
            zipf.write(file_path, arcname)
print(f"Created zip archive at: {zip_path}")
