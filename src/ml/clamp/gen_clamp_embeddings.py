import os
import torch
import subprocess
import pandas as pd
from unidecode import unidecode
from utils import CLaMP, MusicPatchilizer
from transformers import AutoTokenizer
from rich.progress import track
from rich.console import Console

CLAMP_MODEL_NAME = "sander-wood/clamp-small-1024"
TEXT_MODEL_NAME = "distilroberta-base"
TEXT_LENGTH = 128
PATCH_LENGTH = 64
SKIP_MXL = True

console = Console(log_time_format="%m-%d %H:%M:%S.%f")
dataset = "test"
p_in = os.path.join("data", "datasets", dataset, "synthetic")
p_out = os.path.join("inference", dataset)

# init torch device
if torch.cuda.is_available():
    device = torch.device("cuda")
    console.log(f"using GPU {torch.cuda.get_device_name(0)}")

else:
    console.log("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# load CLaMP model
model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)
music_length = model.config.max_length
model = model.to(device)
model.eval()

# initialize patchilizer, tokenizer, and softmax
patchilizer = MusicPatchilizer()
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
softmax = torch.nn.Softmax(dim=1)


def mid2mxl(p_file: str) -> str:
    file = os.path.basename(p_file)
    pf_out = os.path.join(p_out, os.path.basename(file)[:-4] + ".mxl")
    try:
        score = converter.parseFile(os.path.join(p_in, file), format="midi")
        pf_out = score.write(fmt="musicxml", fp=pf_out)
        console.log(f"score for '{pf_out}':\n{score.show('text')}")
    except Exception as e:
        console.log(f"Error converting MIDI to abc: {e}")

    return pf_out


def abc_filter(lines: list) -> str:
    """
    Filter out the metadata from the abc file

    Args:
        lines (list): List of lines in the abc file

    Returns:
        music (str): Music string
    """
    music = ""
    for line in lines:
        if (
            line[:2]
            in [
                "A:",
                "B:",
                "C:",
                "D:",
                "F:",
                "G",
                "H:",
                "N:",
                "O:",
                "R:",
                "r:",
                "S:",
                "T:",
                "W:",
                "w:",
                "X:",
                "Z:",
            ]
            or line == "\n"
            or (line.startswith("%") and not line.startswith("%%score"))
        ):
            continue
        else:
            if "%" in line and not line.startswith("%%score"):
                line = "%".join(line.split("%")[:-1])
                music += line[:-1] + "\n"
            else:
                music += line + "\n"
    return unidecode(music)


def load_music(pf_file: str) -> str:
    """
    Load the music from the xml file

    Args:
        filename (str): Path to the xml file

    Returns:
        music (str): Music string
    """
    # console.log(f"converting '{pf_file}'")
    if SKIP_MXL:
        # console.log("skipping intermediate musicxml conversion")
        args = ["ml/clamp/midi2abc", pf_file]
    else:
        path = mid2mxl(pf_file)
        args = (
            [
                "python",
                "inference/xml2abc.py",
                "-m",
                "2",
                "-c",
                "6",
                "-x",
                path,
            ],
        )
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    result = p.communicate()
    output = result[0].decode("utf-8").replace("\r", "")
    music = unidecode(output).split("\n")
    return abc_filter(music)


def convert(files: list[str]):
    keys = [load_music(k) for k in track(files, "converting...")]

    non_empty_keys = []
    non_empty_filenames = []
    for key, pf_file in zip(keys, files):
        if key.strip() != "":
            non_empty_keys.append(key)
            non_empty_filenames.append(pf_file)
        else:
            console.log("File %s not successfully loaded" % (pf_file))

    return non_empty_keys, non_empty_filenames


def encoding_data(data: list[str]) -> list[torch.Tensor]:
    """
    Encode the data into ids

    Args:
        data (list): List of strings

    Returns:
        ids_list (list): List of ids
    """
    ids_list = []
    for item in track(data, "encoding..."):
        patches = patchilizer.encode(
            item, music_length=music_length, add_eos_patch=True
        )
        ids_list.append(torch.tensor(patches).reshape(-1))

    return ids_list


def get_features(ids_list: list) -> list[torch.Tensor]:
    """
    Get the features from the CLaMP model

    Args:
        ids_list (list): List of ids

    Returns:
        features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
    """
    features_list = []
    with torch.no_grad():
        for ids in track(ids_list, "locating..."):
            ids = ids.unsqueeze(0)
            masks = torch.tensor([1] * (int(len(ids[0]) / PATCH_LENGTH))).unsqueeze(0)
            features = model.music_enc(ids, masks)["last_hidden_state"]
            features = model.avg_pooling(features, masks)
            features = model.music_proj(features)

            features_list.append(features[0].cpu())

    return features_list


def main():
    files = [os.path.join(p_in, f) for f in os.listdir(p_in) if f.endswith(".mid") or f.endswith(".midi")
f.endswith(".midi")]
    files.sort()
    console.log(f"Loading {len(files)} segments...")
    keys, filenames = convert(files)
    ids = encoding_data(keys)
    embeddings = get_features(ids)
    df = pd.DataFrame(
        {
            "abc": keys,
            "id": [id.numpy() for id in ids],
            "embedding": [e.numpy() for e in embeddings],
        },
        index=[os.path.basename(f)[:-4] for f in filenames],
    )
    console.log(f"generated dataframe {df.shape}", df.head())
    df.to_parquet(f"{dataset}-clamp.parquet")
    console.log(f"[green bold]done")


if __name__ == "__main__":
    main()
