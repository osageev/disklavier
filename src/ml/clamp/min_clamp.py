import os
import torch
import subprocess
import pandas as pd
from unidecode import unidecode
from utils import CLaMP, MusicPatchilizer
from rich import print, progress
from transformers import AutoTokenizer

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"using GPU {torch.cuda.get_device_name(0)}")

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

CLAMP_MODEL_NAME = "sander-wood/clamp-small-1024"
TEXT_MODEL_NAME = "distilroberta-base"
TEXT_LENGTH = 128
PATCH_LENGTH = 64

# load CLaMP model
model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)
music_length = model.config.max_length
model = model.to(device)
model.eval()

# initialize patchilizer, tokenizer, and softmax
patchilizer = MusicPatchilizer()
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
softmax = torch.nn.Softmax(dim=1)


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
    return music


def load_music(filename: str) -> str:
    """
    Load the music from the xml file

    Args:
        filename (str): Path to the xml file

    Returns:
        music (str): Music string
    """
    p = subprocess.Popen(
        [
            "python",
            "inference/xml2abc.py",
            "-m",
            "2",
            "-c",
            "6",
            "-x",
            filename,
        ],
        stdout=subprocess.PIPE,
    )
    result = p.communicate()
    output = result[0].decode("utf-8").replace("\r", "")
    music = unidecode(output).split("\n")
    return abc_filter(music)


def encoding_data(data: list[str]) -> list[torch.Tensor]:
    """
    Encode the data into ids

    Args:
        data (list): List of strings

    Returns:
        ids_list (list): List of ids
    """
    ids_list = []
    for item in progress.track(data, "encoding", len(data)):
        patches = patchilizer.encode(
            item, music_length=music_length, add_eos_patch=True
        )
        ids_list.append(torch.tensor(patches).reshape(-1))

    return ids_list


def get_features(ids_list):
    """
    Get the features from the CLaMP model

    Args:
        ids_list (list): List of ids

    Returns:
        features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
    """
    features_list = []
    with torch.no_grad():
        for ids in progress.track(ids_list, "Extracting features..."):
            ids = ids.unsqueeze(0)
            masks = torch.tensor([1] * (int(len(ids[0]) / PATCH_LENGTH))).unsqueeze(0)
            features = model.music_enc(ids, masks)["last_hidden_state"]
            features = model.avg_pooling(features, masks)
            features = model.music_proj(features)

            features_list.append(features[0])

    return torch.stack(features_list).to(device)


def main():
    # load keys
    keys = []
    key_filenames = []

    # load filenames
    for root, dirs, files in os.walk("../../inference/20240621"):
        for file in files:
            filename = root + "/" + file
            if filename.endswith(".abc"):
                key_filenames.append(filename)
    print(f"Loading music from {len(key_filenames)} files...")

    # convert to abc
    for filename in progress.track(key_filenames):
        print(f"loading '{filename}'")
        keys.append(filename)
    # encode keys
    key_features = []
    if len(keys) > 0:
        key_ids = encoding_data(keys)
        key_features = get_features(key_ids)

        # save to dataframe
        print(f"saving file")
        df = pd.DataFrame(
            {
                "filenames": [os.path.basename(k) for k in key_filenames],
                "keys": keys,
                "ids": key_ids,
                "features": [f.cpu() for f in key_features],
            }
        )
        df.to_parquet("20240621_all.parquet")
    print("DONE")


if __name__ == "__main__":
    main()
