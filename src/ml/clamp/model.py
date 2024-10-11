import os
import torch
import subprocess
from unidecode import unidecode
from rich.progress import track
from rich.console import Console
from transformers import AutoTokenizer
from .utils import CLaMP, MusicPatchilizer

console = Console(log_time_format="%m-%d %H:%M:%S.%f")
MIDI2ABC = "files/midi2abc"


class Clamp:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            console.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            console.log("No GPU available, using the CPU instead.")
            self.device = torch.device("cpu")

        self.clamp_model_name = "sander-wood/clamp-small-1024"
        self.text_model_name = "distilroberta-base"
        self.text_length = 128
        self.patch_length = 64

        # Load CLaMP model
        self.model = CLaMP.from_pretrained(self.clamp_model_name)
        self.music_length = self.model.config.max_length
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()

        # Initialize patchilizer and tokenizer
        self.patchilizer = MusicPatchilizer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.softmax = torch.nn.Softmax(dim=1)

        if self.verbose:
            console.log("Initialization complete.")

    def abc_filter(self, lines: list) -> str:
        """
        Filter out the metadata from the abc file.

        Args:
            lines (list): List of lines in the abc file.

        Returns:
            music (str): Music string.
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

    def load_music(self, file_path: str) -> str:
        """
        Load music data from a file.

        Args:
            file_path (str): Path to the music file.

        Returns:
            lines (list): List of lines from the music file.
        """
        p = subprocess.Popen(["pwd"], stdout=subprocess.PIPE)
        print(f"pwd result: '{p.communicate()[0].decode('utf-8')}'")
        p = subprocess.Popen([MIDI2ABC, file_path], stdout=subprocess.PIPE)
        result = p.communicate()
        output = result[0].decode("utf-8").replace("\r", "")
        lines = unidecode(output).split("\n")
        return self.abc_filter(lines)

    def encode(self, music: list[str]) -> list[torch.Tensor]:
        """
        Encode the music string into tensor format.

        Args:
            music (str): Music string to encode.

        Returns:
            tensor (torch.Tensor): Encoded tensor representation of the music.
        """
        ids_list = []
        for item in music:
            patches = self.patchilizer.encode(
                item, music_length=self.music_length, add_eos_patch=True
            )
            ids_list.append(torch.tensor(patches).reshape(-1))

        return ids_list

    def get_features(self, ids_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Get the features from the CLaMP model.

        Args:
            ids_list (list): List of encoded music tensors.

        Returns:
            features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size).
        """
        features_list = []
        with torch.no_grad():
            for ids in ids_list:
                ids = ids.unsqueeze(0)
                masks = torch.tensor(
                    [1] * (int(len(ids[0]) / self.patch_length))
                ).unsqueeze(0)
                features = self.model.music_enc(ids, masks)["last_hidden_state"]
                features = self.model.avg_pooling(features, masks)
                features = self.model.music_proj(features)

                features_list.append(features[0])
        return torch.stack(features_list).cpu()

    def run(self, files: list[str]) -> torch.Tensor:
        """
        Run the complete process of loading music, filtering, encoding, and extracting features.

        Args:
            file_path (str): Path to the music file.

        Returns:
            features (torch.Tensor): Extracted features from the music.
        """
        music = [self.load_music(k) for k in files]

        non_empty_keys = []
        non_empty_filenames = []
        for key, pf_file in zip(music, files):
            if key.strip() != "":
                non_empty_keys.append(key)
                non_empty_filenames.append(pf_file)
            else:
                if self.verbose:
                    console.log("File %s not successfully loaded" % (pf_file))

        encoded_music = self.encode(non_empty_keys)
        features = self.get_features(encoded_music)

        return features


if __name__ == "__main__":
    filenames = [
        f[:-4]
        for f in os.listdir("data/datasets/test/dataset samples")
        if f.endswith(".mid")
    ]
    model = Clamp()
    model.run(filenames)
