import os
import sys
import time
import torch
from argparse import ArgumentParser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from ml import model_list
from ml.clap.model import Clap
from ml.clamp.model import Clamp
from ml.classifier.model import Classifier
from ml.specdiff.model import SpectrogramDiffusion, config

from utils import basename, console

SUPPORTED_EXTENSIONS = (".mid", ".midi", ".wav")
REQUIRES_SPECDIFF = ("clf-4note", "clf-speed", "clf-tpose")

class UploadHandler(FileSystemEventHandler):
    tag = "[#5f00af]panthr[/#5f00af]:"

    models = {}

    def __init__(self):
        # TODO: there has to be a better way to do this
        console.log(f"{self.tag} loading clap model")
        self.clap = Clap()
        self.models["clap"] = self.clap
        console.log(f"{self.tag} loading clamp model")
        self.clamp = Clamp()
        self.models["clamp"] = self.clamp
        console.log(f"{self.tag} loading specdiff model")
        self.specdiff = SpectrogramDiffusion(config)  # weird way to do this but w/e
        self.models["specdiff"] = self.specdiff
        console.log(f"{self.tag} loading 4 note classifier model")
        self.clf_4note = Classifier(input_dim=768, hidden_dims=[128], output_dim=120)
        self.clf_4note.load_state_dict(
            torch.load(
                os.path.join(os.getcwd(), "data", "models", "clf-4note.pth"),
                weights_only=True,
            ),
            strict=False,
        )
        self.clf_4note.eval()
        self.models["clf-4note"] = self.clf_4note
        console.log(f"{self.tag} loading speed classifier model")
        self.clf_speed = Classifier(input_dim=768, hidden_dims=[128], output_dim=120)
        self.clf_speed.load_state_dict(
            torch.load(
                os.path.join(os.getcwd(), "data", "models", "clf-speed.pth"),
                weights_only=True,
            ),
            strict=False,
        )
        self.clf_speed.eval()
        self.models["clf-speed"] = self.clf_speed
        console.log(f"{self.tag} loading transpose classifier model")
        self.clf_transpose = Classifier(
            input_dim=768, hidden_dims=[128], output_dim=120
        )
        self.clf_transpose.load_state_dict(
            torch.load(
                os.path.join(os.getcwd(), "data", "models", "clf-tpose.pth"),
                weights_only=True,
            ),
            strict=False,
        )
        self.clf_transpose.eval()
        self.models["clf-tpose"] = self.clf_transpose
        console.log(f"{self.tag}[green] initialization complete")

    def on_created(self, event):
        uploaded_file = str(event.src_path)
        if not uploaded_file.endswith(SUPPORTED_EXTENSIONS):
            return

        # TODO: filetype/model input type compatibility typechecking
        time.sleep(0.5)  # allow file to finish uploading
        requested_model = basename(uploaded_file).split("_")[-1]
        console.log(
            f"{self.tag} sending file '{event.src_path}' to be embedded by {requested_model}"
        )
        if requested_model in REQUIRES_SPECDIFF:
            pre_embed = self.specdiff.embed(uploaded_file)
            tmp_path = f"{os.path.splitext(uploaded_file)[0]}_tmp.pt"
            torch.save(pre_embed, tmp_path)
            console.log(
                f"{self.tag} wrote pre-embedding to '{tmp_path}'"
            )
            uploaded_file = tmp_path

        embedding = self.models[requested_model].embed(uploaded_file)
        console.log(f"{self.tag} got embedding {embedding.shape}")
        torch.save(embedding, f"{os.path.splitext(event.src_path)[0]}.pt")
        console.log(
            f"{self.tag} wrote embedding to '{os.path.splitext(event.src_path)[0]}.pt'"
        )


def monitor_folder(args):
    event_handler = UploadHandler()
    observer = Observer()
    observer.schedule(event_handler, args.folder, recursive=False)
    observer.start()
    try:
        with console.status(f"monitoring '{args.folder}'"):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="watchdog for generating embeddings from uploaded recordings"
    )
    parser.add_argument(
        "-i",
        "--folder",
        type=str,
        default="data/outputs/uploads",
        help="path to monitor for changes",
    )
    args = parser.parse_args()

    monitor_folder(args)
