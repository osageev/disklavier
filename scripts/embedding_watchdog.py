import os
import sys
import time
import json
import torch
from rich.console import Console
from argparse import ArgumentParser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from ml.clamp.model import Clamp
from models import model_list
from models.specdiff import SpectrogramDiffusion

console = Console(log_time_format="%m-%d %H:%M:%S.%f")
SUPPORTED_EXTENSIONS = (".mid", ".midi")

class UploadHandler(FileSystemEventHandler):
    tag = "[#5f00af]panthr[/#5f00af]:"

    def __init__(self, model_name: str):
        # load config
        pf_model_config = os.path.join("models", "configs", model_name, "config.json")
        with open(pf_model_config, "r") as f:
            config = json.load(f)
        console.log(f"{self.tag} initializing model with config:\n", config)

        # load model
        match model_name:
            case "clamp":
                self.model = Clamp()
            case "specdiff":
                self.model = SpectrogramDiffusion(config)
            case _:
                raise ModuleNotFoundError(
                    f"Unsupported model specified: {args.model}\nOnly the following models are currently supported: {model_list}"
                )
        console.log(f"{self.tag}[green] initialization complete")

    def on_created(self, event):
        if str(event.src_path).endswith(SUPPORTED_EXTENSIONS):
            time.sleep(1)
            console.log(
                f"{self.tag} sending file '{event.src_path}' to be embedded by {self.model.name}"
            )
            embedding = self.model.embed(str(event.src_path))
            console.log(f"{self.tag} got embedding {embedding.shape}")
            torch.save(embedding, f"{os.path.splitext(event.src_path)[0]}.pt")
            console.log(f"{self.tag} wrote embedding to '{os.path.splitext(event.src_path)[0]}.pt'")
        else:
            raise TypeError("only the following filetypes are supported:", SUPPORTED_EXTENSIONS)


def monitor_folder(args):
    event_handler = UploadHandler(args.model)
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
        "-i", "--folder", type=str, default=None, help="path to monitor for changes"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="specdiff",
        help="model to use for embedding generation",
    )
    args = parser.parse_args()

    monitor_folder(args)
