import os
import sys
import time
import torch
from rich.console import Console
from argparse import ArgumentParser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from ml.clamp.model import Clamp

console = Console(log_time_format="%m-%d %H:%M:%S.%f")


class UploadHandler(FileSystemEventHandler):
    tag = "[#5f00af]panthr[/#5f00af]:"
    def __init__(self):
        console.log(f"{self.tag} initializing model")
        self.model = Clamp()
        console.log(f"{self.tag}[green] initialization complete")

    def on_created(self, event):
        if str(event.src_path).endswith('.mid'):
            time.sleep(1)
            console.log(f"{self.tag} sending file '{event.src_path}' to be embedded by {self.model.name}")
            embedding = self.model.embed([str(event.src_path)])
            console.log(f"{self.tag} got embedding {embedding.shape}")
            torch.save(embedding, f'{event.src_path[:-4]}.pt')
            console.log(f"{self.tag} wrote embedding to '{event.src_path[:-4]}.pt'")
            

def monitor_folder(folder_path):
    event_handler = UploadHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    try:
        with console.status(f"monitoring '{folder_path}'"):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    
if __name__=="__main__":
    parser = ArgumentParser(description="watchdog for generating embeddings from uploaded recordings")
    parser.add_argument("--folder", type=str, default=None, help="path to monitor for changes")
    args = parser.parse_args()
    
    monitor_folder(args.folder)
