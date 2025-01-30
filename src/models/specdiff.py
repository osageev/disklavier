import torch
import numpy as np
from rich.console import Console
from diffusers import MidiProcessor
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)

console = Console(log_time_format="%m-%d %H:%M:%S.%f")


class SpectrogramDiffusion:
    tag = "[orange]spcdif[/orange]:"
    name = "SpectrogramDiffusion"

    def __init__(self, config) -> None:
        console.log(f"{self.tag} initializing spectrogram diffusion model")

        torch.set_grad_enabled(False)
        self.processor = MidiProcessor()
        self.encoder = SpectrogramNotesEncoder(**config.encoder_config).cuda(
            device=config.device
        )
        self.encoder.eval()
        sd = torch.load(config.encoder_weights_path, weights_only=True)
        self.encoder.load_state_dict(sd)

        console.log(f"{self.tag} model initialization complete")

    def embed(self, path: str) -> np.ndarray:
        console.log(f"{self.tag} generating embedding for '{path}'")
        console.log(f"{self.tag} tokenizing")
        tokens = self.processor(path)
        console.log(f"{self.tag} generated {len(tokens)} tokens:", tokens)

        console.log(f"{self.tag} embedding")
        with torch.autocast("cuda"):
            tokens_mask = tokens > 0
            tokens_embedded, tokens_mask = self.encoder(
                encoder_input_tokens=tokens, encoder_inputs_mask=tokens_mask
            )
        console.log(f"{self.tag} generated embedding ({tokens_embedded.shape})")
        avg_embedding = tokens_embedded[tokens_mask].mean(0)
        console.log(f"{self.tag} embedding complete")

        return avg_embedding / np.linalg.norm(avg_embedding, keepdims=True)
