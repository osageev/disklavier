import os
import torch
import numpy as np
from diffusers import MidiProcessor
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)

