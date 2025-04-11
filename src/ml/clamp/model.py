import os  ## conda activate oodenv
import torch
import argparse
import requests
import numpy as np
from tqdm import tqdm
from transformers import BertConfig, AutoTokenizer

from ml.clamp.config import *
from ml.clamp.clamp_utils import *
from utils import console


class Clamp3:
    tag = "[#a3d2ca]clamp3[/#a3d2ca]:"
    name = "CLaMP3"

    def __init__(self, device: str, epoch: int = 512):
        print(f"{self.tag} initializing {self.name} model")

        self.device = device
        print(f"{self.tag} using device: {self.device}")

        # Model and configuration setup
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=MAX_AUDIO_LENGTH,
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE // 64,
            intermediate_size=M3_HIDDEN_SIZE * 4,
            max_position_embeddings=PATCH_LENGTH,
        )
        self.model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3,
        )
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.patchilizer = M3Patchilizer()

        # print parameter number
        print(
            "Total Parameter Number: "
            + str(sum(p.numel() for p in self.model.parameters()))
        )

        # Load model weights
        self.model.eval()

        checkpoint_path = "/home/finlay/disklavier/data/models/" + CLAMP3_WEIGHTS_PATH

        if epoch is not None:
            checkpoint_path = checkpoint_path.replace(".pth", f"_{epoch}.pth")

        if not os.path.exists(checkpoint_path):
            print("No CLaMP 3 weights found. Downloading from Hugging Face...")
            checkpoint_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
            response = requests.get(checkpoint_url, stream=True)

            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(checkpoint_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Weights file downloaded successfully.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(
            f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}"
        )
        self.model.load_state_dict(checkpoint["model"])
        print(f"{self.tag} model loaded successfully")

    def embed(self, filename: str, get_global: bool = True) -> torch.Tensor:
        if filename.endswith(".mid"):
            filename = convert_midi2mtf(filename, "data")

        with open(filename, "r", encoding="utf-8") as f:
            item = f.read()

        if filename.endswith(".mtf"):
            input_data = self.patchilizer.encode(item, add_special_patches=True)
            input_data = torch.tensor(input_data)
        else:
            raise ValueError("Invalid file extension (must be .mtf or .mid)")

        segment_list = []
        for i in range(0, len(input_data), PATCH_LENGTH):
            segment_list.append(input_data[i : i + PATCH_LENGTH])
        segment_list[-1] = input_data[-PATCH_LENGTH:]

        last_hidden_states_list = []
        for input_segment in segment_list:
            input_masks = torch.tensor([1] * input_segment.size(0))
            pad_indices = (
                torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long()
                * self.patchilizer.pad_token_id
            )
            input_masks = torch.cat(
                (input_masks, torch.zeros(PATCH_LENGTH - input_segment.size(0))), 0
            )
            input_segment = torch.cat((input_segment, pad_indices), 0)
            last_hidden_states = self.model.get_symbolic_features(
                symbolic_inputs=input_segment.unsqueeze(0).to(self.device),
                symbolic_masks=input_masks.unsqueeze(0).to(self.device),
                get_global=get_global,
            )
            if not get_global:
                last_hidden_states = last_hidden_states[
                    :, : input_masks.sum().long().item(), :
                ]
            last_hidden_states_list.append(last_hidden_states)

        if not get_global:
            last_hidden_states_list = [
                last_hidden_states[0] for last_hidden_states in last_hidden_states_list
            ]
            last_hidden_states_list[-1] = last_hidden_states_list[-1][
                -(len(input_data) % PATCH_LENGTH) :
            ]
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        else:
            full_chunk_cnt = len(input_data) // PATCH_LENGTH
            remain_chunk_len = len(input_data) % PATCH_LENGTH
            if remain_chunk_len == 0:
                feature_weights = torch.tensor(
                    [PATCH_LENGTH] * full_chunk_cnt, device=self.device
                ).view(-1, 1)
            else:
                feature_weights = torch.tensor(
                    [PATCH_LENGTH] * full_chunk_cnt + [remain_chunk_len],
                    device=self.device,
                ).view(-1, 1)

            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
            last_hidden_states_list = last_hidden_states_list * feature_weights
            last_hidden_states_list = (
                last_hidden_states_list.sum(dim=0) / feature_weights.sum()
            )
        return last_hidden_states_list
