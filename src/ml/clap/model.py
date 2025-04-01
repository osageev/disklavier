import laion_clap
from torch import Tensor


class Clap:
    def __init__(self):
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(
            "src/ml/clap/checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"
        )

    def embed(self, audio_file: str) -> Tensor:
        return (
            self.model.get_audio_embedding_from_filelist(
                x=[audio_file], use_tensor=True
            )
            .detach()
            .cpu()
        )
