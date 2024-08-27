import os
import numpy as np

from utils import console
from .worker import Worker


class Seeker(Worker):
    def __init__(self, params, table_path: str, dataset_path: str):
        self.tag = params.tag
        self.params = params
        self.table_path = table_path
        self.dataset_path = dataset_path
        self.rng = np.random.default_rng(self.params.seed)

        console.log(f"{self.tag} initialization complete")

    def get_random(self) -> str:
        """Select a random file from the dataset and return the path to it."""
        return os.path.join(
            self.dataset_path,
            self.rng.choice(
                [m for m in os.listdir(self.dataset_path) if m.endswith(".mid")]
            ),
        )
