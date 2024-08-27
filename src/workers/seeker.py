import os
import numpy as np
import pandas as pd

from utils import console
from .worker import Worker


class Seeker(Worker):
    sim_table: pd.DataFrame
    neighbor_table: pd.DataFrame
    trans_table: pd.DataFrame
    played_files: list[str] = []
    transition_probability = 0.0
    transformation = {"transpose": 0, "shift": 0}

    def __init__(self, params, table_path: str, dataset_path: str):
        # load state
        self.tag = params.tag
        self.params = params
        self.table_path = table_path
        self.dataset_path = dataset_path
        self.rng = np.random.default_rng(self.params.seed)

        # load similarity table
        sim_table_path = os.path.join(self.table_path, "sim.parquet")
        console.log(f"{self.tag} looking for similarity table '{sim_table_path}'")
        if os.path.isfile(sim_table_path):
            with console.status("\t\t\t      loading similarities file..."):
                self.sim_table = pd.read_parquet(sim_table_path)
            console.log(
                f"{self.tag} loaded {len(self.sim_table)}*{len(self.sim_table.columns)} sim table"
            )
            console.log(self.sim_table.head())
        else:
            console.log(f"{self.tag} error loading similarity table, exiting...")
            exit() # TODO: handle this better (return an error, let main handle it)

        # load neighbor table
        neighbor_table_path = os.path.join(self.table_path, "neighbor.parquet")
        console.log(f"{self.tag} looking for neighbor table '{neighbor_table_path}'")
        if os.path.isfile(neighbor_table_path):
            with console.status("\t\t\t      loading neighbor file..."):
                self.neighbor_table = pd.read_parquet(neighbor_table_path)
            console.log(
                f"{self.tag} loaded {len(self.neighbor_table)}*{len(self.neighbor_table.columns)} neighbor table"
            )
            console.log(self.neighbor_table.head())
        else:
            console.log(f"{self.tag} error loading neighbor table, exiting...")
            exit() # TODO: handle this better (return an error, let main handle it)

        # load transformation table
        trans_table_path = os.path.join(self.table_path, "transformations.parquet")
        console.log(f"{self.tag} looking for tranformation table '{trans_table_path}'")
        if os.path.isfile(trans_table_path):
            with console.status("\t\t\t      loading tranformation file..."):
                self.trans_table = pd.read_parquet(trans_table_path)
            console.log(
                f"{self.tag} loaded {len(self.trans_table)}*{len(self.trans_table.columns)} transformation table"
            )
            console.log(self.trans_table.head())
        else:
            console.log(f"{self.tag} error loading tranformation table, exiting...")
            exit() # TODO: handle this better (return an error, let main handle it)
        
        console.log(f"{self.tag} [green]successfully loaded tables")
        console.log(f"{self.tag} initialization complete")

    def get_neighbor(self, current_file_path: str) -> str:
        return ""

    def get_random(self) -> str:
        """Select a random file from the dataset and return the path to it."""
        return os.path.join(
            self.dataset_path,
            self.rng.choice(
                [m for m in os.listdir(self.dataset_path) if m.endswith(".mid")]
            ),
        )
