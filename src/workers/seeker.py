import os
import numpy as np
import pandas as pd

from utils import console
from .worker import Worker

NEIGHBOR_COL_PRIORITIES = ["next", "next_2", "prev", "prev_2"]


class Seeker(Worker):
    sim_table: pd.DataFrame
    neighbor_table: pd.DataFrame
    trans_table: pd.DataFrame
    played_files: list[str] = []
    allow_multiple_plays = False
    transition_probability = 0.0
    transformation = {"transpose": 0, "shift": 0}

    def __init__(
        self, params, table_path: str, dataset_path: str, verbose: bool = False
    ):
        # load state
        self.tag = params.tag
        self.params = params
        self.mode = params.mode
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.rng = np.random.default_rng(self.params.seed)
        self.verbose = verbose

        # load similarity table
        pf_sim_table = os.path.join(self.p_table, "sim.parquet")
        console.log(f"{self.tag} looking for similarity table '{pf_sim_table}'")
        if os.path.isfile(pf_sim_table):
            with console.status("\t\t\t      loading similarities file..."):
                self.sim_table = pd.read_parquet(pf_sim_table)
            console.log(
                f"{self.tag} loaded {len(self.sim_table)}*{len(self.sim_table.columns)} sim table"
            )
            console.log(self.sim_table.head())
        else:
            console.log(f"{self.tag} error loading similarity table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        # load neighbor table
        pf_neighbor_table = os.path.join(self.p_table, "neighbor.parquet")
        console.log(f"{self.tag} looking for neighbor table '{pf_neighbor_table}'")
        if os.path.isfile(pf_neighbor_table):
            with console.status("\t\t\t      loading neighbor file..."):
                self.neighbor_table = pd.read_parquet(pf_neighbor_table)
            console.log(
                f"{self.tag} loaded {len(self.neighbor_table)}*{len(self.neighbor_table.columns)} neighbor table"
            )
            console.log(self.neighbor_table.head())
        else:
            console.log(f"{self.tag} error loading neighbor table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        # load transformation table
        pf_trans_table = os.path.join(self.p_table, "transformations.parquet")
        console.log(f"{self.tag} looking for tranformation table '{pf_trans_table}'")
        if os.path.isfile(pf_trans_table):
            with console.status("\t\t\t      loading tranformation file..."):
                self.trans_table = pd.read_parquet(pf_trans_table)
            console.log(
                f"{self.tag} loaded {len(self.trans_table)}*{len(self.trans_table.columns)} transformation table"
            )
            console.log(self.trans_table.head())
        else:
            console.log(f"{self.tag} error loading tranformation table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        console.log(f"{self.tag} [green]successfully loaded tables")
        console.log(f"{self.tag} initialization complete")

    def get_next(self) -> str:
        match self.mode:
            case "sequential":
                next_file = self._get_neighbor(self.played_files[-1])
            case "repeat":
                next_file = self.played_files[0]
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        self.played_files.append(next_file)

        return os.path.join(self.p_dataset, next_file)

    def _get_neighbor(self, current_file_path: str) -> str:
        current_file = os.path.basename(current_file_path)

        for col_name in NEIGHBOR_COL_PRIORITIES:
            neighbor = self.neighbor_table.loc[current_file, col_name]

            # only play files once
            if neighbor in self.played_files and not self.allow_multiple_plays:
                neighbor = None

            # found a neighbor
            if neighbor != None:
                if self.verbose:
                    console.log(
                        f"{self.tag} found neighboring file '{neighbor}' at position '{col_name}'"
                    )
                return f"{neighbor}"

        console.log(
            f"{self.tag} unable to find neighbor for '{current_file}', choosing randomly"
        )

        return self._get_random()

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        self.played_files.append(random_file)

        return os.path.join(self.p_dataset, random_file)

    def _get_random(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} choosing random file")
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
        )

        # only play files once
        if not self.allow_multiple_plays:
            while random_file in self.played_files:
                random_file = self.rng.choice(
                    [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
                )

        return random_file
