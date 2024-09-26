import os
import numpy as np
import pandas as pd
from shutil import copy2
from pretty_midi import PrettyMIDI
from mido import bpm2tempo
import redis
from redis.commands.search.query import Query
from scipy.spatial.distance import cosine

from .worker import Worker
from utils import console
from models import model_list, clamp

NEIGHBOR_COL_PRIORITIES = ["next", "next_2", "prev", "prev_2"]


class Seeker(Worker):
    sim_table: pd.DataFrame
    neighbor_table: pd.DataFrame
    trans_table: pd.DataFrame
    n_track_repeats: int = 0
    played_files: list[str] = []
    allow_multiple_plays = False
    transformation = {"transpose": 0, "shift": 0}

    def __init__(
        self,
        params,
        table_path: str,
        dataset_path: str,
        playlist_path: str,
        bpm: int,
        verbose: bool = False,
    ):
        super().__init__(params, verbose=verbose)
        self.mode = params.mode
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.p_playlist = playlist_path
        self.bpm = bpm
        self.tempo = bpm2tempo(self.bpm)
        self.rng = np.random.default_rng(self.params.seed)
        self.verbose = verbose

        if params.mode == "best":
            self.metric = params.metric

        if params.metric in model_list:
            self.load_model()

        if params.redis:
            self.redis_client = redis.Redis(
                host=params.redis.host,
                port=params.redis.port,
                db=params.redis.db,
                decode_responses=True,
            )

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
            case "best":
                next_file = self._get_best()
            case "easy":
                next_file = self._get_easy()
            case "sequential":
                next_file = self._get_neighbor()
            case "repeat":
                next_file = self.played_files[-1]
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        self.played_files.append(self.base_file(next_file))

        return next_file

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        self.played_files.append(random_file)

        return os.path.join(self.p_dataset, random_file)

    def _get_best(self) -> str:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )
            if self.redis_client:
                console.log(f"{self.tag} using redis db")

        console.log(f"{self.tag} {self.played_files}")
        if len(os.path.basename(self.played_files[-1])[:-4].split("_")) == 3:
            track, segment, transformation = os.path.basename(self.played_files[-1])[
                :-4
            ].split("_")
            self.transformation["transpose"] = int(transformation[1:2])
            self.transformation["shift"] = int(transformation[4:5])
        else:
            track, segment = os.path.basename(self.played_files[-1])[:-4].split("_")
        current_transformation = f"t{self.transformation['transpose']:02d}s{self.transformation['shift']:02d}"

        console.log(
            f"{self.tag} extracted '{self.played_files[-1]}' -> '{track}' and '{segment}' and '{current_transformation}'"
        )

        if track == "player_recording":
            match self.metric:
                case "clamp":
                    console.log(
                        f"{self.tag} got clamp embedding from {self.played_files[-1]}"
                    )
                    query_embedding = clamp.get_embedding(
                        self.model, self.played_files[-1]
                    )[0]
                    console.log(
                        f"{self.tag} got clamp embedding {query_embedding.shape}"
                    )
                case _:
                    if self.verbose:
                        console.log(f"{self.tag} defaulting to pitch histogram metric")
                    query_embedding = PrettyMIDI(
                        self.played_files[-1]
                    ).get_pitch_class_histogram(
                        use_duration=True, use_velocity=True, normalize=True
                    )
            if self.verbose:
                console.log(
                    f"{self.tag} calculated embedding for recording: {query_embedding.shape}"
                )
        else:
            q_k = f"files:{track}_{segment}_{current_transformation}"
            console.log(f"{self.tag} querying with key '{q_k}'")
            query_embedding = self.redis_client.json().get(q_k, f"$.{self.metric}")[0]
            if self.verbose:
                console.log(
                    f"{self.tag} got embedding for '{q_k}': {len(query_embedding)}"
                )
        played_keys = [k.split(":")[-1] for k in self.construct_keys()]
        played_files = "|".join(played_keys)
        q = f"(-@files:{{{played_files}}})=>[KNN 10 @{self.metric} $query_vector AS vector_score]" 
        console.log(f"{self.tag} trying query:\n'{q}'")
        nearest_neighbors = (
            self.redis_client.ft(f"idx:files_{self.metric}_vss")
            .search(
                Query(q)
                .sort_by("vector_score")
                .return_fields("vector_score", "id")
                .dialect(4),
                {"query_vector": np.array(query_embedding, dtype=np.float32).tobytes()},
            )
            .docs  # type: ignore
        )

        console.log(f"{self.tag} got nearest neighbors:", nearest_neighbors)

        next_file = self._get_random()
        for neighbor in nearest_neighbors:
            filename = neighbor["id"].split(":")[-1]
            # if not self.allow_multiple_plays:
            if self.base_file(filename) not in self.played_files:
                next_file = f"{filename}.mid"
                break

        best_shift = self.match_pitch(next_file, current_transformation)
        console.log(
            f"{self.tag} best shift for '{os.path.basename(next_file)}' is actually {best_shift}"
        )

        return os.path.join(os.path.dirname(self.p_dataset), "train", next_file)

    def _get_easy(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} played files: {self.played_files}")
            console.log(f"{self.tag} num_repeats: {self.n_track_repeats}")
        if self.n_track_repeats < 8:
            console.log(f"{self.tag} transitioning to next segment")
            self.n_track_repeats += 1
            return self._get_neighbor()
        else:
            console.log(f"{self.tag} transitioning to next track")
            self.n_track_repeats = 0
            return self._get_random()

    def _get_neighbor(self) -> str:
        current_file = os.path.basename(self.played_files[-1])

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
                return str(neighbor)

        console.log(
            f"{self.tag} unable to find neighbor for '{current_file}', choosing randomly"
        )

        return self._get_random()

    def _get_random(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} choosing random file")
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
        )

        # only play files once
        if not self.allow_multiple_plays:
            base_file = self.base_file(random_file)
            while base_file in self.played_files:
                base_file = self.base_file(random_file)
                random_file = self.rng.choice(
                    [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
                )

        return str(random_file)

    def transform(self, midi_file: str = "") -> str:
        pf_in = self.played_files[-1] if midi_file == "" else midi_file
        pf_out = os.path.join(
            self.p_playlist,
            f"{len(self.played_files):02d} {os.path.basename(pf_in)}",
        )
        pf_out = os.path.join(self.p_playlist, f"{len(self.played_files):02d} {pf_in}")
        console.log(f"{self.tag} transforming '{pf_in}' to '{pf_out}'")

        # return transform(pf_in, pf_out, self.bpm, self.transformation)
        copy2(
            os.path.join(os.path.dirname(self.p_dataset), "train", pf_in),
            pf_out,
        )

        return pf_out

    def base_file(self, filename: str) -> str:
        pieces = os.path.basename(filename).split("_")
        return f"{pieces[0]}_{pieces[1]}.mid"

    def construct_keys(self):
        for filename in self.played_files:
            yield f"files:{filename[:-4]}_t00s00.mid"

    def load_model(self):
        match self.metric:
            case "clamp":
                import torch

                # init torch device
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    console.log(f"using GPU {torch.cuda.get_device_name(0)}")

                else:
                    console.log(f"No GPU available, using the CPU instead.")
                    device = torch.device("cpu")
                self.model = clamp.CLaMP.from_pretrained(clamp.CLAMP_MODEL_NAME)
                if self.verbose:
                    console.log(f"{self.tag} loaded model:\n{self.model.eval}")
                self.model = self.model.to(device)
            case _:
                raise TypeError(f"Unsupported model specified: {self.metric}")

    def match_pitch(self, midi_file: str, transform: str) -> int:

        original_midi = PrettyMIDI(
            os.path.join(
                os.path.dirname(self.p_dataset),
                "train",
                f"{self.played_files[-1][:-4]}_{transform}.mid",
            )
        )
        original_pitch_class_histogram = original_midi.get_pitch_class_histogram(
            use_duration=True, use_velocity=True, normalize=True
        )

        best_shift = 0
        best_similarity = float("-inf")

        for shift in range(-12, 13):
            shifted_midi = PrettyMIDI(
                os.path.join(os.path.dirname(self.p_dataset), "train", midi_file)
            )
            for instrument in shifted_midi.instruments:
                for note in instrument.notes:
                    note.pitch += shift
            shifted_pitch_class_histogram = shifted_midi.get_pitch_class_histogram(
                use_duration=True, use_velocity=True, normalize=True
            )

            similarity = 1 - cosine(
                original_pitch_class_histogram, shifted_pitch_class_histogram
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_shift = shift

        return best_shift
