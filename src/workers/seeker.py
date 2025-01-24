import os
import faiss
import numpy as np
import pandas as pd
from shutil import copy2
from mido import bpm2tempo
from pretty_midi import PrettyMIDI
from scipy.spatial.distance import cosine

from .worker import Worker
from utils import console, panther

SUPPORTED_EXTENSIONS = (".mid", ".midi")


class Seeker(Worker):
    sim_table: pd.DataFrame
    neighbor_table: pd.DataFrame
    trans_table: pd.DataFrame
    n_track_repeats: int = 0
    played_files: list[str] = []
    allow_multiple_plays = False
    transformation = {"transpose": 0, "shift": 0}
    neighbor_col_priorities = ["next", "next_2", "prev", "prev_2"]
    matches_pos = 0
    matches_mode = "cplx"
    playlist = {}
    pitch_match = False

    def __init__(
        self,
        params,
        table_path: str,
        dataset_path: str,
        playlist_path: str,
        bpm: int,
    ):
        super().__init__(params, bpm=bpm)
        self.mode = params.mode
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.p_playlist = playlist_path
        self.rng = np.random.default_rng(self.params.seed)

        if hasattr(params, "pf_recording"):
            self.pf_recording = params.pf_recording

        if params.mode == "best":
            self.metric = params.metric

        # if params.metric in model_list:
        #     self.load_model()

        # load embeddings
        pf_emb_table = os.path.join(self.p_table, f"{params.metric}.h5")
        console.log(f"{self.tag} looking for embedding table '{pf_emb_table}'")
        if os.path.isfile(pf_emb_table):
            with console.status("\t\t\t      loading embeddings file..."):
                self.emb_table = pd.read_hdf(pf_emb_table)
            console.log(
                f"{self.tag} loaded {len(self.emb_table)}*{len(self.emb_table.columns)} embeddings table"
            )
            console.log(self.emb_table.head())

            console.log("building FAISS index...")
            match self.metric:
                case "clamp":
                    self.emb_table["normed_embeddings"] = self.emb_table[
                        "embeddings"
                    ].apply(lambda x: x / np.linalg.norm(x))
                    embedding_dim = 768
                case "specdiff":
                    self.emb_table["normed_embeddings"] = self.emb_table[
                        "embeddings"
                    ].apply(lambda x: x / np.linalg.norm(x))
                    embedding_dim = 768
                case _:
                    self.emb_table["normed_embeddings"] = self.emb_table.iloc[
                        :, 0:12
                    ].apply(lambda row: row.tolist(), axis=1)
                    embedding_dim = 12

            console.log(self.emb_table[["normed_embeddings"]].head())

            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index.add(
                np.array(
                    self.emb_table["normed_embeddings"].to_list(), dtype=np.float32
                )
            )  # type: ignore
            console.log(
                f"FAISS index built ({np.array(self.emb_table["normed_embeddings"].to_list(), dtype=np.float32).shape})"
            )
        else:
            console.log(f"{self.tag} error loading embeddings table, exiting...")
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

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")
        console.log(f"{self.tag} initialization complete")

    def get_next(self) -> tuple[str, float]:
        similarity = 0.0
        match self.mode:
            case "best":
                next_file, similarity = self._get_best()
            case "easy":
                next_file = self._get_easy()
            case "playlist":
                next_file = self._read_playlist()
            case "repeat":
                next_file = self.played_files[-1]
            case "sequential":
                next_file = self._get_neighbor()
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        if self.pitch_match and len(self.played_files):
            console.log(
                f"{self.tag} pitch matching '{self.base_file(next_file)}' to '{self.played_files[-1]}'"
            )
            if self.played_files[-1].split("_")[0] == "player-recording":
                base_pch = PrettyMIDI(
                    os.path.join(self.pf_recording)
                ).get_pitch_class_histogram(True, True)
            else:
                base_pch = PrettyMIDI(
                    os.path.join(self.p_dataset, self.played_files[-1])
                ).get_pitch_class_histogram(True, True)
            best_match = {"file": None, "sim": -1}
            for pitch in range(12):
                shift = next_file.split("_")[-1][4:]  # also contains .mid
                transposed_file = f"{next_file[:-11]}_t{pitch:02d}s{shift}"
                shifted_pch = PrettyMIDI(transposed_file).get_pitch_class_histogram(
                    True, True
                )
                similarity = float(1 - cosine(shifted_pch, base_pch))
                if similarity > best_match["sim"]:
                    best_match["sim"] = similarity
                    best_match["file"] = transposed_file
                    if self.verbose:
                        console.log(f"{self.tag} improved match:\n{best_match}")
            if best_match["file"] == None and self.verbose:
                console.log(f"{self.tag} next file was already optimally pitch matched")
            else:
                next_file = best_match["file"]

        self.played_files.append(os.path.basename(next_file))

        return next_file, similarity

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        return os.path.join(self.p_dataset, random_file)

    def _get_best(self) -> tuple[str, float]:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )

        console.log(f"{self.tag} played files:\n{self.played_files}")
        # handle parsing file names with and without augmentation
        if len(os.path.basename(self.played_files[-1])[:-4].split("_")) == 3:
            track, segment, transformation = os.path.basename(self.played_files[-1])[
                :-4
            ].split("_")
            self.transformation["transpose"] = int(transformation[1:2])
            self.transformation["shift"] = int(transformation[4:5])
        else:
            track, segment = os.path.basename(self.played_files[-1])[:-4].split("_")
        current_transformation = f"t{self.transformation['transpose']:02d}s{self.transformation['shift']:02d}"
        q_key = f"{track}_{segment}_{current_transformation}"

        if self.verbose:
            console.log(
                f"{self.tag} extracted '{self.played_files[-1]}' -> '{track}' and '{segment}' and '{current_transformation}'"
            )

        if track == "player-recording":
            match self.metric:
                case "clamp":
                    console.log(
                        f"{self.tag} getting clamp embedding for '{self.pf_recording}'"
                    )
                    q_embedding = panther.calc_embedding(self.pf_recording)
                    console.log(f"{self.tag} got clamp embedding {q_embedding.shape}")
                case "specdiff":
                    console.log(
                        f"{self.tag} getting spectrogram diffusion embedding for '{self.pf_recording}'"
                    )
                    raise NotImplementedError("nah")
                case _:
                    if self.verbose:
                        console.log(f"{self.tag} defaulting to pitch histogram metric")
                    q_embedding = PrettyMIDI(
                        self.pf_recording
                    ).get_pitch_class_histogram(True, True)
                    q_embedding = q_embedding.reshape(1, -1)
                    console.log(f"{self.tag} {q_embedding}")
            if self.verbose:
                console.log(
                    f"{self.tag} calculated embedding for recording: {q_embedding.shape}"
                )
        else:
            q_embedding = np.array(
                [self.emb_table.loc[q_key, "normed_embeddings"]],
                dtype=np.float32,
            )

        console.log(f"{self.tag} querying with key '{q_key}'")
        similarities, indices = self.faiss_index.search(q_embedding, 5000)  # type: ignore
        # console.log(f"{self.tag} indices {indices[:10]}\nsimilarities {similarities[:10]}")
        # NO SHIFT
        indices, similarities = zip(
            *[
                (i, s)
                for i, s in zip(indices[0], similarities[0])
                if str(self.emb_table.index[i]).endswith("s00")
            ]
        )
        nearest_neighbors = {}
        for i, s in zip(indices, similarities):
            nearest_neighbors[str(self.emb_table.index[i])] = float(s)

        if self.verbose:
            console.log(
                f"{self.tag} got nearest neighbors to '{q_key}':",
                sorted(
                    nearest_neighbors.items(), key=lambda item: item[1], reverse=True
                )[:10],
            )

        next_file = self._get_neighbor()
        console.log(f"{self.tag} 'random' file is '{next_file}'")
        played_files = [os.path.basename(self.base_file(f)) for f in self.played_files]
        for i_neighbor, similarity in zip(indices, similarities):
            segment_name = str(self.emb_table.index[i_neighbor])
            if track == "player-recording":
                next_file = f"{segment_name}.mid"
                break
            # check if file has already been played
            if segment_name in played_files:
                continue

            next_segment_name = self.base_file(segment_name)
            next_track = next_segment_name.split("_")[0]
            last_track = self.played_files[-1].split("_")[0]
            played_tracks = [file.split("_")[0] for file in self.played_files]
            # switch to different track after 2 segments
            # if len(self.played_files) % 2 == 0:
            if next_track in played_tracks:
                console.log(
                    f"{self.tag} transitioning to next track and skipping '{next_segment_name}'"
                )
                continue
            else:
                next_file = f"{segment_name}.mid"
                break
            # NO SHIFT
            if (
                next_segment_name not in played_files
                and segment_name.endswith("s00")
                and next_track == last_track
            ):
                next_file = f"{segment_name}.mid"
                break

        console.log(
            f"{self.tag} best match is '{next_file}' with similarity {similarity:.05f}"
        )
        return os.path.join(self.p_dataset, next_file), similarity

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

        # transforms not needed
        if len(current_file.split("_")) > 2:
            track, segment, _ = current_file.split("_")
            current_file = f"{track}_{segment}.mid"

        try:
            for col_name in self.neighbor_col_priorities:
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
        except KeyError:
            console.log(
                f"{self.tag} unable to find neighbor for '{current_file}', choosing randomly"
            )

            return self._get_random()
        return self._get_random()

    def _get_random(self) -> str:
        console.log(
            f"{self.tag} choosing randomly from '{self.p_dataset}':\n{[m for m in os.listdir(self.p_dataset)][:5]}"
        )
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(SUPPORTED_EXTENSIONS)],
            1,
        )[0]

        # correct for missing augmentation information
        if "_t" not in random_file:
            random_file = random_file[:-4] + "_t00s00.mid"

        if self.verbose:
            console.log(f"{self.tag} chose random file '{random_file}'")

        # only play files once
        if not self.allow_multiple_plays:
            base_file = self.base_file(random_file)
            while base_file in self.played_files:
                base_file = self.base_file(random_file)
                random_file = self.rng.choice(
                    [
                        m
                        for m in os.listdir(self.p_dataset)
                        if m.endswith(SUPPORTED_EXTENSIONS)
                    ]
                )

        # NO TRANSFORMS
        random_file = os.path.splitext(random_file)[0][:-6] + "t00s00.mid"

        return str(random_file)

    def _read_playlist(self) -> str:
        if self.verbose:
            console.log(
                f"{self.tag} playing matches for '{[self.playlist.keys()][self.matches_pos]}'"
            )
            console.log(f"{self.tag} already played files:\n{self.played_files}")

        for i, (q, ms) in enumerate(self.playlist.items()):
            if i == self.matches_pos:
                console.log(f"{self.tag} [grey30]{i}\t'{q}'")
                for mode, matches in ms.items():
                    console.log(f"{self.tag} [grey30]\t'{mode}'")
                    if mode == self.matches_mode:
                        for f, s in matches[:5]:
                            base_files = [
                                os.path.basename(self.base_file(f))
                                for f in self.played_files
                            ]
                            # console.log(
                            #     f"{self.tag} looking for '{self.base_file(f)}' in {base_files}"
                            # )
                            if self.base_file(f) in base_files:
                                console.log(f"{self.tag} [grey30]\t\t'{f}'\t{s}")
                            else:
                                console.log(f"{self.tag} [grey70]\t\t'{f}'\t{s}")
                                return os.path.join(self.p_dataset, f"{f}.mid")
                        if self.matches_mode == "cplx":
                            raise EOFError("playlist complete")
                        console.log(f"{self.tag} switching modes")
                        self.matches_mode = "cplx"
                        return os.path.join(self.p_dataset, f"{q}.mid")
        return ""

    def transform(self, midi_file: str = "") -> str:
        pf_in = self.base_file(self.played_files[-1]) if midi_file == "" else midi_file
        pf_out = os.path.join(
            self.p_playlist,
            f"{len(self.played_files):02d} {os.path.basename(pf_in)}",
        )
        pf_out = os.path.join(self.p_playlist, f"{len(self.played_files):02d} {pf_in}")
        console.log(f"{self.tag} transforming '{pf_in}' to '{pf_out}'")

        copy2(
            os.path.join(os.path.dirname(self.p_dataset), pf_in),
            pf_out,
        )

        return pf_out

    def base_file(self, filename: str) -> str:
        pieces = os.path.basename(filename).split("_")
        # return f"{pieces[0]}_{pieces[1]}_{pieces[2][:-4]}.mid"
        return f"{pieces[0]}_{pieces[1]}.mid"

    def construct_keys(self):
        for filename in self.played_files:
            yield f"files:{filename[:-4]}_t00s00.mid"

    def load_model(self):
        match self.metric:
            case "clamp":
                raise NotImplementedError("CLaMP model is no longer supported")
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    console.log(f"{self.tag} Using GPU {torch.cuda.get_device_name(0)}")
                else:
                    console.log(f"{self.tag} No GPU available, using the CPU instead")
                    device = torch.device("cpu")
                self.model = clamp.CLaMP.from_pretrained(clamp.CLAMP_MODEL_NAME)
                if self.verbose:
                    console.log(f"{self.tag} Loaded model:\n{self.model.eval}")
                self.model = self.model.to(device)  # type: ignore
            case _:
                raise TypeError(
                    f"{self.tag} Unsupported model specified: {self.metric}"
                )
