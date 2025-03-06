import os
import h5py
import faiss
import numpy as np
import pandas as pd
from shutil import copy2
from pretty_midi import PrettyMIDI
from scipy.spatial.distance import cosine
import json
import networkx as nx
import matplotlib.pyplot as plt

from .worker import Worker
from utils import console, panther

SUPPORTED_EXTENSIONS = (".mid", ".midi")
EMBEDDING_SIZES = {
    "pitch-histogram": 12,
    "specdiff": 768,
    "clf-4note": 128,
    "clf-speed": 128,
    "clf-tpose": 128,
}


class Seeker(Worker):
    # required tables
    filenames: list[str] = []
    faiss_index: faiss.IndexFlatIP
    neighbor_table: pd.DataFrame
    neighbor_col_priorities = ["next", "next_2", "prev", "prev_2"]
    # forced track change interval
    n_transition_interval: int = 8  # 16 bars since segments are 2 bars
    # number of repeats for easy mode
    n_segment_repeats: int = 0
    # playback trackers
    played_files: list[str] = []
    allow_multiple_plays = False
    # transformation tracker
    transformation = {"transpose": 0, "shift": 0}
    # playlist mode position tracker
    matches_pos = 0
    # TODO: i forget what this does tbh, rewrite entire playlist mode
    matches_mode = "cplx"
    playlist = {}
    # force pitch match after finding next segment
    pitch_match = False
    # supported ml models for post-processing embeddings
    model_list = [
        "clf_4note",
        "clf_speed",
        "clf_tpose",
    ]
    # path tracking variables
    current_path = []
    current_path_position = 0

    def __init__(
        self,
        params,
        table_path: str,
        dataset_path: str,
        playlist_path: str,
        bpm: int,
    ):
        super().__init__(params, bpm=bpm)
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.p_playlist = playlist_path
        self.rng = np.random.default_rng(self.params.seed)
        # path tracking variables
        self.current_path = []
        self.current_path_position = 0

        if hasattr(self.params, "pf_recording"):
            self.pf_recording = self.params.pf_recording
        if self.params.metric in self.model_list:
            self.model = self.load_model()

        # load embeddings and FAISS index
        pf_emb_table = os.path.join(self.p_table, f"{self.params.metric}.h5")
        console.log(f"{self.tag} looking for embedding table '{pf_emb_table}'")
        if os.path.isfile(pf_emb_table):
            # load embeddings table from h5 to pd df
            with console.status("\t\t\t      loading embeddings..."):
                emb_column_name = (
                    "histograms"
                    if self.params.metric == "pitch-histogram"
                    else "embeddings"
                )
                with h5py.File(pf_emb_table, "r") as f:
                    # Convert h5py datasets to numpy arrays first to make them iterable
                    emb_array = np.array(f[emb_column_name])
                    filenames_array = np.array(f["filenames"])

                    self.emb_table = pd.DataFrame(
                        list(
                            [e, e / np.linalg.norm(e, keepdims=True)] for e in emb_array
                        ),
                        index=[str(name[0], "utf-8") for name in filenames_array],
                        columns=["embeddings", "normed_embeddings"],
                    )
            console.log(
                f"{self.tag} loaded {len(self.emb_table)}*{len(self.emb_table.columns)} embeddings table"
            )
            # self.emb_table["normed_embeddings"] = self.emb_table[
            #     "embeddings"
            # ] / np.linalg.norm(self.emb_table["embeddings"], axis=1, keepdims=True)

            if self.verbose:
                console.log(self.emb_table.head())
                console.log(f"{self.tag} normalized embeddings")
                console.log(self.emb_table["normed_embeddings"].head())
                # console.log(
                #     [
                #         np.linalg.norm(e)
                #         for e in self.emb_table["normed_embeddings"].sample(
                #             10, random_state=self.params.seed
                #         )
                #         if np.linalg.norm(e) != 1
                #     ]
                # )

        # build FAISS index
        console.log(f"{self.tag} building FAISS index...")
        self.filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.p_dataset)
            if f.endswith(SUPPORTED_EXTENSIONS)
        ]
        console.log(
            f"{self.tag} found {len(self.filenames)} files in '{self.p_dataset}'\n{self.filenames[:5]}"
        )
        self.faiss_index = faiss.read_index(
            os.path.join(self.p_table, f"{self.params.metric}.index")
        )
        # eventually will probably have to replace this with a MATCH CASE statement
        # self.faiss_index = faiss.IndexFlatIP(EMBEDDING_SIZES[self.params.metric])
        # self.faiss_index.add(
        #     np.array(
        #         self.emb_table["normed_embeddings"].to_list(), dtype=np.float32
        #     )
        # )  # type: ignore
        console.log(f"{self.tag} FAISS index built ({self.faiss_index.ntotal})")
        # else:
        #     console.log(f"{self.tag} error loading embeddings table, exiting...")
        #     exit()  # TODO: handle this better (return an error, let main handle it)

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
        match self.params.mode:
            case "best":
                next_file, similarity = self._get_best(hop=False)
            case "timed_hops":
                next_file, similarity = self._get_best(hop=True)
            case "easy":
                next_file = self._get_easy()
            case "playlist":
                next_file = self._read_playlist()
            case "repeat":
                next_file = self.played_files[-1]
            case "sequential":
                next_file = self._get_neighbor()
            case "graph":
                next_file = self._get_graph()
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        if self.pitch_match and len(self.played_files) > 0:
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

    def _get_best(self, hop: bool = False) -> tuple[str, float]:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )
            console.log(
                f"{self.tag} {len(self.played_files)} played files:\n{self.played_files}"
            )

        # handle parsing file names with and without augmentation
        # TODO: remove this and global transformation tracking until it is relevant again
        if (
            len(os.path.splitext(os.path.basename(self.played_files[-1]))[0].split("_"))
            == 3
        ):
            track, segment, transformation = os.path.basename(self.played_files[-1])[
                :-4
            ].split("_")
            self.transformation["transpose"] = int(transformation[1:3])
            self.transformation["shift"] = int(transformation[4:6])
        else:
            track, segment = os.path.basename(self.played_files[-1])[:-4].split("_")
            console.log(
                f"{self.tag} len {len(os.path.splitext(os.path.basename(self.played_files[-1]))[0].split("_"))} from {os.path.splitext(os.path.basename(self.played_files[-1]))[0].split("_")}"
            )
        current_transformation = f"t{self.transformation['transpose']:02d}s{self.transformation['shift']:02d}"
        query_file = f"{track}_{segment}_{current_transformation}"

        if self.verbose:
            console.log(
                f"{self.tag} extracted '{self.played_files[-1]}' -> '{track}' and '{segment}' and '{current_transformation}'"
            )

        # load query embedding
        if track == "player-recording":
            match self.params.metric:
                case "clamp" | "specdiff":
                    console.log(
                        f"{self.tag} getting [bold]{self.params.metric}[/bold] embedding for '{self.pf_recording}'"
                    )
                    query_embedding = panther.calc_embedding(self.pf_recording)
                    console.log(
                        f"{self.tag} got [bold]{self.params.metric}[/bold] embedding {query_embedding.shape}"
                    )
                case _:
                    if self.verbose:
                        console.log(f"{self.tag} defaulting to pitch histogram metric")
                    query_embedding = PrettyMIDI(
                        self.pf_recording
                    ).get_pitch_class_histogram(True, True)
                    query_embedding = query_embedding.reshape(1, -1)
                    console.log(f"{self.tag} {query_embedding}")
        else:
            query_embedding = np.array(
                self.faiss_index.reconstruct(self.filenames.index(query_file)),
                dtype=np.float32,
            )
            query_embedding = np.array(
                [self.emb_table.loc[query_file, "normed_embeddings"]],
                dtype=np.float32,
            )
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # query index
        if self.verbose and track != "player-recording":
            console.log(
                f"{self.tag} querying with key '{query_file}' from index {self.emb_table.index.get_loc(query_file)}"
            )
        similarities, indices = self.faiss_index.search(query_embedding, 1000)  # type: ignore
        if self.verbose:
            console.log(f"{self.tag} indices:\n\t", indices[0][:10])
            console.log(f"{self.tag} similarities:\n\t", similarities[0][:10])

        # reformat and filter shifted files as it often sounds bad
        indices, similarities = zip(
            *[
                (i, d)
                for i, d in zip(indices[0], similarities[0])
                if str(self.emb_table.index[i]).endswith("s00")
            ]
        )

        # find most similar valid match
        if track == "player-recording":
            next_file = self._get_random()
        else:
            next_file = self._get_neighbor()
        console.log(f"{self.tag} 'random' file is '{next_file}'")
        played_files = [os.path.basename(self.base_file(f)) for f in self.played_files]
        for i_neighbor, similarity in zip(indices, similarities):
            segment_name = str(self.emb_table.index[i_neighbor])
            if track == "player-recording":
                next_file = f"{segment_name}.mid"
                break
            # dont replay files
            if segment_name in played_files:
                continue

            next_segment_name = self.base_file(segment_name)
            next_track = next_segment_name.split("_")[0]
            last_track = self.played_files[-1].split("_")[0]
            # switch to different track after self.n_transition_interval segments
            if hop and self.n_segment_repeats >= self.n_transition_interval:
                played_tracks = [file.split("_")[0] for file in self.played_files]
                if next_track in played_tracks:
                    console.log(
                        f"{self.tag} transitioning to next track and skipping '{next_segment_name}'"
                    )
                    continue
                else:
                    next_file = f"{segment_name}.mid"
                    break
            # no shift because it sounds bad
            if (
                next_segment_name not in played_files
                and segment_name.endswith("s00")
                # and next_track == last_track
            ):
                next_file = f"{segment_name}.mid"
                break

        # add fake transformation string for test dataset
        if os.path.basename(self.p_dataset) == "test":
            next_file = os.path.splitext(next_file)[0] + "_t00s00.mid"

        console.log(
            f"{self.tag} best match is '{next_file}' with similarity {similarity:.05f}"
        )
        return os.path.join(self.p_dataset, next_file), similarity

    def _get_easy(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} played files: {self.played_files}")
            console.log(f"{self.tag} num_repeats: {self.n_segment_repeats}")
        if self.n_segment_repeats < self.n_segment_repeats:
            console.log(f"{self.tag} transitioning to next segment")
            self.n_segment_repeats += 1
            return self._get_neighbor()
        else:
            console.log(f"{self.tag} transitioning to next track")
            self.n_segment_repeats = 0
            return (
                self._get_random()
            )  # TODO: modify this to get nearest neighbor from different track

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

    def _get_graph(self) -> str:
        """
        Find the nearest segment with a different track using FAISS,
        load the relevant networkx graph, and plot the path.

        Returns
        -------
        str
            The next segment in the path to play.
        """
        # Get the seed file (most recently played file)
        if not self.played_files:
            return (
                self._get_random()
            )  # If no files have been played yet, choose randomly

        seed_file = self.played_files[-1]
        console.log(f"{self.tag} using seed file '{seed_file}' for graph navigation")

        # Extract the track from the seed file
        seed_track = seed_file.split("_")[0]
        console.log(f"{self.tag} seed track is '{seed_track}'")

        # Get the embedding for the seed file
        # Handle augmented filenames if needed
        if len(seed_file.split("_")) >= 3:
            seed_key = "_".join(seed_file.split("_")[:3])
            if seed_key[-4:] == ".mid":
                seed_key = seed_key[:-4]
        else:
            seed_key = "_".join(seed_file.split("_")[:2])
            if seed_key[-4:] == ".mid":
                seed_key = seed_key[:-4]
            seed_key = f"{seed_key}_t00s00"

        console.log(f"{self.tag} seed key for embedding lookup: '{seed_key}'")

        # Get the embedding for the seed file
        try:
            seed_embedding = np.array(
                [self.emb_table.loc[seed_key, "normed_embeddings"]],
                dtype=np.float32,
            )
        except KeyError:
            console.log(
                f"{self.tag} could not find embedding for '{seed_key}', choosing randomly"
            )
            return self._get_random()

        # Find nearest segments using FAISS
        console.log(f"{self.tag} searching for nearest segment from a different track")
        similarities, indices = self.faiss_index.search(seed_embedding, 500)  # type: ignore

        # Find the nearest segment from a different track
        nearest_segment = None
        nearest_similarity = 0.0

        for idx, similarity in zip(indices[0], similarities[0]):
            segment_name = str(self.emb_table.index[idx])
            segment_track = segment_name.split("_")[0]

            # Skip segments from the same track
            if segment_track == seed_track:
                continue

            # Found a segment from a different track
            nearest_segment = segment_name
            nearest_similarity = float(similarity)
            break

        if nearest_segment is None:
            console.log(
                f"{self.tag} could not find a segment from a different track, choosing randomly"
            )
            return self._get_random()

        console.log(
            f"{self.tag} nearest segment from different track: '{nearest_segment}' with similarity {nearest_similarity:.4f}"
        )

        # Load the relevant graph files
        graph_dir = os.path.join("data", "datasets", "20250110", "graphs")

        # Load source track graph
        source_graph_path = os.path.join(graph_dir, f"{seed_track}.json")
        target_track = nearest_segment.split("_")[0]
        target_graph_path = os.path.join(graph_dir, f"{target_track}.json")

        # Verify graph files exist
        if not os.path.exists(source_graph_path):
            console.log(
                f"{self.tag} source graph file '{source_graph_path}' not found, choosing randomly"
            )
            return self._get_random()

        if not os.path.exists(target_graph_path):
            console.log(
                f"{self.tag} target graph file '{target_graph_path}' not found, choosing randomly"
            )
            return self._get_random()

        # Load the graphs
        console.log(f"{self.tag} loading source graph from '{source_graph_path}'")
        console.log(f"{self.tag} loading target graph from '{target_graph_path}'")

        try:
            with open(source_graph_path, "r") as f:
                source_graph_data = json.load(f)
            source_graph = nx.node_link_graph(source_graph_data)

            with open(target_graph_path, "r") as f:
                target_graph_data = json.load(f)
            target_graph = nx.node_link_graph(target_graph_data)
        except Exception as e:
            console.log(f"{self.tag} error loading graph files: {e}, choosing randomly")
            return self._get_random()

        # Prepare nodes for path finding
        seed_node = "_".join(seed_key.split("_")[:2])
        target_node = "_".join(nearest_segment.split("_")[:2])

        console.log(f"{self.tag} finding path from '{seed_node}' to '{target_node}'")

        # Create combined graph
        combined_graph = nx.compose(source_graph, target_graph)

        # Add an edge connecting the two graphs with a weight based on similarity
        edge_weight = max(1.0 - nearest_similarity, 0.01)  # Ensure weight is positive
        combined_graph.add_edge(seed_node, target_node, weight=edge_weight)

        # Find the shortest path
        try:
            path = nx.shortest_path(
                combined_graph, source=seed_node, target=target_node, weight="weight"
            )
            console.log(f"{self.tag} found path with {len(path)} nodes")
        except nx.NetworkXNoPath:
            console.log(f"{self.tag} no path found, choosing randomly")
            return self._get_random()

        # Store the path in the class
        self.current_path = path
        self.current_path_position = 0

        # Return the next segment in the path (excluding the seed file)
        if len(path) > 1:
            next_node = path[1]  # Get the next node in the path
            self.current_path_position = 1
            next_file = f"{next_node}_t00s00.mid"  # Add transformation
            console.log(f"{self.tag} next file in path: '{next_file}'")
            return next_file
        else:
            # If path has only one node, return the target
            next_file = f"{target_node}_t00s00.mid"
            console.log(f"{self.tag} returning target file: '{next_file}'")
            return next_file

    def _get_random(self) -> str:
        console.log(
            f"{self.tag} choosing randomly from '{self.p_dataset}':\n{[m for m in os.listdir(self.p_dataset)][:5]}"
        )
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(SUPPORTED_EXTENSIONS)],
            1,
        )[0]

        # correct for missing augmentation information if not present
        if len(random_file.split("_")) < 3:
            random_file = random_file[:-4] + "_t00s00.mid"

        # only play files once
        if not self.allow_multiple_plays:
            base_file = self.base_file(random_file)
            while base_file in self.played_files:
                random_file = self.rng.choice(
                    [
                        m
                        for m in os.listdir(self.p_dataset)
                        if m.endswith(SUPPORTED_EXTENSIONS)
                    ]
                )
                base_file = self.base_file(random_file)

        # no shift -- too risky
        random_file = os.path.splitext(random_file)[0][:-3] + "s00.mid"

        if self.verbose:
            console.log(f"{self.tag} chose random file '{random_file}'")

        return str(random_file)

    def _read_playlist(self) -> str:
        # TODO: modify this to work from current playlist paradigm
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
        match self.params.metric:
            case "clamp":
                raise NotImplementedError("CLaMP model is no longer supported")
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    console.log(f"{self.tag} Using GPU {torch.cuda.get_device_name(0)}")
                else:
                    console.log(f"{self.tag} No GPU available, using the CPU instead")
                    device = torch.device("cpu")
                self.params.model = clamp.CLaMP.from_pretrained(clamp.CLAMP_MODEL_NAME)
                if self.verbose:
                    console.log(f"{self.tag} Loaded model:\n{self.params.model.eval}")
                self.params.model = self.params.model.to(device)  # type: ignore
            case "cdl-4note" | "clf-speed" | "clf-tpose":
                import torch
                from utils.models import Classifier

                clf = Classifier(768, [128], 120)
                clf.load_state_dict(
                    torch.load(
                        os.path.join("data", "models", self.params.metric + ".pth")
                    )
                )

                return clf
            case _:
                raise TypeError(
                    f"{self.tag} Unsupported model specified: {self.params.metric}"
                )
