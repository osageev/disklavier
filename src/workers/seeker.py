import os
import numpy as np
import pandas as pd
from shutil import copy2
from mido import bpm2tempo
from pretty_midi import PrettyMIDI
import redis
import faiss
from redis.commands.search.query import Query
from scipy.spatial.distance import cosine

from .worker import Worker
from utils import console, panther


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
    pitch_match = True

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

        if hasattr(params, "pf_recording"):
            self.pf_recording = params.pf_recording
        if hasattr(params, "panther_user"):
            self.panther_user = params.panther_user

        if params.mode == "best":
            self.metric = params.metric

        # if params.metric in model_list:
        #     self.load_model()

        if params.redis:
            self.redis_client = redis.Redis(
                host=params.redis.host,
                port=params.redis.port,
                db=params.redis.db,
                decode_responses=True,
            )

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
            if self.metric == "clamp":
                self.emb_table["normed_embeddings"] = self.emb_table[
                    "embeddings"
                ].apply(lambda x: x / np.linalg.norm(x))
            else:
                self.emb_table["normed_embeddings"] = self.emb_table.iloc[
                    :, 0:12
                ].apply(lambda row: row.tolist(), axis=1)

            console.log(self.emb_table[["normed_embeddings"]].head())

            self.faiss_index = faiss.IndexFlatIP(768 if self.metric == "clamp" else 12)
            self.faiss_index.add(np.array(self.emb_table["normed_embeddings"].to_list(), dtype=np.float32))  # type: ignore
            console.log(
                f"FAISS index built ({np.array(self.emb_table["normed_embeddings"].to_list(), dtype=np.float32).shape})"
            )
        else:
            console.log(f"{self.tag} error loading embeddings table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        # # load similarity table
        # pf_sim_table = os.path.join(self.p_table, "sim.parquet")
        # console.log(f"{self.tag} looking for similarity table '{pf_sim_table}'")
        # if os.path.isfile(pf_sim_table):
        #     with console.status("\t\t\t      loading similarities file..."):
        #         self.sim_table = pd.read_parquet(pf_sim_table)
        #     console.log(
        #         f"{self.tag} loaded {len(self.sim_table)}*{len(self.sim_table.columns)} sim table"
        #     )
        #     console.log(self.sim_table.head())
        # else:
        #     console.log(f"{self.tag} error loading similarity table, exiting...")
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

        # # load transformation table
        # pf_trans_table = os.path.join(self.p_table, "transformations.parquet")
        # console.log(f"{self.tag} looking for tranformation table '{pf_trans_table}'")
        # if os.path.isfile(pf_trans_table):
        #     with console.status("\t\t\t      loading tranformation file..."):
        #         self.trans_table = pd.read_parquet(pf_trans_table)
        #     console.log(
        #         f"{self.tag} loaded {len(self.trans_table)}*{len(self.trans_table.columns)} transformation table"
        #     )
        #     console.log(self.trans_table.head())
        # else:
        #     console.log(f"{self.tag} error loading tranformation table, exiting...")
        #     exit()  # TODO: handle this better (return an error, let main handle it)

        console.log(f"{self.tag} initialization complete")

    def get_next(self) -> str:
        match self.mode:
            case "best":
                next_file = self._get_best()
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

        return next_file

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        return os.path.join(self.p_dataset, random_file)

    def _get_best(self) -> str:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )

        console.log(f"{self.tag} played files:\n{self.played_files}")
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

        # if self.verbose:
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

        # if self.verbose:
        console.log(
            f"{self.tag} got nearest neighbors to '{q_key}':",
            sorted(nearest_neighbors.items(), key=lambda item: item[1], reverse=True)[
                :10
            ],
        )

        next_file = self._get_neighbor()
        console.log(f"{self.tag} random file is '{next_file}'")
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
            # switch to different track after 2 segments
            if len(self.played_files) % 2 == 0:
                if next_track == last_track:
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
        console.log(
            f"{self.tag} vector compare\n\t{q_embedding}\n\t{self.emb_table.loc[next_file[:-4], "normed_embeddings"]}"
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
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
        )

        if self.verbose:
            console.log(f"{self.tag} chose random file '{random_file}'")

        # only play files once
        if not self.allow_multiple_plays:
            base_file = self.base_file(random_file)
            while base_file in self.played_files:
                base_file = self.base_file(random_file)
                random_file = self.rng.choice(
                    [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
                )

        # NO SHIFT
        random_file = random_file[:-7] + "s00.mid"

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
            os.path.join(os.path.dirname(self.p_dataset), "train", pf_in),
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


#     q_k = f"files:{track}_{segment}_{current_transformation}"
#     q_k = f"{track}_{segment}_{current_transformation}"
#     console.log(f"{self.tag} querying with key '{q_k}'")
#     # query_embedding = self.redis_client.json().get(q_k, f"$.{self.metric}")
#     query_embedding = self.emb_table.loc[q_k, "embeddings"]
#     console.log(f"{self.tag} got embedding {query_embedding.shape}")
#     # if query_embedding:
#     #     query_embedding = query_embedding[0]
#     #     if self.verbose:
#     #         console.log(
#     #             f"{self.tag} got embedding for '{q_k}': ({len(query_embedding)})"  # type: ignore
#     #         )
#     # else:
#     #     console.log(f"{self.tag} failed to get embedding for '{q_k}'")
# played_keys = [k.split(":")[-1] for k in self.construct_keys()]
# played_files = "|".join(played_keys)
# # q = f"(-@files:{{{played_files}}})=>[KNN 10 @{self.metric} $query_vector AS vector_score]"
# # q = f"(*)=>[KNN 10 @{self.metric} $query_vector AS vector_score]"
# # console.log(f"{self.tag} trying query:\n'{q}'")
# # console.log(
# #     f"{self.tag} vector:\n",
# #     np.array(query_embedding, dtype=np.float32).tobytes(),
# # )
# # nearest_neighbors = (
# #     self.redis_client.ft(f"idx:files_{self.metric}_vss")
# #     .search(
# #         Query(q)
# #         .sort_by("vector_score")
# #         .return_fields("vector_score", "id")
# #         .dialect(4),
# #         {"query_vector": np.array(query_embedding, dtype=np.float32).tobytes()},
# #     )
# #     .docs  # type: ignore
# # )

# similarities = []
# for chunk in self.emb_chunks:
#     embeddings = np.vstack(
#         chunk.loc[:, "embeddings"].values
#     )  # Convert to 2D array
#     sims = np.dot(query_embedding, embeddings.T) / (
#         np.linalg.norm(query_embedding) * np.linalg.norm(embeddings, axis=1)
#     )

#     # Update top similarities
#     for idx, sim in enumerate(sims):
#         if len(similarities) < 1000:
#             similarities.append(
#                 (float(sim), chunk.index[idx])
#             )  # Store similarity and corresponding index
#         else:
#             # If we have 1000 similarities, check if the current one is higher than the lowest
#             min_sim = min(similarities, key=lambda x: x[0])
#             if sim > min_sim[0]:
#                 similarities.remove(min_sim)  # Remove the lowest similarity
#                 similarities.append((float(sim), chunk.index[idx]))
# similarities.sort(reverse=True, key=lambda x: x[0])
# nearest_neighbors = [{"sim": s[0], "id": s[1]} for s in similarities]

# similarities = []
# for index, row in self.emb_table.iterrows():
#     similarity = 1 - cosine(query_embedding, row["embeddings"])
#     similarities.append((index, similarity))


# Traceback (most recent call last):
#   File "/Users/finlay/Documents/Programming/disklavier/src/main.py", line 273, in <module>
#     main(args, params)
#   File "/Users/finlay/Documents/Programming/disklavier/src/main.py", line 136, in main
#     pf_next_file = seeker.get_next()
#                    ^^^^^^^^^^^^^^^^^
#   File "/Users/finlay/Documents/Programming/disklavier/src/workers/seeker.py", line 165, in get_next
#     shifted_pch = PrettyMIDI(transposed_file).get_pitch_class_histogram(
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/finlay/Documents/Programming/disklavier/.venv/lib/python3.12/site-packages/pretty_midi/pretty_midi.py", line 63, in __init__
#     midi_data = mido.MidiFile(filename=midi_file)
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/finlay/Documents/Programming/disklavier/.venv/lib/python3.12/site-packages/mido/midifiles/midifiles.py", line 319, in __init__
#     with open(filename, 'rb') as file:
#          ^^^^^^^^^^^^^^^^^^^^
# FileNotFoundError: [Errno 2] No such file or directory: 'data/datasets/20240621/train/20240511-050-01_05_t00s-0518.mid'
