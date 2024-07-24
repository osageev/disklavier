import os
import json
import pretty_midi
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import cosine
from rich.progress import (
    track,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import multiprocessing
from functools import partial

from utils import console
import utils.metrics as metrics
from utils.plot import plot_histograms
from utils.midi import transform

from typing import Tuple, Dict


class Seeker:
    p = "[yellow]seeker[/yellow]:"
    sim_table: pd.DataFrame
    count = 0
    transition_probability = 0. # max is 1.0

    def __init__(
        self,
        params,
        input_dir: str,
        output_dir: str,
        tempo: int,
        dataset: str,
        mode: str,
    ) -> None:
        """"""
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params.tempo = tempo
        self.params.seed = 1 if self.params.seed is None else self.params.seed
        self.rng = np.random.default_rng(self.params.seed)
        self.mode=mode

        # load similarity table
        sim_table_path = os.path.join("data", "tables", f"{dataset}_sim.parquet")
        console.log(f"{self.p} looking for similarity table '{sim_table_path}'")
        if os.path.isfile(sim_table_path):
            console.log(f"{self.p} loading sim table at '{sim_table_path}'")
            with console.status("\t\t\t      loading similarities file..."):
                self.sim_table = pd.read_parquet(sim_table_path)
        else:
            console.log(f"{self.p} error loading similarity table, exiting...")
            exit()
        
        # load neighbor table
        neighbor_table_path = os.path.join("data", "tables", f"{dataset}_neighbor.parquet")
        console.log(f"{self.p} looking for neighbor table '{neighbor_table_path}'")
        if os.path.isfile(neighbor_table_path):
            console.log(f"{self.p} loading neighbor table at '{neighbor_table_path}'")
            with console.status("\t\t\t      loading neighbor file..."):
                self.neighbor_table = pd.read_parquet(neighbor_table_path)
        else:
            console.log(f"{self.p} error loading neighbor table, exiting...")
            exit()
        
        # load transformation table
        trans_table_path = os.path.join("data", "tables", f"{dataset}_transformations.parquet")
        console.log(f"{self.p} looking for tranformation table '{trans_table_path}'")
        if os.path.isfile(trans_table_path):
            console.log(f"{self.p} loading tranformation table at '{trans_table_path}'")
            with console.status("\t\t\t      loading tranformation file..."):
                self.trans_table = pd.read_parquet(trans_table_path)
        else:
            console.log(f"{self.p} error loading tranformation table, exiting...")
            exit()
        console.log(f"{self.p} [green]successfully loaded tables")

        console.log(f"{self.p} initialized to use metric '{self.params.property}'")

    def get_most_similar_file(
        self, filename: str, different_parent=True, bump_trans=False
    ) -> Dict:
        """"""
        console.log(f"{self.p} finding most similar file to '{filename}' (MODE={self.mode})")

        if bump_trans:
            console.log(f"{self.p} increasing transition probability {self.transition_probability} -> {self.transition_probability + self.params.transition_increment}")
            self.transition_probability += self.params.transition_increment

            if self.transition_probability > 1.:
                self.transition_probability = 0

        parent_track, _, _ = filename.split("_")

        row = self.sim_table.loc[filename]
        sorted_row = row.sort_values(key=lambda x: x.str['sim'], ascending=False)

        change_track = self.rng.choice([True, False], p=[self.transition_probability, 1 - self.transition_probability])
        console.log(f"{self.p} rolled {change_track} w/tprob {self.transition_probability}")

        if change_track:
            for next_filename, val in sorted_row.items():
                next_track, _ = next_filename.split("_")

                if next_track == parent_track or next_filename == filename:
                    # console.log(f"{self.p} skipping invalid match\n\t'{next_track}' == '{parent_track}' or\n\t'{next_filename}' == '{filename}'")
                    continue
                if next_track != parent_track and next_filename != filename:
                    value = val
                    value["filename"] = next_filename
                    break
        else:
            value = {
                "filename": self.rng.choice(self.neighbor_table.loc[filename]),
                "sim": -1.,
                "transformations": {
                    "shift": 0,
                    "trans": 0,
                },
            }

        console.log(
            f"{self.p} found '{value['filename']}' with similarity {value["sim"]:.03f}", value
        )

        return value
  
    def match_recording(self, recording_path: str):
        console.log(
            f"{self.p} finding most similar vector to '{recording_path}' with metric '{self.params.property}'"
        )
        recording = pretty_midi.PrettyMIDI(recording_path)
        recording_metric = recording.get_pitch_class_histogram(True, True)

        most_similar_segment = ""
        highest_similarity = -1.0
        best_transformations = {}

        # TODO: VECTORIZE THIS
        for segment_name in track(self.trans_table.index.tolist(), "calculating similarities...", refresh_per_second=1, update_period=1.0):
            for semi in list(range(12)): # [list(p) for p in itertools.product(list(range(12)), list(range(8)))]
                beat = 0
                similarity = float(1 - cosine(recording_metric, self.trans_table.at[segment_name, f"{semi:02d}{beat:02d}"]))  # type: ignore
                if similarity > highest_similarity:
                    # console.log(f"{self.p} updating similarity {highest_similarity:.03f} -> {similarity:.03f}\n\t'{segment_name}' -> '{most_similar_segment}'\tt{semi} & s{beat}")
                    highest_similarity = similarity
                    most_similar_segment = segment_name
                    best_transformations = {
                        "transpose": semi,
                        "shift": beat
                    }

        console.log(
            f"{self.p} \tfound '{most_similar_segment}' with similarity {highest_similarity:.03f} using transformations {best_transformations}"
        )

        return most_similar_segment, highest_similarity, best_transformations

    def reset_plays(self) -> None:
        for k in self.properties.keys():
            self.properties[k]["played"] = 0

    def get_random(self) -> str:
        return os.path.join(self.input_dir, self.rng.choice(os.listdir(self.input_dir)))


###############################################################################
#############################      GRAVEYARD     ##############################
###############################################################################

# def build_top_n_table(self, n: int = 10, vision: int = 2) -> None:
#     """ """
#     parquet = os.path.join(
#         self.output_dir,
#         f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}-{self.params.property}.parquet",
#     )

#     self.load_similarities(parquet)

#     if self.sim_table is not None:
#         console.log(
#             f"{self.p} loaded existing similarity file from '{parquet}' ({self.sim_table.shape})\n",
#             self.sim_table.columns,
#             self.sim_table.index[:4],
#         )

#         return

#     if n % 2:
#         console.log(
#             f"{self.p} [yellow]odd value passed in for n ([/yellow]{n}[yellow]), rounding down"
#         )
#         n -= 1

#     vectors = [
#         {
#             "name": filename,
#             "metric": details["properties"][self.params.property],
#         }
#         for filename, details in self.properties.items()
#     ]

#     if self.params.property == "pr_blur_c":
#         vectors = [
#             {"name": v["name"], "metric": np.asarray(v["metric"]).flatten()}
#             for v in vectors
#             if v["name"].endswith("n00.mid")
#         ]
#     if self.params.property == "pr_blur":
#         vectors = [
#             {"name": v["name"], "metric": np.asarray(v["metric"]).flatten()}
#             for v in vectors
#         ]

#     names = [v["name"] for v in vectors]
#     vecs = [v["metric"] for v in vectors]

#     console.log(
#         f"{self.p} building top-{n} similarity table for {len(vecs)} vectors from '{self.input_dir}' using metric '{self.params.property}'"
#     )

#     labels = [
#         "loop  ",
#         "prev 1",
#         "next 1",
#         "prev 2",
#         "next 2",
#         "diff 1",
#         "diff 2",
#         "diff 3",
#         "diff 4",
#         "diff 5",
#     ]
#     column_labels = [[label, f"sim-{i + 1}"] for i, label in enumerate(labels)]
#     column_labels = [label for sublist in column_labels for label in sublist]

#     self.sim_table = pd.DataFrame(
#         [["", -1.0] * len(labels)] * len(names),
#         index=names,
#         columns=column_labels,
#     )

#     console.log(
#         f"{self.p} initialized table ({len(names)}, {len(column_labels)}) with rows:\n{names[:5]}\nand columns:\n{column_labels}"
#     )

#     progress = Progress(
#         SpinnerColumn(),
#         *Progress.get_default_columns(),
#         TimeElapsedColumn(),
#         MofNCompleteColumn(),
#         refresh_per_second=1,
#     )
#     sims_task = progress.add_task("calculating sims", total=len(vecs) ** 2)
#     with progress:
#         for name in self.sim_table.index:
#             # console.log(f"\n{self.p} populating row '{name}'")

#             i = int(self.sim_table.index.get_loc(name))  # type: ignore
#             # i_name, i_seg_num, i_shift = name.split("_")
#             i_name, i_seg_num = name.split("_")
#             i_seg_start, i_seg_end = i_seg_num.split("-")
#             i_seg_end = i_seg_end.split(".")[0]

#             # populate first five columns
#             # get prev file(s)
#             prv2_file = None
#             prev_file = self.get_prev(name)
#             if prev_file:
#                 if int(i_seg_start) != 0:
#                     prv2_file = self.get_prev(prev_file)

#             # get next file(s)
#             nxt2_file = None
#             next_file = self.get_next(name)
#             if next_file:
#                 nxt2_file = self.get_next(next_file)

#             names = [name, prev_file, next_file, prv2_file, nxt2_file]

#             for j, k in zip(range(0, n, 2), names):
#                 self.sim_table.iat[i, j] = k

#             # update second five columns
#             for other_name in self.sim_table.index:
#                 # console.log(f"{self.p} checking col '{names[j]}'")
#                 j = int(self.sim_table.index.get_loc(other_name))  # type: ignore
#                 # j_name, j_seg_num, j_shift = other_name.split("_")
#                 j_name, j_seg_num = other_name.split("_")

#                 # console.log(f"{self.p} v i", vecs[i])
#                 # console.log(f"{self.p} v j", vecs[j])

#                 sim = float(1 - cosine(vecs[i], vecs[j]))

#                 diff_track_range = range(n + 1, n * 2, 2)
#                 if i_name != j_name:  # clip is from a different track
#                     self.replace_smallest_sim(
#                         name,
#                         other_name,
#                         sim,
#                         diff_track_range,
#                     )

#                 progress.update(sims_task, advance=1)

#     console.log(
#         f"{self.p} Generated a similarity table of shape {self.sim_table.shape}"
#     )

#     self.sim_table.to_parquet(parquet, index=True)
#     if os.path.isfile(parquet):
#         console.log(f"{self.p} succesfully saved similarities file '{parquet}'")
#     else:
#         console.log(f"{self.p} error saving similarities file '{parquet}'")
#         raise FileNotFoundError

# def replace_smallest_sim(
#     self, src_row: str, cmp_file: str, sim: float, col_range: range
# ) -> None:
#     row_index = self.sim_table.index.get_loc(src_row)
#     smallest_value = float("inf")
#     smallest_index = None

#     # console.log(f"{self.p} checking row:\n{self.sim_table.iloc[row_index]}")

#     for col in col_range:
#         current_value = self.sim_table.iloc[row_index, col]  # type: ignore
#         # console.log(f"{self.p} got value at [{row_index}, {col}]: {current_value}")

#         if current_value < sim and current_value < smallest_value:
#             smallest_value = current_value
#             smallest_index = col

#     # If a smaller value was found, replace the tuple at its index
#     if smallest_index is not None:
#         self.sim_table.iat[row_index, smallest_index] = sim
#         self.sim_table.iat[row_index, smallest_index - 1] = cmp_file

    
# def get_msf_new(self, filename: str):
#     """finds the filename and similarity of the next most similar unplayed file in the similarity table
#     NOTE: will go into an infinite loop once all files are played!
#     """
#     console.log(
#         f"{self.p} finding most similar file to '{filename}'",
#     )

#     self.properties[filename]["played"] += 1  # mark current file as played

#     columns = list(self.sim_table.columns[::2].values)
#     roll = self.rng.choice(columns, p=self.probs)
#     if columns.index(roll) > 5:
#         console.log(f"{self.p} \t[blue1]TRACK TRANSITION[/blue1] (rolled '{roll}')")

#     if self.params.calc_trans and not filename.endswith("n00.mid"):
#         self.last_trans = filename[-7:]
#         filename = filename[:-7] + "n00.mid"

#     if self.params.max_sim:
#         roll = self.get_max_sim(filename)

#     next_filename = self.sim_table.at[filename, f"{roll}"]

#     # when the source file is at the start or end of a track the prev/next
#     # columns respectively can be None
#     while next_filename == "" or next_filename == None:
#         console.log(f"{self.p} \t[blue1]REROLL[/blue1] (rolled '{roll}')")
#         roll = self.rng.choice(columns, p=self.probs)
#         next_filename = self.sim_table.at[filename, f"{roll}"]

#     next_col = self.sim_table.columns.get_loc(roll) + 1  # type: ignore
#     similarity = float(self.sim_table.at[filename, self.sim_table.columns[next_col]])

#     # check transposition if using centered blur
#     if self.params.calc_trans:
#         next_filename, similarity = self.pitch_transpose(
#             os.path.join(self.input_dir, filename),
#             os.path.join(self.input_dir, next_filename),
#             similarity,
#         )

#     console.log(
#         f"{self.p} \tfound '{next_filename}' with similarity {similarity:.03f}"
#     )

#     return next_filename, similarity


# def get_ms_to_recording(self, recording_path: str) -> Tuple[str | None, float]:
#     console.log(
#         f"{self.p} finding most similar vector to '{recording_path}' with metric '{self.params.property}'"
#     )

#     midi = pretty_midi.PrettyMIDI(recording_path)

#     match self.params.property:
#         case "energy":
#             cmp_metric = metrics.energy(recording_path)
#         case "pr_blur":
#             cmp_metric = metrics.blur_pr(midi, False)
#         case "pr_blur_c":
#             cmp_metric = metrics.blur_pr(midi)
#         case "contour":
#             cmp_metric = metrics.contour(
#                 midi, self.params.beats_per_seg, self.params.tempo
#             )
#         case "contour-complex":
#             cmp_metric = metrics.contour(
#                 midi, self.params.beats_per_seg, self.params.tempo, False
#             )
#         case _:
#             cmp_metric = midi.get_pitch_class_histogram()

#     most_similar_vector = None
#     highest_similarity = -1.0  # since cosine similarity ranges from -1 to 1
#     vector_array = [
#         {"name": filename, "metric": details["properties"][self.params.property]}
#         for filename, details in self.properties.items()
#     ]

#     for vector_data in vector_array:
#         name, vector = vector_data.values()
#         similarity = float(1 - cosine(cmp_metric, vector))  # type: ignore
#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             most_similar_vector = name

#     console.log(
#         f"{self.p} \tfound '{most_similar_vector}' with similarity {highest_similarity:.03f}"
#     )

#     if self.params.calc_trans:
#         most_similar_vector, highest_similarity = self.pitch_transpose(
#             recording_path, os.path.join(self.input_dir, str(most_similar_vector))
#         )

#     return most_similar_vector, highest_similarity

# def pitch_transpose(
#     self, seed: str, match: str, piano_roll_sim: float = -1.0
# ) -> Tuple[str, float]:

#     seed_ph = pretty_midi.PrettyMIDI(seed).get_pitch_class_histogram()
#     match_ph = pretty_midi.PrettyMIDI(match).get_pitch_class_histogram()
#     match_ph_sim = float(1 - cosine(seed_ph, match_ph))
#     # console.log(
#     #     f"{self.p} \tunshifted match ('{os.path.basename(seed)}' :: '{os.path.basename(match)}') has ph sim {match_ph_sim:.03f}"
#     # )

#     if piano_roll_sim > 0:
#         seed = seed[:-7] + self.last_trans
#         seed_ph = pretty_midi.PrettyMIDI(seed).get_pitch_class_histogram()
#         match_ph_sim = float(1 - cosine(seed_ph, match_ph))

#         # console.log(f"{self.p} \tshifted '{os.path.basename(seed)}' has ph sim {match_ph_sim:.03f}")

#     best_match = os.path.basename(match)
#     best_sim = match_ph_sim

#     t_files = []
#     for transposition in self.trans_options:
#         t_file = match[:-7] + transposition

#         # not all transposition options for each file will exist
#         if not os.path.exists(t_file):
#             continue

#         t_ph = pretty_midi.PrettyMIDI(t_file).get_pitch_class_histogram()
#         t_sim = float(1 - cosine(seed_ph, t_ph))
#         t_files.append((t_file, t_sim))

#         if t_sim > best_sim:
#             best_match = os.path.basename(t_file)
#             best_sim = t_sim

#             console.log(
#                 f"{self.p} \tbetter tpos '{os.path.basename(t_file)}' @ sim {t_sim:.03f}"
#             )

#     # plot_histograms(
#     #     [seed_ph, match_ph],
#     #     [os.path.basename(f) for f in [seed, match]],
#     #     os.path.join(
#     #         self.output_dir,
#     #         "plots",
#     #         "_test",
#     #         f"{self.count}-{os.path.basename(seed)[:-4]}-src.png",
#     #     ),
#     #     (2, 1),
#     #     f"neutral sim = {match_ph_sim:.03f}",
#     # )
#     # plot_histograms(
#     #     [pretty_midi.PrettyMIDI(f).get_pitch_class_histogram() for f, _ in t_files],
#     #     [f"{os.path.basename(f)[-7:-4]} ({s:.02f})" for f, s in t_files],
#     #     os.path.join(
#     #         self.output_dir,
#     #         "plots",
#     #         "_test",
#     #         f"{self.count}-{os.path.basename(seed)[:-4]}-phs.png",
#     #     ),
#     #     (4, 3),
#     #     f"best match: {os.path.basename(best_match)[:-4]}",
#     # )
#     self.count += 1

#     return best_match, best_sim


# def get_prev(self, filename):
#     # i_name, i_seg_num, i_shift = filename.split("_")
#     i_name, i_seg_num = filename.split("_")
#     i_seg_start, i_seg_end = i_seg_num.split("-")
#     i_seg_end = i_seg_end.split(".")[0]
#     delta = int(i_seg_end) - int(i_seg_start)

#     if int(i_seg_start) == 0:
#         return None

#     # prev_file = f"{i_name}_{int(i_seg_start) - delta:04d}-{i_seg_start}_{i_shift}"
#     prev_file = f"{i_name}_{int(i_seg_start) - delta:04d}-{i_seg_start}"

#     for key in self.properties.keys():
#         # k_name, k_seg_num, k_shift = key.split("_")
#         k_name, k_seg_num = key.split("_")
#         k_seg_start, k_seg_end = k_seg_num.split("-")
#         k_seg_end = k_seg_end.split(".")[0]

#         if (
#             k_name == i_name
#             # and k_shift == i_shift
#             and abs(int(k_seg_end) - int(i_seg_start)) <= 2
#         ):
#             prev_file = key

#     return prev_file

# def get_next(self, filename):
#     # i_name, i_seg_num, i_shift = filename.split("_")
#     i_name, i_seg_num = filename.split("_")
#     i_seg_start, i_seg_end = i_seg_num.split("-")
#     i_seg_end = i_seg_end.split(".")[0]
#     delta = int(i_seg_end) - int(i_seg_start)

#     # next_file = f"{i_name}_{i_seg_end}-{int(i_seg_end) + delta:04d}_{i_shift}"
#     next_file = f"{i_name}_{i_seg_end}-{int(i_seg_end) + delta:04d}"
#     if next_file not in self.properties.keys():
#         next_file = None
#         for key in self.properties.keys():
#             # k_name, k_seg_num, k_shift = key.split("_")
#             k_name, k_seg_num = key.split("_")
#             k_seg_start, k_seg_end = k_seg_num.split("-")
#             k_seg_end = k_seg_end.split(".")[0]

#             if (
#                 k_name == i_name
#                 # and k_shift == i_shift
#                 and abs(int(k_seg_start) - int(i_seg_end)) <= 2
#             ):
#                 next_file = key

#     return next_file

# def sort_row(self, row):
#     """Sorts the specified sections of a DataFrame row in descending order based on 'sim' values.

#     This function assumes the row contains a fixed part and a sortable part, where the sortable
#     part consists of 'diff' and 'sim' column pairs. The function sorts these pairs by the 'sim'
#     value in descending order while keeping each 'diff' directly in front of its corresponding 'sim'.

#     Args:
#         row (pd.Series): A single row of a DataFrame to be sorted. It is expected that the
#                         row contains mixed data types with 'diff' being filenames (str) and
#                         'sim' being numeric scores (float).

#     Returns:
#         pd.Series: A series with the first part unchanged and the last part sorted based on the
#                 'sim' values.

#     Note:
#         The function is designed to operate within a DataFrame.apply() method which allows it
#         to be applied row-wise. It specifically manages rows that split at index 10, where indices
#         from 10 onward contain 'diff' and 'sim' pairs.

#     Example:
#         # Assuming 'df' is a DataFrame loaded with the appropriate columns and data structure:
#         sorted_df = df.apply(sort_row, axis=1)
#     """
#     fixed_part = row.iloc[:10]  # Assumes the first 10 entries do not need sorting
#     to_sort_part = row.iloc[10:]  # The part that needs sorting

#     # Create a DataFrame from the parts that need sorting
#     # Assuming every two columns are 'diff' and 'sim' pairs starting from index 10
#     df_to_sort = pd.DataFrame(
#         {
#             "diff": to_sort_part[::2].values,  # Assumes even indices are 'diff'
#             "sim": to_sort_part[1::2].values,  # Assumes odd indices are 'sim'
#         },
#         index=pd.MultiIndex.from_arrays(
#             [to_sort_part[::2].index, to_sort_part[1::2].index]
#         ),
#     )

#     # Sort the DataFrame based on 'sim' values in descending order
#     sorted_df = df_to_sort.sort_values(by="sim", ascending=False)

#     # Flatten the sorted DataFrame back into a Series
#     sorted_series = pd.Series(
#         data=sorted_df.values.flatten(),
#         index=[idx for sub_idx in sorted_df.index for idx in sub_idx],
#     )

#     # Concatenate the fixed part and the sorted part
#     return pd.concat([fixed_part, sorted_series])


# def get_max_sim(self, row_label):
#     row = self.sim_table.loc[row_label]
#     most_similar_v = -1
#     most_similar_i = 1
#     for i, (k, v) in enumerate(row.items()):
#         if str(k).startswith("sim") and float(v) > most_similar_v:
#             most_similar_v = v
#             most_similar_i = i

#     return row.index[most_similar_i - 1]

# def build_similarity_table(self) -> None:
#     """"""
#     sim_file = f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}-{self.params.property}.parquet"
#     console.log(f"{self.p} looking for similarity file '{sim_file}'")
#     parquet = os.path.join(self.output_dir, sim_file)
#     self.load_similarities(parquet)

#     if self.sim_table is not None:
#         console.log(f"{self.p} loaded existing similarity file from '{parquet}'")
#         console.log(f"{self.p} {self.sim_table.head()}")
#     else:
#         vectors = [
#             {
#                 "name": filename,
#                 "metric": details["properties"][self.params.property],
#             }
#             for filename, details in self.properties.items()
#         ]

#         names = [v["name"] for v in vectors]
#         vecs = [v["metric"] for v in vectors]

#         console.log(f"{self.p} building similarity table for {len(vecs)} vectors")

#         self.sim_table = pd.DataFrame(index=names, columns=names, dtype="float64")

#         # compute cosine similarity for each pair of vectors
#         with Progress() as progress:
#             sims_task = progress.add_task(
#                 f"{self.p} calculating sims", total=len(vecs) ** 2
#             )
#             for i in range(len(vecs)):
#                 for j in range(len(vecs)):
#                     if i != j:
#                         self.sim_table.iloc[i, j] = 1 - cosine(vecs[i], vecs[j])  # type: ignore
#                     else:
#                         self.sim_table.iloc[i, j] = 1
#                     progress.update(sims_task, advance=1)

#         console.log(
#             f"{self.p} Generated a similarity table of shape {self.sim_table.shape}"
#         )

#         self.sim_table.to_parquet(parquet, index=False)

#         if os.path.isfile(parquet):
#             console.log(f"{self.p} succesfully saved similarities file '{parquet}'")
#         else:
#             console.log(f"{self.p} error saving similarities file '{parquet}'")
#             raise FileNotFoundError
    
# def build_neighbor_table(self, vision: int = 2) -> None:
#     """ """
#     parquet = os.path.join(
#         self.output_dir,
#         f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}-n.parquet",
#     )

#     if os.path.isfile(parquet):
#         self.neighbor_table = pd.read_parquet(parquet)
#         console.log(f"{self.p} building neighbor table from '{parquet}'")

#         return

#     names = list(self.properties.keys())
#     names.sort()

#     console.log(f"{self.p} building nieghbor table for {len(names)} segments")

#     labels = [
#         "loop  ",
#         "prev 1",
#         "next 1",
#         "prev 2",
#         "next 2",
#     ]
#     self.neighbor_table = pd.DataFrame(
#         index=names,
#         columns=labels,
#     )

#     progress = Progress(
#         SpinnerColumn(),
#         *Progress.get_default_columns(),
#         TimeElapsedColumn(),
#         MofNCompleteColumn(),
#         refresh_per_second=1,
#     )
#     sims_task = progress.add_task("finding neighbors", total=len(names))
#     with progress:
#         for name in self.neighbor_table.index:
#             i = int(self.neighbor_table.index.get_loc(name))  # type: ignore
#             i_name, i_seg_num = name.split("_")
#             i_seg_start, i_seg_end = i_seg_num.split("-")
#             i_seg_end = i_seg_end.split(".")[0]

#             # get prev file(s)
#             prv2_file = None
#             prev_file = self.get_prev(name)
#             if prev_file:
#                 if int(i_seg_start) != 0:
#                     prv2_file = self.get_prev(prev_file)

#             # get next file(s)
#             nxt2_file = None
#             next_file = self.get_next(name)
#             if next_file:
#                 nxt2_file = self.get_next(next_file)

#             names = [name, prev_file, next_file, prv2_file, nxt2_file]

#             for j, k in zip(range(5), names):
#                 self.neighbor_table.iat[i, j] = k

#             progress.update(sims_task, advance=1)


#     self.neighbor_table.to_parquet(parquet, index=True)
#     if os.path.isfile(parquet):
#         console.log(f"{self.p} succesfully saved neighbors file '{parquet}'")
#     else:
#         console.log(f"{self.p} error saving neighbors file '{parquet}'")
#         raise FileNotFoundError


    # def build_properties(self) -> None:
    #     dict_file = os.path.join(
    #         self.output_dir,
    #         f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}.json",
    #     )

    #     if os.path.exists(dict_file) and not self.force_rebuild:
    #         console.log(f"{self.p} found existing properties file '{dict_file}'")
    #         with console.status("loading properties file..."):
    #             with open(dict_file, "r") as f:
    #                 self.properties = json.load(f)
    #             console.log(
    #                 f"{self.p} loaded properties for {len(list(self.properties.keys()))} files"
    #             )
    #     else:
    #         console.log(f"{self.p} calculating properties from '{self.input_dir}'")
    #         for file in track(
    #             os.listdir(self.input_dir),
    #             description=f"{self.p} calculating properties",
    #         ):
    #             if file.endswith(".mid") or file.endswith(".midi"):
    #                 file_path = os.path.join(self.input_dir, file)
    #                 midi = pretty_midi.PrettyMIDI(file_path)
    #                 properties = metrics.all_properties(file_path, file, self.params)
    #                 self.properties[file] = {
    #                     "filename": file,
    #                     "properties": properties,
    #                     "played": 0,
    #                 }
    #         console.log(
    #             f"{self.p} calculated properties for {len(list(self.properties.keys()))} files"
    #         )

    #         with open(dict_file, "w") as f:
    #             json.dump(self.properties, f)

    #         if os.path.isfile(dict_file):
    #             console.log(f"{self.p} succesfully saved properties file '{dict_file}'")
    #         else:
    #             console.log(f"{self.p} error saving properties file '{dict_file}'")
    #             raise FileNotFoundError

    #     self.reset_plays()

