import os
import json
import pandas as pd
from scipy.spatial.distance import cosine
import pretty_midi
from tqdm.auto import tqdm

from utils.midi import all_metrics

from typing import List


class Similarity():
    table: pd.DataFrame | None
    metrics = {}
  
    def __init__(self, input_dir: str, output_dir: str, params) -> None:
        """"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params = params


    def build_metrics(self):
        dict_file = os.path.join(self.output_dir, f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}_metrics.json")

        if os.path.exists(dict_file):
            print(f"found existing metrics file '{dict_file}'")
            with open(dict_file, 'r') as f:
                self.metrics = json.load(f)
                print(f"loaded metrics for {len(list(self.metrics.keys()))} files")
        else:
            print(f"calculating metrics from '{self.input_dir}'")
            for file in os.listdir(self.input_dir):
                if file.endswith('.mid') or file.endswith('.midi'):
                    file_path = os.path.join(self.input_dir, file)
                    midi = pretty_midi.PrettyMIDI(file_path)
                    metrics = all_metrics(midi, self.params)
                    self.metrics[file] = {
                    "notes": [],
                    "metrics": metrics,
                    "played": 0,
                    }
            print(f"calculated metrics for {len(list(self.metrics.keys()))} files")

            with open(dict_file, 'w') as f:
                json.dump(self.metrics, f)

            if os.path.isfile(dict_file):
                print(f"succesfully saved metrics file '{dict_file}'")
            else:
                print(f"error saving metrics file '{dict_file}'")
                raise FileNotFoundError
            
        self.reset_plays()


    def build_similarity_table(self):
        """"""
        parquet = os.path.join(self.output_dir, "similarities.parquet")
        self.table = self.load_similarities(parquet)

        if self.table is not None:
            print(f"loaded existing similarity file from '{parquet}'")
        else:
            vectors = [
                {'name': filename, 'metric': details['metrics']['pitch_histogram']}
                for filename, details in self.metrics.items()
            ]

            names = [v['name'] for v in vectors]
            vecs = [v['metric'] for v in vectors]

            print(f"building similarity table for {len(vecs)} vectors")

            self.table = pd.DataFrame(index=names, columns=names, dtype="float64")

            # compute cosine similarity for each pair of vectors
            for i in tqdm(range(len(vecs)), unit="vectors"):
                for j in range(len(vecs)):
                    if i != j:
                        self.table.iloc[i, j] = 1 - cosine(vecs[i], vecs[j]) # type: ignore
                    else:
                        self.table.iloc[i, j] = 1

            
            print(f"Generated a similarity table of shape {self.table.shape}")

            self.table.to_parquet(parquet, index=False)

            if os.path.isfile(parquet):
                print(f"succesfully saved similarities file '{parquet}'")
            else:
                print(f"error saving similarities file '{parquet}'")
                raise FileNotFoundError


    def get_most_similar_file(self, filename, different_parent=True):
        """finds the filename and similarity of the next most similar unplayed file in the similarity table
            NOTE: will go into an infinite loop once all files are played!
        """
        n = 1
        similarity = 1
        next_file_played = 1
        next_filename = None
        self.metrics[filename]["played"] = 1

        while next_file_played:
            nl = self.table[filename].nlargest(n)
            next_filename = self.table.columns[nl.index[-1]]
            similarity = nl.iloc[-1]
            next_file_played = self.metrics[next_filename]["played"]
            n += 1

        return next_filename, similarity
        

    def midi_to_ph(self, midi_file: str):
        """"""
        midi = pretty_midi.PrettyMIDI(midi_file)
        return midi.get_pitch_class_histogram()
        
    
    def find_most_similar_vector(self, target_vector):
        """unused"""
        most_similar_vector = None
        highest_similarity = -1  # since cosine similarity ranges from -1 to 1
        vector_array = [
                {'name': filename, 'metric': details['metrics']['pitch_histogram']}
                for filename, details in self.metrics.items()
            ]

        for vector_data in vector_array:
            name, vector = vector_data.values()
            similarity = 1 - cosine(target_vector, vector)  # type: ignore
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_vector = name

        return most_similar_vector, highest_similarity
    

    def reset_plays(self) -> None:
        for k in self.metrics.keys():
            self.metrics[k]["played"] = 0


    def load_similarities(self, parquet_path) -> pd.DataFrame | None:
        if os.path.isfile(parquet_path):
            return pd.read_parquet(parquet_path)
