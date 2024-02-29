import pandas as pd
from scipy.spatial.distance import cosine

from typing import List


class Similarity():
  
    def __init__(self, params):
        """"""
        self.params = params


    def build_similarity_table(self, vectors: List) -> pd.DataFrame:
        """"""
        # Extract names and vectors
        names = [v['name'] for v in vectors]
        vecs = [v['metric'] for v in vectors]

        # Initialize an empty DataFrame
        df = pd.DataFrame(index=names, columns=names, dtype="float64")

        # Compute cosine similarity for each pair of vectors
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                if i != j:
                    # Compute cosine similarity and store in DataFrame
                    df.iloc[i, j] = 1 - cosine(vecs[i], vecs[j]) # type: ignore
                else:
                    # Cosine similarity of a vector with itself is 1
                    df.iloc[i, j] = 1

        return df


    def get_most_similar_file(self, filename, different_parent=True):
        """"""
        n = 1
        similarity = 1
        next_file_played = 1
        next_filename = filename
        midi_metrics[filename]["played"] = 1

        while next_file_played:
            nl = similarity_table.loc[filename].nlargest(n)
            next_filename = nl.index[-1]
            similarity = nl.iloc[-1]
            next_file_played = midi_metrics[next_filename]["played"]
            n += 1

        return next_filename, similarity
        
    
    def find_most_similar_vector(self, target_vector, vector_array):
        """"""
        most_similar_vector = None
        highest_similarity = -1  # since cosine similarity ranges from -1 to 1

        for vector_data in vector_array:
            name, vector = vector_data.values()
            similarity = 1 - cosine(target_vector, vector)  # type: ignore
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_vector = name

        return most_similar_vector