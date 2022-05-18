# %% Imports:
import re
from itertools import compress

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from solvers.models import load_language

# %% Load original spammy model:
model = load_language("english", None)
print("loaded original model")

# %% function to filter out trash words:


def words_filter(x: str):
    patterns = r"^[a-zA-Z\.]+$"
    only_dots_pattern = r"^\.*$"
    if re.match(patterns, x) is None or re.match(only_dots_pattern, x) is not None:
        return False
    else:
        return True


# %% create two lists of cleaned data:
logical_idx = [words_filter(w) for w in model.index_to_key]
clean_words = list(compress(model.index_to_key, logical_idx))  # [x for x in model.index_to_key if filter(x)]
clean_vectors = model.vectors[logical_idx, :]
print("cleaned data")

# %% Create a Dataframe with the clean data, and aggregate identical words in different capitalization:
vectors_list_of_lists = clean_vectors.tolist()
vectors_list_of_arrays = [np.array(v) for v in vectors_list_of_lists]
lowercase_words = [s.lower() for s in clean_words]

df = pd.DataFrame({"lowercase_words": lowercase_words, "vector": vectors_list_of_arrays})
grouped = df.groupby("lowercase_words")["vector"].agg(np.mean)
print("capitalization unified")

# %% Create a cleaned gensim word2vec model:
clean_model = KeyedVectors(vector_size=300)
clean_model.index_to_key = grouped.index.to_list()
clean_model.key_to_index = {s: i for i, s in enumerate(clean_model.index_to_key)}
agg_vectors = np.stack(grouped, axis=0)
clean_model.vectors = agg_vectors
print("created cleaned model")

# %% Save model:
clean_model.save_word2vec_format(r"language_data\english_cleaned.bin", binary=True)
print("model saved")
