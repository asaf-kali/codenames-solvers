import re
from itertools import compress

import numpy as np
import pandas as pd

from codenames.model_loader import load_language

# clean_word_idx =
# %%
model = load_language("english")
temp_words = model.index_to_key[:100]


# %%
def filter(x: str):
    patterns = r"^[a-zA-Z\.]+$"
    if re.match(patterns, x) is None:
        return False
    else:
        return True


# %%

logical_idx = [filter(w) for w in model.index_to_key]
clean_words = list(compress(model.index_to_key, logical_idx))  # [x for x in model.index_to_key if filter(x)]
clean_vectors = model.vectors[logical_idx, :]

# %%
vectors_list_of_lists = clean_vectors.tolist()
vectors_list_of_arrays = [np.array(v) for v in vectors_list_of_lists]
lowercase_words = [s.lower() for s in clean_words]

df = pd.DataFrame({"word": clean_words, "lowercase_words": lowercase_words, "vector": vectors_list_of_arrays})


# %%


def word_filter(x: str):
    patterns = r"^[a-zA-Z]+$"
    if re.match(patterns, x) is None:
        return False
    else:
        return True


# %%
good_idx = df.loc[1:100, "words"].apply(word_filter)

# %%
for word in model.index_to_key[109990:110000]:
    print(word, word_filter(word))
