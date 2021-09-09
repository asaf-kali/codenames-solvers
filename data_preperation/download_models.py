# %% imports:
import os
from datetime import datetime

import gensim.downloader

from codenames.solvers.utils.model_loader import load_language

models_names = ["glove-wiki-gigaword-50"]
LARGE_MODEL_NAMES = ["glove-wiki-gigaword-100", "glove-wiki-gigaword-300"]

# %% Download and save:
begin_time = datetime.now()
for model_name in models_names:
    begin_time = datetime.now()
    # Download:
    temp_model = gensim.downloader.load(model_name)
    print(f"loaded {model_name}")
    print(datetime.now() - begin_time)
    # Save:
    out_file = os.path.join("language_data", "english", f"{model_name}.bin")
    temp_model.save_word2vec_format(out_file, binary=True)
    print(datetime.now() - begin_time)
    print(f"saved {model_name}")

# %% Try it
m = load_language("glove-wiki-gigaword-50", None)
