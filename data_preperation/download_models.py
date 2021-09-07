# %% imports:
import gensim.downloader
from datetime import datetime
models_names = ['glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-300']
# %% Download and save:
begin_time = datetime.now()
for model_name in models_names:
    begin_time = datetime.now()
    # Download:
    temp_model = gensim.downloader.load(model_name)
    print(f"loaded {model_name}")
    print(datetime.now() - begin_time)
    # Save:
    temp_model.save_word2vec_format(f"language_data\\{model_name}.bin", binary=True)
    print(datetime.now() - begin_time)
    print(f"saved {model_name}")


