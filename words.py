import numpy as np

WORDS_NUMBER = 347222
VECTOR_LENGTH = 100

WORD_VECTORS = np.fromfile("words_vectors.npy")
with open("words.txt", encoding="utf-8") as words_file:
    WORDS = [word.replace("\n", "") for word in words_file.readlines()]
WORD_VECTORS = WORD_VECTORS[:-10].reshape(WORDS_NUMBER, VECTOR_LENGTH)
WORD_VECTORS = WORD_VECTORS / np.linalg.norm(WORD_VECTORS, axis=-1)[:, np.newaxis]
