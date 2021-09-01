from gensim import models


# %%


class Game:
    def __init__(self):
        self.model = models.KeyedVectors.load_word2vec_format(
            r"DataFiles/GoogleNews-vectors-negative300.bin", binary=True
        )
        self.n = len(self.model.index_to_key)
        self.board_size = 25

        self.board_words_indices = None
        self.board = None

        # self.pick_words()

    def pick_words(self):
        # Randomize words and create an attribute of board with word-vectors pairs
        # self.board_words_indices = random.sample(range(int(self.n / 1000), int(self.n / 2)), self.board_size)
        # words = self.model.index_to_key[self.board_words_indices]
        words = [
            "cloak",
            "kiss",
            "flood",
            "mail",
            "skates",
            "paper",
            "frog",
            "skyscrapper",
            "moon",
            "egypt",
            "teacher",
            "avalance",
            "newton",
            "violet",
            "drill",
            "fever",
            "ninja",
            "jupyter",
            "ski",
            "attic",
            "beach",
            "lock",
            "earth",
            "park",
            "gymnast",
        ]

        vectors = self.model.vectors[self.board_words_indices]

        zipped = zip(words, vectors)
        self.board = dict(zipped)


# %%

c = Game()
