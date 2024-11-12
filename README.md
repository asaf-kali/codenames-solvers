# Codenames Solvers

[![Pipeline](https://github.com/asaf-kali/codenames-solvers/actions/workflows/pipeline.yml/badge.svg)](https://github.com/asaf-kali/codenames-solvers/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/asaf-kali/codenames-solvers/graph/badge.svg?token=IC3M4G19B6)](https://codecov.io/gh/asaf-kali/codenames-solvers)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-111111.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://img.shields.io/badge/type%20check-mypy-22aa11)](http://mypy-lang.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-22aa11)](https://github.com/pylint-dev/pylint)

Code infrastructure and player algorithms (solvers) for the Codenames board game. \
This is the active fork of [mkali-personal/codenames](https://github.com/mkali-personal/codenames).


[//]: # ([![Video]&#40;https://github.com/asaf-kali/codenames-solvers/actions/workflows/video.yml/badge.svg&#41;]&#40;https://github.com/asaf-kali/codenames-solvers/actions/workflows/video.yml&#41;)

## Intro

The `solvers` module contains a multiple agent implementations, based on different strategies. \
It is built above the [codenames](https://github.com/asaf-kali/codenames) package (which contains the basic game models
and logic definition), and serves as the brain
for [the-spymaster-bot](https://github.com/asaf-kali/the-spymaster-bot).

### Examples

Given the board:

```
+-------------+------------+----------+--------------+----------------+
|  ‎⬜ money   |  ‎🟦 drama  | ‎⬜ proof | ‎🟥 baseball  | ‎🟥 imagination |
+-------------+------------+----------+--------------+----------------+
|  ‎🟦 steel   |  ‎🟥 trail  | ‎⬜ giant |   ‎🟦 smell   |    ‎⬜ peace    |
+-------------+------------+----------+--------------+----------------+
|  ‎⬜ right   |  ‎⬜ pure   | ‎🟥 loud  | ‎💀 afternoon |  ‎🟥 constant   |
+-------------+------------+----------+--------------+----------------+
|  ‎🟥 fabric  | ‎⬜ violent | ‎🟥 style |  ‎🟦 musical  | ‎🟦 commitment  |
+-------------+------------+----------+--------------+----------------+
| ‎🟦 teaching | ‎🟦 africa  | ‎🟦 palm  |  ‎🟦 series   |    ‎🟥 bear     |
+-------------+------------+----------+--------------+----------------+
```

A `NaiveSpymaster` playing for the blue team will output `"role", 4`. \
From the logs:

```
Creating proposals for group size [4]...
Creating proposals for group size [3]...
Creating proposals for group size [2]...
Creating proposals for group size [1]...
Got 49 proposals.
Best 5 proposals:
('drama', 'musical', 'commitment', 'series') = ('role', 9.34)
('drama', 'musical', 'series') = ('films', 8.09)
('drama', 'musical', 'series') = ('comic', 8.04)
('drama', 'commitment', 'teaching') = ('focuses', 7.88)
('musical', 'commitment', 'teaching') = ('educational', 7.87)
Spymaster: [role] 4 card(s)
```

Some extra data from the solver about the picked hint:

```
{
  "word_group": ["drama", "musical", "commitment", "series"],
  "hint_word": "role",
  "hint_word_frequency": 0.999,
  "distance_group": 0.194,
  "distance_gray": 0.207,
  "distance_opponent": 0.23,
  "distance_black": 0.383,
  "grade": 9.337,
  "board_distances": {
    "drama": 0.151,
    "musical": 0.166,
    "commitment": 0.189,
    "series": 0.194,
    "peace": 0.207,
    ...
    "trail": 0.425,
    "smell": 0.451,
    "palm": 0.487
  }
}
```

### Usage

Find usage examples in the `playground` directory.

## Algorithm

### Naive solver

Based on [Google's word2vec](https://code.google.com/archive/p/word2vec/) embedding.

*Clue generation*:

1. For each card subset `group` of size `{4, 3, 2, 1}` from my unrevealed cards, collect hint proposal:
    1. Find the mean of `group`'s word embeddings.
    2. Find the closest word to this mean (this will be the `hint`).
    3. Calculate the distance from `hint` to all other unrevealed cards on the board.
    4. Ensure the inspected `group` is the closest to the proposed `hint`.
    5. Ensure opponent cards, gray cards, and black card distance to `hint` are **greater** than a specified threshold.
    6. Grade the `hint` proposal (based on the number of cards in `group` and the distances from `hint` to the different
       word groups).
2. After collecting all hint proposals:
    1. If no legal proposal was found, repeat step `1.` without filtering minimal distances (collect "dangerous" hints
       that might be confused with opponent cards).
    2. Otherwise, pick the proposal with the highest grade.

*Guess generation*:

1. Given a `hint` and number of cards, calculate the distance between `hint` and all unrevealed cards on the board.
2. Iterate on number of cards:
    1. Pick the closest card to `hint`.
3. Skip the extra guess.

### GPT solver

Based on OpenAI's ChatGPT API. Doesn't work very well.

## Installation

Clone this repository: `git clone https://github.com/asaf-kali/codenames-solvers`.

### Option 1: Run locally

1. Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation) on your machine.
2. Create a virtual environment (Python >= 3.9).
3. Install dependencies using `make install` (or run the commands from the `Makefile`).
4. Get a [language model](#get-a-language-model): TODO.
5. Inside the `playground` directory, you will find different examples of how to use the solvers.

### Option 2: Use as a package

Currently, this project is not published on PyPI. \
From your project virtual env, install the package with `pip install -e <path_to_this_repo>`.

## Get a language model

This needs to be updated.

### English

1. Download the [zip file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Extract the `GoogleNews-vectors-negative300.bin` file into the `language_data/`
   folder and rename it to `english.bin`.

### Hebrew

Look in the [GitHub repo](https://github.com/Ronshm/hebrew-word2vec).

## Export algorithm explanation video:

Run in terminal: `make video-render`.
If that didn't work: try `python -m manim videos/explanation.py KalirmozExplanation -pql\h`.
