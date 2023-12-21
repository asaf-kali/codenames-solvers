# Codenames Solvers

Code infrastructure and player algorithms for the Codenames board game.\
This is the active fork of [mkali-personal/codenames](https://github.com/mkali-personal/codenames).

[![Tests](https://github.com/asaf-kali/codenames-solvers/actions/workflows/tests.yml/badge.svg)](https://github.com/asaf-kali/codenames-solvers/actions/workflows/tests.yml)
[![Lint](https://github.com/asaf-kali/codenames-solvers/actions/workflows/lint.yml/badge.svg)](https://github.com/asaf-kali/codenames-solvers/actions/workflows/lint.yml)

[//]: # ([![Video]&#40;https://github.com/asaf-kali/codenames-solvers/actions/workflows/video.yml/badge.svg&#41;]&#40;https://github.com/asaf-kali/codenames-solvers/actions/workflows/video.yml&#41;)

## Intro

The `solvers` module contains a multiple agent implementations, based on different strategies.

### Naive solver

Based on [Google's word2vec](https://code.google.com/archive/p/word2vec/) embedding.

*Hint generation*:
1. For each card group size `{4, 3, 2, 1}` from my unrevealed cards, collect hint proposal:
   1. Find the mean of the group's word embeddings.
   2. Find the closest word to this mean (this will be the hint word).
   3. Calculate the distance from this word to all other unrevealed cards on the board.
   4. Ensure the inspected word group is the closest to the proposed word.
   5. Ensure opponent cards, gray cards, and black card distance are greater than a specified threshold.
   6. Grade the hint proposal (based on number of cards in the group and distances to the word groups).
2. After collecting all hint proposal, pick the one with the highest grade.

*Guess generation*:
1. Given a hint and number of cards, calculate the distance between the hint and all unrevealed cards on the board.
2. Iterate on number of cards:
   1. Pick the closest card to the hint word.

### GPT solver

Based on OpenAI's ChatGPT API. Doesn't work very well.

## Installation

Clone this repository: `git clone https://github.com/asaf-kali/codenames-solvers`.

### Option 1: Run locally

1. Make sure you have [Poetry installed](https://python-poetry.org/docs/#installation) on your machine.
2. Create a virtual environment (Python >= 3.8).
3. Install dependencies using `make install` (or run the commands from the `Makefile`).
4. Get a language model: TODO.
5. Inside the `playground` directory, you will find different examples of how to use the solvers.

### Option 2: Use as a package

Currently, this project is not published on PyPI. \
From your project virtual env, install the package with `pip install -e <path_to_this_repo>`.

## Get a language model

### English

1. Download the [zip file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Extract the `GoogleNews-vectors-negative300.bin` file into the `language_data/`
   folder and rename it to `english.bin`.

### Hebrew

Look in the [GitHub repo](https://github.com/Ronshm/hebrew-word2vec).

## Export algorithm explanation video:
Run in terminal: `make video-render`.
If that didn't work: try `python -m manim videos/explanation.py KalirmozExplanation -pql\h`.
