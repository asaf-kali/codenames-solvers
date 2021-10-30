# Codenames

[![Tests](https://github.com/mkali-personal/codenames/actions/workflows/tests.yml/badge.svg)](https://github.com/mkali-personal/codenames/actions/workflows/tests.yml)
[![Lint](https://github.com/mkali-personal/codenames/actions/workflows/lint.yml/badge.svg)](https://github.com/mkali-personal/codenames/actions/workflows/lint.yml)
[![Video](https://github.com/mkali-personal/codenames/actions/workflows/video.yml/badge.svg)](https://github.com/mkali-personal/codenames/actions/workflows/video.yml)

Codenames board game player algorithms and game infrastructure.

## Intro

* This is based on [Google's word2vec algorithm](https://code.google.com/archive/p/word2vec/).
* [Planning document](https://docs.google.com/presentation/d/1RBwIRRtiqs30q3cF3HOAIZLEH6HoPZ_lY_x7SrbBfrc/edit#slide=id.p)
  .
* [Online gaming platform]().

## Installation

1. Create a virtual environment.
2. Install dependencies using `make install` (or run the same command from the `Makefile`).

**After that**:

* To use the online [Namecoding](https://namecoding.herokuapp.com/) adapter:
    1. Download Selenium's [Google Chrome adapter](https://selenium-python.readthedocs.io/installation.html#drivers)
       (mind your OS version and Google Chrome version!).
    2. Put the `chromedriver` file in `./codenames/online`.
* Follow the `Get the data` section in this file to download language files.

## Get the data

### English

1. Download the [zip file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Extract the `GoogleNews-vectors-negative300.bin` file into the `language_data/`
   folder and rename it to `english.bin`.

### Hebrew

Look in the [GitHub repo](https://github.com/Ronshm/hebrew-word2vec).

### How to work

1. `git checkout main`
2. `git pull`
3. `git checkout [-b] <my_branch>`
4. `git merge main`
5. Work work work...
6. `git commit -m "Message"`
7. `git push`

### How to export explanation video:
Run in terminal: `python -m manim videos/explanation.py KalirmozExplanation -pql\h`
