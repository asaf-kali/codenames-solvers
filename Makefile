PYTHON_TEST_COMMAND=pytest
ifeq ($(OS),Windows_NT)
	OPEN_FILE_COMMAND=start
	DEL_COMMAND=del
else
	OPEN_FILE_COMMAND=xdg-open
	DEL_COMMAND=gio trash
endif
SYNC=--sync
.PHONY: build

# Install

upgrade-pip:
	pip install --upgrade pip

install-ci: upgrade-pip
	pip install poetry==1.5.1
	poetry config virtualenvs.create false

install-test:
	poetry install --all-extras --only main --only test

install-lint:
	poetry install --only lint

install-dev: upgrade-pip
	poetry install --all-extras --without video $(SYNC)
	pre-commit install

install-video:
	sudo apt-get update
	sudo apt-get install ffmpeg pkg-config libcairo2-dev libpango1.0-dev --fix-missing
	poetry install --only video

install: lock-check install-dev lint cover

# Poetry

lock:
	poetry lock --no-update

lock-check:
	poetry lock --check

# Test

test:
	python -m $(PYTHON_TEST_COMMAND)

cover:
	coverage run -m $(PYTHON_TEST_COMMAND)
	coverage html
	$(OPEN_FILE_COMMAND) htmlcov/index.html &
	$(DEL_COMMAND) .coverage*

# Packaging

build:
	$(DEL_COMMAND) -f dist/*
	poetry build

#upload:
#	twine upload dist/*
#
#upload-test:
#	twine upload --repository testpypi dist/*

# Video

video-render:
	python -m manim render videos/explanation.py --progress_bar display -p -f

gource:
	gource \
	--seconds-per-day 0.1 \
	-2560x1440 \
	--stop-at-end \
    --highlight-users \
    --hide mouse,filenames \
    --date-format "%b %d, %Y" \
    --file-idle-time 0 \
    --background-colour 000000 \
    --output-framerate 30 \
    --output-ppm-stream - \
	| ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i - -b 65536K gource.mp4

gource-all:
	bash ./videos/gource-all.sh \
	../the-spymaster-backend \
	../codenames \
	./ \
	../model-trainer \
	../resources \
	../the-spymaster-bot \
	../the-spymaster-infra \
	../the-spymaster-solvers \
	../the-spymaster-util

# Lint

format:
	ruff . --fix
	black .
	isort .

check-ruff:
	ruff .

check-black:
	black --check .

check-isort:
	isort --check .

check-mypy:
	mypy .

check-pylint:
	pylint solvers/ --fail-under=10

lint: format
	pre-commit run --all-files
	@make check-pylint --no-print-directory
