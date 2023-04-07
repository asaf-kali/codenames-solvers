PYTHON_TEST_COMMAND=pytest
DEL_COMMAND=gio trash
LINE_LENGTH=120
.PHONY: build

# Install

upgrade-pip:
	pip install --upgrade pip

install-run: upgrade-pip
	pip install -r requirements.txt

install-test:
	pip install -r requirements-dev.txt
	@make install-run --no-print-directory

install-lint:
	pip install -r requirements-lint.txt

install-dev: install-lint install-test
	pre-commit install

install: install-dev lint cover

# Test

test:
	python -m $(PYTHON_TEST_COMMAND)

cover:
	coverage run -m $(PYTHON_TEST_COMMAND)
	coverage html
	xdg-open htmlcov/index.html &
	$(DEL_COMMAND) .coverage*

# Packaging

build:
	$(DEL_COMMAND) -f dist/*
	python -m build

#upload:
#	twine upload dist/*
#
#upload-test:
#	twine upload --repository testpypi dist/*

# Video

video-install:
	make install
	sudo apt-get update
	sudo apt-get install ffmpeg pkg-config libcairo2-dev libpango1.0-dev --fix-missing
	pip install -r requirements-video.txt

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
	../the-spymaster-util \
	../model-trainer \
	../codenames \
	./ \
	../the-spymaster-bot \
	../the-spymaster-backend \
	../the-spymaster-solvers

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
	pylint solvers/ --fail-under=9.5

lint: format
	pre-commit run --all-files
	@make check-pylint --no-print-directory
