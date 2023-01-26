PYTHON_TEST_COMMAND=pytest
DEL_COMMAND=gio trash
LINE_LENGTH=120
.PHONY: build

# Install

install-run:
	pip install --upgrade pip
	pip install -r requirements.txt

install-test:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	@make install-run --no-print-directory

install-dev:
	@make install-test --no-print-directory
	pre-commit install

install: install-dev test lint

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
	black .
	isort .

lint-check:
	black . --check
	isort . --check
	mypy .
	flake8 . --max-line-length=$(LINE_LENGTH) --ignore=E203,W503

lint: format
	pre-commit run --all-files

unzip_data:
	echo "TODO"
