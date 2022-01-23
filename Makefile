LINE_LENGTH=120

# Install

install-run:
	pip install --upgrade pip
	pip install -r requirements.txt

install-test:
	@make install-run --no-print-directory
	pip install -r requirements-dev.txt

install-dev:
	@make install-test --no-print-directory
	pre-commit install

install: install-dev

test:
	python -m pytest -s

# Video

video-install:
	make install
	sudo apt-get update
	sudo apt-get install ffmpeg pkg-config libcairo2-dev libpango1.0-dev --fix-missing
	pip install -r requirements-video.txt

video-render:
	python -m manim render videos/explanation.py --progress_bar display -p -f

# Lint

lint-only:
	black . -l $(LINE_LENGTH)
	isort . --profile black

lint-check:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --check --skip __init__.py
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH)

lint:
	@make lint-only --no-print-directory
	@make lint-check --no-print-directory

unzip_data:
	echo "TODO"
