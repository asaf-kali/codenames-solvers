LINE_LENGTH=120

init:
	make install
	make tests

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

tests:
	pytest

# Video

video-install:
	make install
	sudo apt-get update
	sudo apt-get install ffmpeg pkg-config libcairo2-dev libpango1.0-dev --fix-missing
	pip install -r requirements-video.txt

video-render:
	python -m manim render videos/explanation.py --progress_bar display -p -f

# Linting

lint:
	black . -l $(LINE_LENGTH)
	isort . --profile black
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --check
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH) --exclude codenames/old

unzip_data:
	echo "TODO"
