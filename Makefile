PYTHON_TEST_COMMAND=pytest
DEL_COMMAND=gio trash
LINE_LENGTH=120
.PHONY: build

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

install: install-dev test

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
	$(DEL_COMMAND) -f dist/
	python -m build

upload:
	twine upload dist/*

upload-test:
	twine upload --repository testpypi dist/*

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
	black .
	isort .

lint-check:
	black . --check
	isort . --check
	mypy .
	flake8 . --max-line-length=$(LINE_LENGTH) --ignore=E203,W503

lint: lint-only
	pre-commit run --all-files

unzip_data:
	echo "TODO"
