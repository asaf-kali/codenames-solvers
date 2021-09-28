LINE_LENGTH=120

init:
	make install
	make tests

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

install-video:
	make install
	sudo apt-get install python3.9-dev pkg-config libcairo2-dev libpango1.0-dev
	pip install -r requirements-video.txt

tests:
	pytest

# Linting

lint:
	black . -l $(LINE_LENGTH)
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	flake8 . --max-line-length=$(LINE_LENGTH) --exclude codenames/old
	mypy . --ignore-missing-imports

unzip_data:
	echo "TODO"
