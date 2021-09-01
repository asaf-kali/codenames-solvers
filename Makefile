LINE_LENGTH=120

init:
	make install
	make unzip_data

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt


# Linting

lint:
	black . -l $(LINE_LENGTH)
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	flake8 . --max-line-length=$(LINE_LENGTH) --exclude print_manager/migrations
	mypy . --ignore-missing-imports

unzip_data:
	echo "TODO"
