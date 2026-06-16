.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test - run all unit tests"
	@echo " make test-log - run all unit tests and write log to pytest.log"
	@echo " make test-nb - run nbmake tests on RVC3 vision chapter notebooks"
	@echo " make coverage - run unit tests and coverage report"
	@echo " make typehints - run mypy type-hint coverage report and open HTML"
	@echo " make docs - build Sphinx documentation"
	@echo " make docs-nitpicky - build Sphinx docs with nitpicky cross-reference checks"
	@echo " make docs-strict - nitpicky Sphinx build with warnings treated as errors"
	@echo " make view - open Sphinx doco build (uses open for MacOS)"
	@echo " make dist - preview dist build"
	@echo " make clean - remove dist and docs build files"
	@echo " make help - this message$(BLACK)"

test:
	validate-pyproject pyproject.toml
	python -m pytest

test-log:
	python -m pytest -W all --tb=short 2>&1 | tee pytest.log
	@echo "pytest log --> pytest.log"

test-nb:
	python -m pytest --nbmake notebooks

test-rvc:
	MVTB_TEST_MODE=True python -m pytest --nbmake tests/RVC

coverage:
	coverage run --source='src/machinevisiontoolbox' -m pytest
	coverage report
	coverage html
	open -a Safari htmlcov/index.html

typehints:
	-mypy src/machinevisiontoolbox --ignore-missing-imports --html-report /tmp/mypy-typehints
	open -a Safari /tmp/mypy-typehints/index.html

docs: .FORCE
	(cd docs; make html O="-w warnings.txt")

docs-nitpicky: .FORCE
	conda run --no-capture-output -n dev sphinx-build -n -b html docs/source docs/build/html

docs-strict: .FORCE
	conda run --no-capture-output -n dev sphinx-build -n -W -b html docs/source docs/build/html

view: .FORCE
	open -a Safari docs/build/html/index.html

dist: .FORCE
	# $(MAKE) test
	python -m build
	ls -lh dist/*

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist build

