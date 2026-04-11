.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test - run all unit tests"
	@echo " make test-log - run all unit tests and write log to pytest.log"
	@echo " make test-nb - run nbmake tests on RVC3 vision chapter notebooks"
	@echo " make coverage - run unit tests and coverage report"
	@echo " make docs - build Sphinx documentation"
	@echo " make view - open Sphinx doco build (uses open for MacOS)"
	@echo " make dist - preview dist build"
	@echo " make pypi - build the dist, upload to PyPI, tag the release and push the tag to origin"
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

docs: .FORCE
	(cd docs; make html O="-w warnings.txt")

view: .FORCE
	open -a Safari docs/build/html/index.html

dist: .FORCE
	# $(MAKE) test
	python -m build
	ls -lh dist/*

pypi: .FORCE
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Error: uncommitted changes present, aborting upload"; exit 1; \
	fi
	python -m build
	$(eval VERSION := $(shell grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/'))
	@echo "Uploading version $(VERSION) to PyPI"
	twine upload dist/*
	git tag v$(VERSION)
	git push origin v$(VERSION)

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist build

