.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo " make dist - build dist files"
	@echo " make upload - upload to PyPI"
	@echo " make clean - remove dist and docs build files"
	@echo " make help - this message$(BLACK)"


dist: .FORCE
	python -m build --wheel
	ls -lh dist/*

pypi: .FORCE
	python -m build
	$(eval VERSION := $(shell grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/'))
	@echo "Uploading version $(VERSION) to PyPI"
	twine upload --verbose dist/*

clean: .FORCE
	-rm -r *.egg-info
	-rm -r dist
