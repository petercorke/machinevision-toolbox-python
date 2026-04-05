# RVC3 tests

These are notebooks for the vision chapters (10-16) of the RVC3-python book, distributed 
as part of the `RVC3` package.

They are the ultimate regression test for the toolbox, the contract that any changes
don't affect the published code examples.  There are no tests of results so this
function more as a _smoke test_.

The tests can be run by `pytest --nbmake` and `nbmake` needs to be installed.

Some cells can't work in a test enironment:

* require interaction with graphics
* require bdsim

These cells are kept out of the test by the magic command `%%skiptest` which skips the cell if the envariable `MVTB_TEST_MODE` is set.

Each notebook has modified logic at the top to create the magic command and check the
envariable which is set within `conftest.py` (the configuration for `nbmake`).
