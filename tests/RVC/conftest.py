import os

# Ensure notebooks run via nbmake treat themselves as test mode,
# so cells guarded by %%skiptest (IS_TEST) are skipped.
os.environ["MVTB_TEST_MODE"] = "True"

# Use a non-interactive backend so %matplotlib widget doesn't hang headless.
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    """Set defaults for notebook execution in this test folder."""
    if not getattr(config.option, "nbmake", False):
        return

    # nbmake timeout is per-cell. Respect an explicit CLI value if provided.
    if (
        hasattr(config.option, "nbmake_timeout")
        and config.option.nbmake_timeout is None
    ):
        config.option.nbmake_timeout = 60
