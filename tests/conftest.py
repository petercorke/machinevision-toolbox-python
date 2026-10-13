import os

import pytest


def _real_gui_wanted() -> bool:
    """True when the tests should use real windows, not stubs.

    CI (GitHub Actions sets ``CI=true``) keeps real windows so that the OpenCV
    HighGUI calls are exercised on each OS and OpenCV version.  Locally, set
    ``MVTB_TEST_REAL_GUI=1`` to do the same.
    """
    ci = os.environ.get("CI", "").lower() not in ("", "0", "false")
    return ci or os.environ.get("MVTB_TEST_REAL_GUI") == "1"


# Keep local test runs headless by default.
if not _real_gui_wanted():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MVTB_TEST_MODE", "True")


@pytest.fixture(autouse=True)
def _suppress_gui(monkeypatch):
    """Disable GUI popups during tests.

    Many tests exercise display code paths; these patches keep behavior testable
    without opening OpenCV or Matplotlib windows.  They are skipped when real
    windows are wanted, see :func:`_real_gui_wanted`.
    """
    if _real_gui_wanted():
        return

    try:
        import cv2

        if hasattr(cv2, "namedWindow"):
            monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
        if hasattr(cv2, "imshow"):
            monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
        if hasattr(cv2, "waitKey"):
            monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: -1)
        if hasattr(cv2, "destroyWindow"):
            monkeypatch.setattr(cv2, "destroyWindow", lambda *args, **kwargs: None)
        if hasattr(cv2, "destroyAllWindows"):
            monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    except Exception:
        pass
