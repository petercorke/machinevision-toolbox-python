from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from machinevisiontoolbox.base import imageio


def test_window_title_is_forwarded():
    """Forward the title to a supported window manager."""
    manager = Mock()
    figure = SimpleNamespace(canvas=SimpleNamespace(manager=manager))
    with patch.object(imageio, "_plt") as pyplot:
        pyplot.return_value.gcf.return_value = figure
        imageio.set_window_title("camera")
    manager.set_window_title.assert_called_once_with("camera")


@pytest.mark.parametrize("error", [AttributeError, RuntimeError])
def test_unsupported_window_title_is_ignored(error):
    """Tolerate window managers that cannot set a title."""
    with patch.object(imageio, "_plt", side_effect=error("unsupported")):
        imageio.set_window_title("camera")


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_unexpected_window_title_errors_propagate(error):
    """Do not hide unrelated failures or user interrupts."""
    with patch.object(imageio, "_plt", side_effect=error()):
        with pytest.raises(error):
            imageio.set_window_title("camera")
