"""The file picker only falls back for unavailable toolbox data."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from machinevisiontoolbox.base import imageio


@pytest.fixture
def picker(monkeypatch):
    dialog = Mock(return_value="selected.png")
    root = Mock()
    tk = SimpleNamespace(
        Tk=Mock(return_value=root), filedialog=SimpleNamespace(askopenfilename=dialog)
    )
    monkeypatch.setitem(__import__("sys").modules, "tkinter", tk)
    return dialog, root


@pytest.mark.parametrize(
    "error", [ModuleNotFoundError("mvtbdata"), ValueError("missing images")]
)
def test_missing_data_uses_home(monkeypatch, picker, error):
    monkeypatch.setattr(imageio, "mvtb_path_to_datafile", Mock(side_effect=error))
    dialog, root = picker
    assert imageio._pick_imagefile() == Path("selected.png")
    assert dialog.call_args.kwargs["initialdir"] == str(Path.home())
    root.destroy.assert_called_once()


def test_data_directory_is_used(monkeypatch, picker, tmp_path):
    monkeypatch.setattr(imageio, "mvtb_path_to_datafile", Mock(return_value=tmp_path))
    dialog, _ = picker
    assert imageio._pick_imagefile() == Path("selected.png")
    assert dialog.call_args.kwargs["initialdir"] == str(tmp_path)


@pytest.mark.parametrize(
    "error", [RuntimeError("broken resolver"), KeyboardInterrupt()]
)
def test_unexpected_resolver_errors_propagate(monkeypatch, picker, error):
    monkeypatch.setattr(imageio, "mvtb_path_to_datafile", Mock(side_effect=error))
    with pytest.raises(type(error)):
        imageio._pick_imagefile()
    picker[0].assert_not_called()
