"""``_isnotebook`` decides how ``Camera`` reuses its image plane, so it has to
recognise every environment a notebook runs in.  These use a fake ``get_ipython`` and
patched ``sys.modules`` / ``sys.platform``: no real Jupyter, Colab or JupyterLite
kernel is exercised."""

import builtins
import sys
import types

import pytest

from machinevisiontoolbox.base.imageio import _isnotebook


@pytest.fixture
def plain(monkeypatch):
    """Standard interpreter: no IPython, not Colab, not Pyodide"""
    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    return monkeypatch


def _shell(monkeypatch, name):
    shell = type(name, (), {})()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)


def test_a_standard_interpreter_is_not_a_notebook(plain):
    assert _isnotebook() is False


def test_a_jupyter_kernel_is_a_notebook(plain):
    _shell(plain, "ZMQInteractiveShell")

    assert _isnotebook() is True


def test_ipython_in_a_terminal_is_not_a_notebook(plain):
    _shell(plain, "TerminalInteractiveShell")

    assert _isnotebook() is False


def test_colab_is_a_notebook(plain):
    plain.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    _shell(plain, "Shell")

    assert _isnotebook() is True


def test_jupyterlite_is_a_notebook_whatever_its_shell_is_called(plain):
    plain.setattr(sys, "platform", "emscripten")
    _shell(plain, "SomeOtherShell")

    assert _isnotebook() is True


def test_an_unknown_shell_is_not_a_notebook(plain):
    _shell(plain, "Shell")  # not Colab: google.colab is not loaded

    assert _isnotebook() is False
