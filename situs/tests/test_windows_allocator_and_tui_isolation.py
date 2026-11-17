"""Tests verifying platform-safe PyTorch CUDA allocator configuration,
terminal stderr & warning isolation in TUI mode, and full-screen screen refresh/redraw commands."""

import os
import sys
import warnings
import pytest
from unittest.mock import patch, MagicMock
from prompt_toolkit.document import Document

from situs.worker.engine import configure_pytorch_allocator
from situs.tui import (
    TuiStderrRedirector,
    OpenCodeSlashCompleter,
    execute_slash_command,
)


def test_windows_allocator_config_removes_expandable_segments():
    """Verify configure_pytorch_allocator on Windows strips expandable_segments:True
    (which is unsupported by WDDM and triggers UserWarning in CUDAAllocatorConfig.h)
    and configures max_split_size_mb:128 instead."""
    orig = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        with patch("sys.platform", "win32"):
            configure_pytorch_allocator()
            conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            assert "expandable_segments" not in conf
            assert "max_split_size_mb:128" in conf
            assert "garbage_collection_threshold:0.8" in conf
    finally:
        if orig is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = orig
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def test_linux_allocator_config_preserves_expandable_segments():
    """Verify configure_pytorch_allocator on Linux/POSIX enables expandable_segments:True."""
    orig = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
        with patch("sys.platform", "linux"):
            configure_pytorch_allocator()
            conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            assert "expandable_segments:True" in conf
    finally:
        if orig is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = orig
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def test_tui_stderr_redirector():
    """Verify that TuiStderrRedirector routes writes to logger.warning instead of the console."""
    mock_logger = MagicMock()
    redirector = TuiStderrRedirector(mock_logger)

    # Empty write should be a no-op
    redirector.write("")
    assert not mock_logger.warning.called

    # Write full line
    redirector.write("CUDA memory allocation failed on device 0\n")
    mock_logger.warning.assert_called_with("[stderr] CUDA memory allocation failed on device 0")

    # Partial write followed by flush
    mock_logger.reset_mock()
    redirector.write("Incomplete message")
    assert not mock_logger.warning.called
    redirector.flush()
    mock_logger.warning.assert_called_with("[stderr] Incomplete message")


def test_warning_filters_drop_expandable_segments_warning():
    """Verify warnings filter drops expandable_segments UserWarning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.filterwarnings("ignore", message=".*expandable_segments not supported on this platform.*")
        warnings.warn("expandable_segments not supported on this platform", UserWarning)
        assert len(recorded) == 0


def test_tui_slash_refresh_and_redraw_completer():
    """Verify /refresh and /redraw are available in the TUI slash command autocompleter."""
    completer = OpenCodeSlashCompleter()
    doc_ref = Document("/ref")
    completions = [c.text for c in completer.get_completions(doc_ref, None)]
    assert "/refresh" in completions

    doc_red = Document("/red")
    red_completions = [c.text for c in completer.get_completions(doc_red, None)]
    assert "/redraw" in red_completions


def test_tui_slash_refresh_and_redraw_execution():
    """Verify /refresh and /redraw trigger REDRAW in log_output_fn and return True."""
    messages = []
    session_state = {}
    captured = []

    def log_capture(title, renderable):
        captured.append((title, renderable))

    res1 = execute_slash_command("/refresh", "http://127.0.0.1:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert res1 is True
    assert ("REDRAW", None) in captured

    captured.clear()
    res2 = execute_slash_command("/redraw", "http://127.0.0.1:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert res2 is True
    assert ("REDRAW", None) in captured
