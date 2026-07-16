"""Figure-extension API regression tests."""

import inspect

import pytest

from allesfitter._figure_output import normalize_file_extension
from allesfitter.general_output import show_initial_guess
from allesfitter.mcmc_output import mcmc_output
from allesfitter.nested_sampling_output import ns_output


@pytest.mark.parametrize(("value", "expected"), [("png", ".png"), (".PDF", ".pdf")])
def test_normalize_file_extension(value, expected):
    assert normalize_file_extension(value) == expected


def test_rejects_unsupported_file_extension():
    with pytest.raises(ValueError, match="unsupported figure file extension"):
        normalize_file_extension("exe")


@pytest.mark.parametrize("function", [show_initial_guess, mcmc_output, ns_output])
def test_terminal_figure_default_remains_pdf(function):
    assert inspect.signature(function).parameters["file_extension"].default == ".pdf"
