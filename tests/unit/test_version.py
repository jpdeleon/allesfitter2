"""Guard the single-source version automation.

``allesfitter/_version.py`` is the one place the version is written. These
tests assert that the runtime attribute and the module agree, that the string
is PEP 440-ish, and — when the package is installed — that the distribution
metadata (driven by ``[tool.setuptools.dynamic]`` in pyproject) matches too.
A drift here means someone hard-coded a version somewhere again.
"""

from __future__ import annotations

import re

import pytest

import allesfitter
from allesfitter import _version


def test_runtime_version_comes_from_single_source():
    assert allesfitter.__version__ == _version.__version__


def test_version_string_is_wellformed():
    # e.g. "1.2.10" — digits and dots, at least major.minor.
    assert re.fullmatch(r"\d+(\.\d+)+", _version.__version__), _version.__version__


def test_distribution_metadata_matches_when_installed():
    from importlib.metadata import PackageNotFoundError, version

    try:
        dist = version("allesfitter")
    except PackageNotFoundError:
        pytest.skip("allesfitter is not installed as a distribution")
    assert dist == _version.__version__
