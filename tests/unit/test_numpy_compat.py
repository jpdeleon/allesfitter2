"""Compatibility checks for NumPy warning classes moved in NumPy 2."""

import numpy as np


def test_numpy_warning_classes_resolve_on_installed_numpy():
    from allesfitter._numpy_compat import RankWarning, VisibleDeprecationWarning

    namespace = np.exceptions if hasattr(np, "exceptions") else np
    assert VisibleDeprecationWarning is namespace.VisibleDeprecationWarning
    assert RankWarning is namespace.RankWarning


def test_basement_imports_with_numpy_2():
    from allesfitter.basement import Basement

    assert Basement is not None
