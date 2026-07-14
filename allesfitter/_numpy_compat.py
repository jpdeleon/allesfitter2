"""Small compatibility aliases for APIs moved by NumPy 2."""

from __future__ import annotations

import numpy as np

_warnings = np.exceptions if hasattr(np, "exceptions") else np

VisibleDeprecationWarning = _warnings.VisibleDeprecationWarning
RankWarning = _warnings.RankWarning
