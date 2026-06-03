"""Single source of truth for the allesfitter2 version.

This module is intentionally dependency-free so that it can be imported
(or statically parsed by setuptools) without pulling in the rest of the
package. The version is exposed three ways that all read from here:

- ``allesfitter.__version__`` (re-exported in ``allesfitter/__init__.py``)
- the installed distribution metadata, via the ``[tool.setuptools.dynamic]``
  ``version = {attr = "allesfitter._version.__version__"}`` entry in
  ``pyproject.toml``
- ``importlib.metadata.version("allesfitter")`` once installed

Bump this string only; everything else follows automatically.
"""

from __future__ import annotations

__version__ = "1.2.10"
