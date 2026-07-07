"""allesfitter web GUI — a FastAPI shell over the allesfitter fitting engine.

The submodules are deliberately import-light: :mod:`config_writer`, :mod:`models`,
:mod:`instruments`, :mod:`staging`, :mod:`validate`, and :mod:`runstore` pull in
only the standard library plus the engine, so they can be used headless and
unit-tested without FastAPI installed. The web layer (:mod:`app`, :mod:`routes`,
:mod:`cli`) imports FastAPI/uvicorn lazily.

Install the optional web dependencies with ``uv sync --extra webgui`` (or
``pip install allesfitter[webgui]``).
"""

from allesfitter.webgui import instruments, models

__all__ = ["instruments", "models"]
