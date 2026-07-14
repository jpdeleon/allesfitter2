import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 compatibility
    tomllib = pytest.importorskip("tomli")


def test_web_gui_dependencies_are_not_in_the_core_install():
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        project = tomllib.load(file)["project"]

    def dependency_name(requirement):
        return re.split(r"[<>=!~;@\[]", requirement, maxsplit=1)[0].strip().lower()

    core_names = {dependency_name(requirement) for requirement in project["dependencies"]}
    web_names = {
        dependency_name(requirement) for requirement in project["optional-dependencies"]["webgui"]
    }

    assert {"fastapi", "jinja2", "python-multipart", "pyyaml", "uvicorn"} <= web_names
    assert web_names.isdisjoint(core_names)
