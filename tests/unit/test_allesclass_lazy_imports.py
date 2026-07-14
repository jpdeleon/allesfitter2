"""Regression coverage for dependencies deferred by the package import."""

from pathlib import Path


def test_allesclass_loads_dependencies_lazily():
    """Constructing allesclass must not depend on former eager imports."""
    import allesfitter

    datadir = Path(__file__).parents[2] / "examples" / "TOI-4645"
    alles = allesfitter.allesclass(datadir, quiet=True)

    assert alles.BASEMENT is not None
