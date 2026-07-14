"""Shared validation for user-selectable figure output formats."""

from __future__ import annotations

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".svg", ".webp"})


def normalize_file_extension(value: str = ".pdf") -> str:
    """Return a lowercase, leading-dot figure extension supported by Matplotlib."""
    extension = str(value).strip().lower()
    if not extension.startswith("."):
        extension = "." + extension
    if extension not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(f"unsupported figure file extension {value!r}; choose one of: {supported}")
    return extension
