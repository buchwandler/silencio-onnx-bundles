from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

from .manifest import load_release_spec, sha256_file

__all__ = ["__version__", "sha256_file", "load_release_spec"]
