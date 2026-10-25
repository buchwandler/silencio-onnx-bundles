from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_release_spec(path: str | Path, key: str) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("Unsupported release catalog schema")
    try:
        return data["releases"][key]
    except KeyError as exc:
        raise KeyError(f"Unknown release key: {key}") from exc


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_snapshot_pin(snapshot: Path, spec: Mapping[str, Any]) -> str:
    """Check that the resolved snapshot matches an immutable release revision."""
    upstream = spec["upstream"]
    requested = str(upstream["revision"])
    resolved = snapshot.name

    if spec.get("redistribution") == "allowed" and not _SHA_RE.fullmatch(requested):
        raise ValueError("Public release upstream revision must be a full commit hash")
    if _SHA_RE.fullmatch(requested) and resolved != requested:
        raise ValueError(
            f"Resolved upstream revision {resolved} does not match configured pin {requested}"
        )
    return resolved
