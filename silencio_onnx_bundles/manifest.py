from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

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
