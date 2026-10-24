from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


class ExportAdapterError(RuntimeError):
    pass


def _inventory(snapshot: Path) -> dict[str, Any]:
    py = sorted(str(p.relative_to(snapshot)) for p in snapshot.rglob("*.py"))
    return {"python_files": py, "has_silencio_package": (snapshot / "silencio").is_dir()}


def _load_adapter(name: str):
    module_name = f"scripts.exporters.{name.replace('-', '_')}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ExportAdapterError(f"Unknown exporter adapter: {name}") from exc
    return module


def export(snapshot: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(snapshot))
    adapter_name = str(spec.get("exporter", "silencio-current"))
    adapter = _load_adapter(adapter_name)
    try:
        return adapter.export(snapshot=snapshot, out_dir=out_dir, spec=spec)
    except Exception as exc:
        inventory = _inventory(snapshot)
        (out_dir / "upstream-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        raise ExportAdapterError(
            f"Silencio exporter adapter {adapter_name!r} failed. "
            f"Upstream inventory was written to {out_dir / 'upstream-inventory.json'}. "
            "Pin the upstream revision and update scripts/exporters/silencio_current.py "
            "to match its actual tensor contract."
        ) from exc
