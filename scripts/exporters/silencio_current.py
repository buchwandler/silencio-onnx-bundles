from __future__ import annotations

from pathlib import Path
from typing import Any


def export(*, snapshot: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Pinned adapter boundary for the first Silencio ONNX export.

    This starter deliberately refuses to fabricate tensor names/shapes. Implement this
    function against a pinned BlueRoseLabs/Silencio revision, producing the component
    files below and returning their contracts.
    """
    expected = ["text_encoder.onnx", "adapter.onnx", "dit_step.onnx", "vocos.onnx"]
    source_hint = snapshot / "silencio"
    raise RuntimeError(
        "Exporter contract not implemented yet. Inspect the pinned upstream source at "
        f"{source_hint}. Expected MVP component set: {', '.join(expected)}. "
        "Keep G2P, sentence/chunk scheduling and the 3-step solver in the host runtime."
    )
