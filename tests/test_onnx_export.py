from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from scripts.onnx_export import (
    OnnxExportError,
    export_component,
    validate_onnx_contract,
)


class TinyDynamicGraph(nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return features * 2, features.mean(dim=-1, keepdim=True)


@pytest.fixture
def tiny_contract() -> dict[str, Any]:
    return {
        "file": "tiny.onnx",
        "inputs": {"features": {"dtype": "float32", "shape": [1, "frames", 2]}},
        "outputs": {
            "scaled": {"dtype": "float32", "shape": [1, "frames", 2]},
            "mean": {"dtype": "float32", "shape": [1, "frames", 1]},
        },
        "dynamic_axes": {
            "features": {1: "frames"},
            "scaled": {1: "frames"},
            "mean": {1: "frames"},
        },
    }


def test_export_component_checks_contract_and_dynamic_runtime(
    tmp_path: Path, tiny_contract: dict[str, Any]
) -> None:
    probe = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    path = tmp_path / "tiny.onnx"
    record = export_component(
        TinyDynamicGraph(),
        (probe,),
        path,
        component="tiny",
        contract=tiny_contract,
        opset=17,
    )

    assert record["sha256"]
    assert record["size"] == path.stat().st_size
    assert record["dynamic_axes"]["features"] == {"1": "frames"}
    assert set(record["probe_parity"]) == {"scaled", "mean"}

    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    longer = np.arange(14, dtype=np.float32).reshape(1, 7, 2)
    scaled, mean = session.run(None, {"features": longer})
    np.testing.assert_array_equal(scaled, longer * 2)
    np.testing.assert_array_equal(mean, longer.mean(axis=-1, keepdims=True))


def test_validate_onnx_contract_rejects_static_shape_drift(
    tmp_path: Path, tiny_contract: dict[str, Any]
) -> None:
    probe = torch.zeros((1, 3, 2), dtype=torch.float32)
    path = tmp_path / "tiny.onnx"
    export_component(
        TinyDynamicGraph(),
        (probe,),
        path,
        component="tiny",
        contract=tiny_contract,
        opset=17,
    )
    bad_contract = {
        **tiny_contract,
        "id": "tiny",
        "inputs": {
            "features": {"dtype": "float32", "shape": [1, 3, 2]}
        },
    }

    with pytest.raises(OnnxExportError, match="axis 1"):
        validate_onnx_contract(path, bad_contract, opset=17)
