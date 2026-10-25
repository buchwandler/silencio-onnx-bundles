from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from silencio_onnx_bundles.manifest import sha256_file


class OnnxExportError(RuntimeError):
    pass


_ONNX_DTYPES = {
    "float32": "FLOAT",
    "int64": "INT64",
    "bool": "BOOL",
}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _output_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def validate_onnx_contract(
    onnx_path: Path,
    contract: Mapping[str, Any],
    *,
    opset: int,
) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(model)

    for direction in ("inputs", "outputs"):
        expected = contract[direction]
        actual_values = getattr(model.graph, "input" if direction == "inputs" else "output")
        actual = {value.name: value for value in actual_values}
        if set(actual) != set(expected):
            raise OnnxExportError(
                f"{contract['id']} {direction} {sorted(actual)} do not match {sorted(expected)}"
            )
        for name, spec in expected.items():
            value = actual[name]
            tensor = value.type.tensor_type
            expected_dtype = _ONNX_DTYPES[spec["dtype"]]
            actual_dtype = onnx.TensorProto.DataType.Name(tensor.elem_type)
            if actual_dtype != expected_dtype:
                raise OnnxExportError(
                    f"{contract['id']} {direction[:-1]} {name} has dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )
            actual_shape = list(tensor.shape.dim)
            expected_shape = spec["shape"]
            if len(actual_shape) != len(expected_shape):
                raise OnnxExportError(
                    f"{contract['id']} {direction[:-1]} {name} has rank {len(actual_shape)}, "
                    f"expected {len(expected_shape)}"
                )
            for axis, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape, strict=True)):
                if isinstance(expected_dim, str):
                    if actual_dim.dim_param != expected_dim:
                        raise OnnxExportError(
                            f"{contract['id']} {direction[:-1]} {name} axis {axis} is "
                            f"{actual_dim.dim_param or actual_dim.dim_value!r}, "
                            f"expected dynamic axis {expected_dim!r}"
                        )
                elif actual_dim.dim_value != expected_dim or actual_dim.dim_param:
                    raise OnnxExportError(
                        f"{contract['id']} {direction[:-1]} {name} axis {axis} is "
                        f"{actual_dim.dim_param or actual_dim.dim_value!r}, expected {expected_dim}"
                    )

    default_opset = {item.domain: item.version for item in model.opset_import}.get("")
    if default_opset != opset:
        raise OnnxExportError(
            f"{contract['id']} uses opset {default_opset}, expected {opset}"
        )


def compare_onnx_probe(
    wrapper: Any,
    onnx_path: Path,
    contract: Mapping[str, Any],
    args: Sequence[Any],
    session: Any | None = None,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, dict[str, float]]:

    import torch

    input_names = list(contract["inputs"])
    output_names = list(contract["outputs"])
    feed = {name: _numpy(value) for name, value in zip(input_names, args, strict=True)}
    if session is None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_values = session.run(output_names, feed)
    with torch.no_grad():
        expected_values = _output_tuple(wrapper(*args))
    if len(expected_values) != len(output_names):
        raise OnnxExportError(
            f"{contract['id']} eager wrapper returned {len(expected_values)} outputs, "
            f"expected {len(output_names)}"
        )

    metrics = {}
    for name, expected_value, actual_value in zip(
        output_names, expected_values, actual_values, strict=True
    ):
        expected = _numpy(expected_value)
        if expected.shape != actual_value.shape or expected.dtype != actual_value.dtype:
            raise OnnxExportError(
                f"{contract['id']} output {name} got {actual_value.dtype}{actual_value.shape}, "
                f"expected {expected.dtype}{expected.shape}"
            )
        if np.issubdtype(actual_value.dtype, np.floating):
            if not np.isfinite(actual_value).all():
                raise OnnxExportError(f"{contract['id']} output {name} contains non-finite values")
            error = np.abs(actual_value - expected)
            metrics[name] = {
                "max_abs_error": float(error.max(initial=0.0)),
                "mean_abs_error": float(error.mean()) if error.size else 0.0,
            }
            if not np.allclose(actual_value, expected, rtol=rtol, atol=atol):
                raise OnnxExportError(
                    f"{contract['id']} output {name} exceeds probe tolerance "
                    f"(max_abs={metrics[name]['max_abs_error']:.8g})"
                )
        elif not np.array_equal(actual_value, expected):
            raise OnnxExportError(f"{contract['id']} output {name} differs from the eager wrapper")
    return metrics


def _normalize_output_shapes(
    onnx_path: Path,
    wrapper: Any,
    args: Sequence[Any],
    contract: Mapping[str, Any],
) -> None:
    import onnx
    import torch

    with torch.no_grad():
        eager_outputs = _output_tuple(wrapper(*args))
    output_names = list(contract["outputs"])
    if len(eager_outputs) != len(output_names):
        raise OnnxExportError(
            f"{contract['id']} eager wrapper returned {len(eager_outputs)} outputs, "
            f"expected {len(output_names)}"
        )

    model = onnx.load(str(onnx_path), load_external_data=True)
    graph_outputs = {value.name: value for value in model.graph.output}
    for name, eager_value in zip(output_names, eager_outputs, strict=True):
        eager_shape = _numpy(eager_value).shape
        expected_shape = contract["outputs"][name]["shape"]
        if len(eager_shape) != len(expected_shape):
            raise OnnxExportError(f"{contract['id']} output {name} has unexpected eager rank")
        dimensions = graph_outputs[name].type.tensor_type.shape.dim
        for axis, expected_dim in enumerate(expected_shape):
            if isinstance(expected_dim, str):
                if contract["dynamic_axes"].get(name, {}).get(axis) != expected_dim:
                    raise OnnxExportError(
                        f"{contract['id']} output {name} axis {axis} is not declared dynamic"
                    )
                dimensions[axis].ClearField("dim_value")
                dimensions[axis].dim_param = expected_dim
            else:
                if eager_shape[axis] != expected_dim:
                    raise OnnxExportError(
                        f"{contract['id']} eager output {name} axis {axis} is "
                        f"{eager_shape[axis]}, expected {expected_dim}"
                    )
                dimensions[axis].ClearField("dim_param")
                dimensions[axis].dim_value = expected_dim
    onnx.save(model, str(onnx_path))


def export_component(
    wrapper: Any,
    args: Sequence[Any],
    out_path: Path,
    *,
    component: str,
    contract: Mapping[str, Any],
    opset: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    import torch

    input_names = list(contract["inputs"])
    output_names = list(contract["outputs"])
    if len(args) != len(input_names):
        raise OnnxExportError(
            f"{component} probe has {len(args)} inputs, expected {len(input_names)}"
        )
    wrapper.eval().requires_grad_(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(args),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=contract["dynamic_axes"],
            opset_version=opset,
            dynamo=False,
        )

    component_contract = {**contract, "id": component}
    _normalize_output_shapes(out_path, wrapper, args, component_contract)
    validate_onnx_contract(out_path, component_contract, opset=opset)
    probe_parity = compare_onnx_probe(
        wrapper, out_path, component_contract, args, rtol=rtol, atol=atol
    )
    dynamic_axes = {
        name: {str(axis): symbol for axis, symbol in axes.items()}
        for name, axes in contract["dynamic_axes"].items()
    }
    return {
        "id": component,
        "file": out_path.name,
        "opset": opset,
        "sha256": sha256_file(out_path),
        "size": out_path.stat().st_size,
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
        "dynamic_axes": dynamic_axes,
        "probe_parity": probe_parity,
    }
