from __future__ import annotations

import hashlib
import json
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from scripts import build_bundle
from scripts.silencio_contracts import COMPONENT_CONTRACTS, COMPONENT_ORDER, OPSET_VERSION
from scripts.verify_candidate import CandidateVerificationError, verify_candidate
from silencio_onnx_bundles.manifest import sha256_file

_REVISION = "e200e2f9ef1887bb7b878045e0872d79a241f128"
_NP_DTYPES = {"float32": np.float32, "int64": np.int64, "bool": np.bool_}
_ONNX_TYPES = {"float32": onnx.TensorProto.FLOAT, "int64": onnx.TensorProto.INT64, "bool": onnx.TensorProto.BOOL}


def _onnx_bytes(component: str, *, wrong_input_dtype: bool = False) -> bytes:
    contract = COMPONENT_CONTRACTS[component]
    inputs = []
    for name, spec in contract["inputs"].items():
        dtype = spec["dtype"]
        if wrong_input_dtype and component == "text_encoder" and name == "phoneme_ids":
            dtype = "float32"
        inputs.append(helper.make_tensor_value_info(name, _ONNX_TYPES[dtype], spec["shape"]))
    outputs = []
    nodes = []
    for index, (name, spec) in enumerate(contract["outputs"].items()):
        outputs.append(
            helper.make_tensor_value_info(name, _ONNX_TYPES[spec["dtype"]], spec["shape"])
        )
        static_shape = [dim if isinstance(dim, int) else 1 for dim in spec["shape"]]
        value = numpy_helper.from_array(np.zeros(static_shape, dtype=_NP_DTYPES[spec["dtype"]]))
        nodes.append(helper.make_node("Constant", [], [name], name=f"constant_{index}", value=value))
    graph = helper.make_graph(nodes, component, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _fake_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / _REVISION
    snapshot.mkdir()
    (snapshot / "LICENSE").write_text("Pinned upstream notice.\n", encoding="utf-8")
    config = {
        "sample_rate": 16000,
        "default_voice": "librispeech_6930",
        "chunk_size": 12,
        "block_size": 4,
        "dit": {"dim": 512, "depth": 4, "heads": 2},
        "vocos": {
            "input_channels": 80,
            "dim": 320,
            "num_layers": 8,
            "n_fft": 640,
            "hop_length": 160,
            "padding": "center",
        },
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def _build_candidate(monkeypatch, tmp_path: Path) -> Path:
    snapshot = _fake_snapshot(tmp_path)
    monkeypatch.setattr(build_bundle, "snapshot_download", lambda **kwargs: snapshot)

    def fake_export(source: Path, bundle_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
        components = []
        for component in COMPONENT_ORDER:
            contract = COMPONENT_CONTRACTS[component]
            path = bundle_dir / contract["file"]
            path.write_bytes(_onnx_bytes(component))
            components.append(
                {
                    "id": component,
                    "file": path.name,
                    "opset": OPSET_VERSION,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "inputs": contract["inputs"],
                    "outputs": contract["outputs"],
                    "dynamic_axes": {
                        name: {str(axis): symbol for axis, symbol in axes.items()}
                        for name, axes in contract["dynamic_axes"].items()
                    },
                    "probe_parity": {
                        name: {"max_abs_error": 0.0, "mean_abs_error": 0.0}
                        for name in contract["outputs"]
                    },
                }
            )
        report = {
            "schema": 1,
            "status": "passed",
            "components": {
                name: {"status": "passed"}
                for name in (
                    "text_encoder",
                    "duration_expansion",
                    "text_prosody",
                    "adapter",
                    "condition_interpolation",
                    "dit_step",
                    "final_mel",
                    "vocos",
                )
            },
            "end_to_end": {"status": "passed"},
            "consumer_determinism": {"status": "passed"},
        }
        (bundle_dir / "parity-report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
        return {"components": components, "parity_report": report}

    monkeypatch.setattr(build_bundle, "export", fake_export)
    dist = tmp_path / "dist"
    build_bundle.main(["silencio-v1", "--dist", str(dist)])
    return dist / "silencio-v1-r1"


def _archive_items(candidate: Path) -> list[tuple[str, bytes]]:
    archive_path = candidate / "silencio-v1-r1.zip"
    with zipfile.ZipFile(archive_path) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _rewrite_archive(candidate: Path, items: list[tuple[str, bytes]]) -> None:
    archive_path = candidate / "silencio-v1-r1.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in items:
                archive.writestr(name, data)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        bundle = json.loads(archive.read("silencio-v1-r1/bundle.json"))
    manifest["licensing"] = bundle["licensing"]
    manifest["abi"] = bundle["abi"]
    manifest["parity"] = bundle["parity"]
    manifest["bundle"]["sha256"] = sha256_file(archive_path)
    manifest["bundle"]["size"] = archive_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (candidate / "SHA256SUMS").write_text(
        f"{manifest['bundle']['sha256']}  {archive_path.name}\n", encoding="utf-8"
    )


def _update_bundle(items: list[tuple[str, bytes]], update) -> list[tuple[str, bytes]]:
    changed = []
    for name, data in items:
        if name == "silencio-v1-r1/bundle.json":
            bundle = json.loads(data)
            update(bundle)
            data = (json.dumps(bundle, indent=2, ensure_ascii=False) + "\n").encode()
        changed.append((name, data))
    return changed


def _update_asset(bundle: dict[str, Any], file: str, data: bytes) -> None:
    record = next(asset for asset in bundle["assets"] if asset["file"] == file)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["size"] = len(data)
    if file == "parity-report.json":
        bundle["parity"]["sha256"] = record["sha256"]
        bundle["parity"]["size"] = record["size"]


def test_verifier_accepts_schema_valid_candidate_and_ort_loadable_components(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    verify_candidate(candidate)


def test_verifier_rejects_duplicate_zip_entries(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    items.append(items[0])
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="duplicate ZIP entry"):
        verify_candidate(candidate)


def test_verifier_rejects_zip_path_traversal(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    items.append(("silencio-v1-r1/../escape.txt", b"unsafe"))
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="unsafe ZIP path"):
        verify_candidate(candidate)


def test_verifier_rejects_missing_component_graph(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = [item for item in _archive_items(candidate) if not item[0].endswith("vocos.onnx")]
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="archive inventory mismatch"):
        verify_candidate(candidate)


def test_verifier_rejects_extra_graph(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    items.append(("silencio-v1-r1/extra.onnx", b"not an allowed graph"))
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="archive inventory mismatch"):
        verify_candidate(candidate)


def test_verifier_rejects_component_digest_mismatch(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _update_bundle(
        _archive_items(candidate),
        lambda bundle: bundle["components"][0].update({"sha256": "0" * 64}),
    )
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="component digest/size mismatch"):
        verify_candidate(candidate)


def test_verifier_rejects_onnx_dtype_contract_mismatch(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    graph_name = "silencio-v1-r1/text_encoder.onnx"
    graph = _onnx_bytes("text_encoder", wrong_input_dtype=True)
    items = [(name, graph if name == graph_name else data) for name, data in items]

    def refresh_component_digest(bundle):
        record = bundle["components"][0]
        record["sha256"] = hashlib.sha256(graph).hexdigest()
        record["size"] = len(graph)

    _rewrite_archive(candidate, _update_bundle(items, refresh_component_digest))
    with pytest.raises(CandidateVerificationError, match="has dtype FLOAT, expected INT64"):
        verify_candidate(candidate)


def test_verifier_rejects_missing_parity_report(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = [item for item in _archive_items(candidate) if not item[0].endswith("parity-report.json")]
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="archive inventory mismatch"):
        verify_candidate(candidate)


def test_verifier_rejects_failed_parity_report(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    report_name = "silencio-v1-r1/parity-report.json"
    report = json.loads(next(data for name, data in items if name == report_name))
    report["end_to_end"]["status"] = "failed"
    report_data = (json.dumps(report) + "\n").encode()
    items = [(name, report_data if name == report_name else data) for name, data in items]
    _rewrite_archive(
        candidate,
        _update_bundle(items, lambda bundle: _update_asset(bundle, "parity-report.json", report_data)),
    )
    with pytest.raises(CandidateVerificationError, match="end-to-end parity must be passed"):
        verify_candidate(candidate)


def test_verifier_rejects_invalid_upstream_provenance(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["upstream"]["resolved_revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CandidateVerificationError, match="upstream revision is not pinned"):
        verify_candidate(candidate)


def test_verifier_rejects_invalid_license_policy(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _update_bundle(
        _archive_items(candidate),
        lambda bundle: bundle["licensing"].update({"redistribution": "denied"}),
    )
    manifest_path = candidate / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["licensing"]["redistribution"] = "denied"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="license policy disallows redistribution"):
        verify_candidate(candidate)


def test_verifier_rejects_missing_required_notice(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _update_bundle(
        _archive_items(candidate),
        lambda bundle: bundle["licensing"].update(
            {"notice_files": [name for name in bundle["licensing"]["notice_files"] if name != "NOTICE"]}
        ),
    )
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="required notice files missing"):
        verify_candidate(candidate)


def test_verifier_rejects_checkpoint_leakage(monkeypatch, tmp_path):
    candidate = _build_candidate(monkeypatch, tmp_path)
    items = _archive_items(candidate)
    items.append(("silencio-v1-r1/training-checkpoint.safetensors", b"forbidden"))
    _rewrite_archive(candidate, items)
    with pytest.raises(CandidateVerificationError, match="checkpoint artifacts are forbidden"):
        verify_candidate(candidate)
