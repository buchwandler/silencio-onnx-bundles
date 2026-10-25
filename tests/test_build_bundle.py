import json
import zipfile
from pathlib import Path

import jsonschema
import pytest

from scripts import build_bundle
from scripts.silencio_contracts import COMPONENT_CONTRACTS, COMPONENT_ORDER, OPSET_VERSION
from silencio_onnx_bundles.manifest import sha256_file


def _fake_snapshot(tmp_path: Path) -> Path:
    revision = "e200e2f9ef1887bb7b878045e0872d79a241f128"
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "LICENSE").write_text("Pinned upstream license notice.\n", encoding="utf-8")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    return snapshot


def _configure_build(monkeypatch, tmp_path: Path, *, checkpoint: bool = False) -> Path:
    snapshot = _fake_snapshot(tmp_path)
    monkeypatch.setattr(build_bundle, "snapshot_download", lambda **kwargs: snapshot)

    def fake_export(source: Path, bundle_dir: Path, spec: dict) -> dict:
        components = []
        for component in COMPONENT_ORDER:
            contract = COMPONENT_CONTRACTS[component]
            path = bundle_dir / contract["file"]
            path.write_bytes(f"fake onnx bytes for {component}".encode())
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
        (bundle_dir / "parity-report.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
        if checkpoint:
            (bundle_dir / "training-checkpoint.safetensors").write_bytes(b"forbidden")
        return {"components": components, "parity_report": report}

    monkeypatch.setattr(build_bundle, "export", fake_export)
    return snapshot


def test_build_writes_schema_valid_metadata_and_packages_required_assets(monkeypatch, tmp_path):
    _configure_build(monkeypatch, tmp_path)
    dist = tmp_path / "dist"
    build_bundle.main(["silencio-v1", "--dist", str(dist)])

    candidate = dist / "silencio-v1-r1"
    bundle_dir = candidate / "bundle"
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("schemas/bundle.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(bundle, schema)

    assert bundle["schema"] == 2
    assert bundle["abi"]["component_order"] == list(COMPONENT_ORDER)
    assert bundle["runtime_semantics"]["cache_semantics"]["initial_frames"] == 0
    assert bundle["runtime_semantics"]["vocos"]["istft"]["output_samples"] == (
        "(mel_frames - 1) * hop_length"
    )
    assert bundle["parity"]["status"] == "passed"
    assert bundle["provenance"]["resolved_revision"] == (
        "e200e2f9ef1887bb7b878045e0872d79a241f128"
    )
    assert {component["id"] for component in bundle["components"]} == set(COMPONENT_ORDER)
    assert {"LICENSE", "MODEL_LICENSES.md", "NOTICE"}.issubset(
        bundle["licensing"]["notice_files"]
    )
    assert "upstream-notices/LICENSE" in bundle["licensing"]["upstream_notice_files"]

    with zipfile.ZipFile(candidate / "silencio-v1-r1.zip") as archive:
        names = set(archive.namelist())
        packaged = {name.removeprefix("silencio-v1-r1/") for name in names}
    assert "bundle.json" in packaged
    assert "parity-report.json" in packaged
    assert "LICENSE" in packaged
    assert "MODEL_LICENSES.md" in packaged
    assert "NOTICE" in packaged
    assert "upstream-notices/LICENSE" in packaged
    assert not any(Path(name).suffix in {".pt", ".pth", ".ckpt", ".safetensors"} for name in packaged)


def test_build_rejects_build_time_checkpoint_artifacts(monkeypatch, tmp_path):
    _configure_build(monkeypatch, tmp_path, checkpoint=True)
    with pytest.raises(RuntimeError, match="build-time checkpoint"):
        build_bundle.main(["silencio-v1", "--dist", str(tmp_path / "dist")])
