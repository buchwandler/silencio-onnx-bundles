from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.silencio_contracts import (  # noqa: E402
    COMPONENT_CONTRACTS,
    COMPONENT_ORDER,
    OPSET_VERSION,
)
from silencio_onnx_bundles.manifest import load_release_spec, validate_snapshot_pin  # noqa: E402

_REQUIRED_PARITY_COMPONENTS = {
    "text_encoder",
    "duration_expansion",
    "text_prosody",
    "adapter",
    "condition_interpolation",
    "dit_step",
    "final_mel",
    "vocos",
}
_REQUIRED_NOTICE_FILES = {"LICENSE", "MODEL_LICENSES.md", "NOTICE"}
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth", ".safetensors"}
_ONNX_DTYPES = {"float32": "FLOAT", "int64": "INT64", "bool": "BOOL"}


class CandidateVerificationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateVerificationError(message)


def _read_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        result = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateVerificationError(f"{label} is not valid JSON") from exc
    _require(isinstance(result, dict), f"{label} must be a JSON object")
    return result


def _safe_bundle_path(name: str, *, tag: str | None = None) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in name
        or any(":" in part for part in path.parts)
        or (tag is not None and (path.parts[0] != tag or len(path.parts) < 2))
    ):
        raise CandidateVerificationError(f"unsafe ZIP path: {name!r}")
    return path


def _check_zip_entries(archive: zipfile.ZipFile, tag: str) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    infos = archive.infolist()
    names = [item.filename for item in infos]
    _require(len(names) == len(set(names)), "duplicate ZIP entry")
    for info in infos:
        path = _safe_bundle_path(info.filename, tag=tag)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        _require(not info.is_dir(), f"unexpected ZIP directory: {info.filename}")
        _require(file_type in {0, stat.S_IFREG}, f"non-regular ZIP entry: {info.filename}")
        _require(
            Path(path.name).suffix.lower() not in _CHECKPOINT_SUFFIXES
            and "checkpoint" not in path.name.lower(),
            f"checkpoint artifacts are forbidden: {info.filename}",
        )
    bad_member = archive.testzip()
    _require(bad_member is None, f"ZIP CRC check failed for {bad_member}")
    for name in names:
        entries[name] = archive.read(name)
    return entries


def _validate_component_record(record: dict[str, Any], component_id: str) -> None:
    contract = COMPONENT_CONTRACTS[component_id]
    _require(record.get("id") == component_id, f"component order mismatch at {component_id}")
    _require(record.get("file") == contract["file"], f"unexpected file for {component_id}")
    _require(record.get("opset") == OPSET_VERSION, f"unexpected opset for {component_id}")
    _require(record.get("inputs") == contract["inputs"], f"input ABI mismatch for {component_id}")
    _require(record.get("outputs") == contract["outputs"], f"output ABI mismatch for {component_id}")
    expected_axes = {
        name: {str(axis): symbol for axis, symbol in axes.items()}
        for name, axes in contract["dynamic_axes"].items()
    }
    _require(record.get("dynamic_axes") == expected_axes, f"dynamic axes mismatch for {component_id}")
    _require(
        set(record.get("probe_parity", {})) == set(contract["outputs"]),
        f"probe parity outputs mismatch for {component_id}",
    )


def _check_value_info(model: Any, component_id: str, direction: str, expected: dict[str, Any]) -> None:
    values = getattr(model.graph, "input" if direction == "inputs" else "output")
    actual = {value.name: value for value in values}
    _require(
        set(actual) == set(expected),
        f"ONNX {component_id} {direction} {sorted(actual)} do not match {sorted(expected)}",
    )
    import onnx

    for name, spec in expected.items():
        value = actual[name]
        tensor = value.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(tensor.elem_type)
        wanted_dtype = _ONNX_DTYPES[spec["dtype"]]
        _require(
            dtype == wanted_dtype,
            f"ONNX {component_id} {direction[:-1]} {name} has dtype {dtype}, expected {wanted_dtype}",
        )
        actual_shape = list(tensor.shape.dim)
        expected_shape = spec["shape"]
        _require(
            len(actual_shape) == len(expected_shape),
            f"ONNX {component_id} {direction[:-1]} {name} has wrong rank",
        )
        for axis, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape, strict=True)):
            if isinstance(expected_dim, str):
                _require(
                    actual_dim.dim_param == expected_dim,
                    f"ONNX {component_id} {direction[:-1]} {name} axis {axis} is "
                    f"{actual_dim.dim_param or actual_dim.dim_value!r}, expected dynamic axis {expected_dim!r}",
                )
            else:
                _require(
                    actual_dim.dim_value == expected_dim and not actual_dim.dim_param,
                    f"ONNX {component_id} {direction[:-1]} {name} axis {axis} has wrong dimension",
                )


def _validate_onnx(component_id: str, record: dict[str, Any], data: bytes) -> None:
    import onnx
    import onnxruntime as ort

    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise CandidateVerificationError(f"ONNX {component_id} failed checker: {exc}") from exc
    _check_value_info(model, component_id, "inputs", COMPONENT_CONTRACTS[component_id]["inputs"])
    _check_value_info(model, component_id, "outputs", COMPONENT_CONTRACTS[component_id]["outputs"])
    default_opset = {item.domain: item.version for item in model.opset_import}.get("")
    _require(default_opset == OPSET_VERSION, f"ONNX {component_id} uses unexpected opset {default_opset}")
    try:
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise CandidateVerificationError(f"ONNX Runtime cannot load {component_id}: {exc}") from exc
    _require(
        {item.name for item in session.get_inputs()} == set(record["inputs"]),
        f"ONNX Runtime input names mismatch for {component_id}",
    )
    _require(
        {item.name for item in session.get_outputs()} == set(record["outputs"]),
        f"ONNX Runtime output names mismatch for {component_id}",
    )


def _validate_parity(bundle: dict[str, Any], entries: dict[str, bytes], tag: str) -> None:
    parity = bundle["parity"]
    path = parity["file"]
    _require(path == "parity-report.json", "unexpected parity report path")
    report_data = entries.get(f"{tag}/{path}")
    _require(report_data is not None, "parity report missing from archive")
    report = _read_json_bytes(report_data, "parity report")
    _require(parity["status"] == "passed", "bundle parity status must be passed")
    _require(parity["consumer_determinism"] == "passed", "consumer determinism must be passed")
    _require(report.get("status") == "passed", "parity report status must be passed")
    _require(
        report.get("consumer_determinism", {}).get("status") == "passed",
        "parity report consumer determinism must be passed",
    )
    _require(report.get("end_to_end", {}).get("status") == "passed", "end-to-end parity must be passed")
    components = report.get("components", {})
    _require(
        _REQUIRED_PARITY_COMPONENTS.issubset(components),
        "parity report is missing required component stages",
    )
    _require(
        all(components[name].get("status") == "passed" for name in _REQUIRED_PARITY_COMPONENTS),
        "one or more parity component stages failed",
    )


def _validate_assets(bundle: dict[str, Any], entries: dict[str, bytes], tag: str) -> set[str]:
    assets = bundle["assets"]
    asset_names = [asset["file"] for asset in assets]
    _require(len(asset_names) == len(set(asset_names)), "duplicate bundle asset record")
    notice_names = set(bundle["licensing"]["notice_files"])
    upstream_names = set(bundle["licensing"]["upstream_notice_files"])
    _require(_REQUIRED_NOTICE_FILES.issubset(notice_names), "required notice files missing")
    _require(
        upstream_names == set(bundle["provenance"]["upstream_notice_files"]),
        "upstream notice provenance mismatch",
    )
    _require(
        all(name.startswith("upstream-notices/") for name in upstream_names),
        "upstream notice path is invalid",
    )
    _require(upstream_names.issubset(notice_names), "upstream notices are absent from notice inventory")
    _require(
        bundle["license"] == bundle["licensing"]["spdx"],
        "license SPDX metadata mismatch",
    )
    _require(bundle["licensing"]["redistribution"] == "allowed", "license policy disallows redistribution")

    parity_assets = [asset for asset in assets if asset["kind"] == "parity_report"]
    _require(
        len(parity_assets) == 1 and parity_assets[0]["file"] == "parity-report.json",
        "parity report asset record missing or duplicated",
    )
    for asset in assets:
        name = asset["file"]
        _safe_bundle_path(name)
        _require(f"{tag}/{name}" in entries, f"bundle asset is missing: {name}")
        data = entries[f"{tag}/{name}"]
        digest = hashlib.sha256(data).hexdigest()
        _require(
            asset["sha256"] == digest and asset["size"] == len(data),
            f"bundle asset digest/size mismatch: {name}",
        )
        if asset["kind"] == "parity_report":
            _require(name == bundle["parity"]["file"], "parity asset path mismatch")
        else:
            _require(asset["kind"] in {"license_notice", "upstream_license_notice"}, "unknown asset kind")
            _require(name in notice_names, f"license notice is not declared: {name}")
            _require(
                (asset["kind"] == "upstream_license_notice") == (name in upstream_names),
                f"license notice kind mismatch: {name}",
            )
    asset_notice_names = {
        asset["file"] for asset in assets if asset["kind"] in {"license_notice", "upstream_license_notice"}
    }
    _require(asset_notice_names == notice_names, "notice asset inventory mismatch")
    return set(asset_names)


def _validate_release_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema",
        "release_key",
        "tag",
        "model_id",
        "model_version",
        "release_version",
        "sample_rate",
        "runtime",
        "frontend",
        "license",
        "redistribution",
        "licensing",
        "abi",
        "parity",
        "upstream",
        "bundle",
    }
    _require(required.issubset(manifest), f"release manifest is missing fields: {sorted(required - manifest.keys())}")
    for name in ("release_key", "tag", "model_id", "model_version", "license"):
        _require(isinstance(manifest[name], str) and bool(manifest[name]), f"release manifest {name} is invalid")
    release_version = manifest["release_version"]
    _require(
        (isinstance(release_version, str) and bool(release_version))
        or (isinstance(release_version, int) and not isinstance(release_version, bool) and release_version > 0),
        "release manifest release_version is invalid",
    )
    _require(isinstance(manifest["sample_rate"], int) and manifest["sample_rate"] > 0, "release sample rate is invalid")
    for name in ("runtime", "frontend", "licensing", "abi", "parity", "upstream", "bundle"):
        _require(isinstance(manifest[name], dict), f"release manifest {name} must be an object")
    upstream = manifest["upstream"]
    _require(
        all(isinstance(upstream.get(name), str) and upstream[name] for name in ("repo", "requested_revision", "resolved_revision")),
        "release upstream provenance is invalid",
    )
    bundle = manifest["bundle"]
    _require(
        isinstance(bundle.get("filename"), str)
        and isinstance(bundle.get("sha256"), str)
        and isinstance(bundle.get("size"), int)
        and bundle["size"] > 0,
        "release bundle record is invalid",
    )


def _validate_release_provenance(
    candidate: Path,
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    catalog_path: Path,
) -> None:
    _require(manifest.get("schema") == 1, "bad release manifest schema")
    _require(manifest.get("redistribution") == "allowed", "release is not redistributable")
    release_key = manifest.get("release_key")
    _require(isinstance(release_key, str) and release_key, "release key missing")
    try:
        spec = load_release_spec(catalog_path, release_key)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CandidateVerificationError(f"release key is not in catalog: {release_key!r}") from exc
    _require(manifest.get("tag") == spec["tag"], "release tag does not match catalog")
    _require(candidate.name == spec["tag"], "candidate directory does not match release tag")
    _require(manifest.get("model_id") == spec["model_id"], "release model id does not match catalog")
    _require(manifest.get("model_version") == spec["model_version"], "release model version does not match catalog")
    _require(manifest.get("release_version") == spec["release_version"], "release version does not match catalog")
    _require(manifest.get("sample_rate") == spec["sample_rate"], "sample rate does not match catalog")
    _require(manifest.get("runtime") == spec["runtime"], "runtime metadata does not match catalog")
    _require(manifest.get("frontend") == spec["frontend"], "frontend metadata does not match catalog")
    _require(manifest.get("license") == spec["license"], "license does not match catalog")
    _require(manifest.get("redistribution") == spec["redistribution"], "redistribution does not match catalog")
    _require(manifest.get("licensing") == bundle["licensing"], "release licensing metadata mismatch")
    _require(manifest.get("abi") == bundle["abi"], "release ABI metadata mismatch")
    _require(manifest.get("parity") == bundle["parity"], "release parity metadata mismatch")

    upstream = manifest.get("upstream", {})
    provenance = bundle["provenance"]
    expected_upstream = spec["upstream"]
    _require(upstream.get("repo") == expected_upstream["repo"], "upstream repository does not match catalog")
    _require(
        upstream.get("requested_revision") == expected_upstream["revision"],
        "requested upstream revision does not match catalog",
    )
    _require(upstream.get("resolved_revision") == expected_upstream["revision"], "upstream revision is not pinned")
    _require(provenance["repository"] == upstream["repo"], "bundle upstream repository mismatch")
    _require(
        provenance["requested_revision"] == upstream["requested_revision"],
        "bundle requested upstream revision mismatch",
    )
    _require(
        provenance["resolved_revision"] == upstream["resolved_revision"],
        "bundle resolved upstream revision mismatch",
    )
    _require(provenance["exporter"] == spec["exporter"], "exporter provenance does not match catalog")
    try:
        validate_snapshot_pin(Path(upstream["resolved_revision"]), spec)
    except (KeyError, ValueError) as exc:
        raise CandidateVerificationError(f"upstream pin validation failed: {exc}") from exc


def _check_candidate_tree(candidate: Path, archive_name: str, expected_files: set[str], entries: dict[str, bytes], tag: str) -> None:
    expected_root = {"bundle", archive_name, "release-manifest.json", "release-notes.md", "SHA256SUMS"}
    actual_root = {path.name for path in candidate.iterdir()}
    _require(actual_root == expected_root, "candidate directory inventory mismatch")
    for path in candidate.iterdir():
        _require(not path.is_symlink(), f"candidate symlinks are forbidden: {path.name}")
    bundle_dir = candidate / "bundle"
    _require(bundle_dir.is_dir(), "candidate bundle directory missing")
    actual_files: set[str] = set()
    for path in bundle_dir.rglob("*"):
        _require(not path.is_symlink(), f"candidate bundle symlink is forbidden: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(bundle_dir).as_posix())
    _require(actual_files == expected_files, "candidate bundle directory inventory mismatch")
    actual_dirs = {
        path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*") if path.is_dir()
    }
    expected_dirs = {
        parent.as_posix()
        for name in expected_files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    _require(actual_dirs == expected_dirs, "candidate bundle directory inventory mismatch")
    for name in expected_files:
        _require((bundle_dir / name).read_bytes() == entries[f"{tag}/{name}"], f"candidate bundle differs from ZIP: {name}")


def verify_candidate(candidate: str | Path, *, catalog: str | Path | None = None) -> None:
    root = Path(candidate)
    _require(root.is_dir() and not root.is_symlink(), "candidate directory is missing or not a directory")
    manifest_path = root / "release-manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "release manifest missing")
    manifest = _read_json_bytes(manifest_path.read_bytes(), "release manifest")
    _require(manifest.get("schema") == 1, "bad release manifest schema")
    _validate_release_manifest(manifest)
    bundle_record = manifest.get("bundle")
    _require(isinstance(bundle_record, dict), "bundle manifest record missing")
    archive_name = bundle_record.get("filename")
    _require(
        isinstance(archive_name, str) and archive_name and Path(archive_name).name == archive_name,
        "bundle archive filename is unsafe",
    )
    archive_path = root / archive_name
    _require(archive_path.is_file() and not archive_path.is_symlink(), "bundle archive missing")
    archive_data = archive_path.read_bytes()
    _require(
        len(archive_data) == bundle_record.get("size"),
        "bundle archive size mismatch",
    )
    _require(
        hashlib.sha256(archive_data).hexdigest() == bundle_record.get("sha256"),
        "bundle digest mismatch",
    )
    sums_path = root / "SHA256SUMS"
    _require(sums_path.is_file() and not sums_path.is_symlink(), "SHA256SUMS missing")
    _require(
        sums_path.read_text(encoding="utf-8") == f"{bundle_record['sha256']}  {archive_name}\n",
        "SHA256SUMS does not match bundle archive",
    )

    tag = manifest.get("tag")
    _require(isinstance(tag, str) and tag, "release tag missing")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = _check_zip_entries(archive, tag)
    except CandidateVerificationError:
        raise
    except Exception as exc:
        raise CandidateVerificationError(f"invalid bundle ZIP: {exc}") from exc
    _require(f"{tag}/bundle.json" in entries, "bundle.json missing from archive")
    bundle = _read_json_bytes(entries[f"{tag}/bundle.json"], "bundle.json")

    try:
        import jsonschema

        schema = json.loads((ROOT / "schemas/bundle.schema.json").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(bundle, schema)
    except ImportError as exc:
        raise CandidateVerificationError("jsonschema is required to verify bundle metadata") from exc
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        raise CandidateVerificationError(f"bundle.json schema validation failed: {exc}") from exc

    _validate_release_provenance(
        root,
        manifest,
        bundle,
        Path(catalog) if catalog is not None else ROOT / "catalog/releases.json",
    )
    _require(bundle["model_id"] == manifest["model_id"], "bundle model id mismatch")
    _require(bundle["version"] == manifest["model_version"], "bundle model version mismatch")
    _require(bundle["sample_rate"] == manifest["sample_rate"], "bundle sample rate mismatch")
    _require(bundle["runtime"] == manifest["runtime"], "bundle runtime mismatch")
    _require(bundle["frontend"] == manifest["frontend"], "bundle frontend mismatch")
    _require(bundle["license"] == manifest["license"], "bundle license mismatch")

    components = bundle["components"]
    expected_files = {
        "bundle.json",
        *(component["file"] for component in components),
        *(asset["file"] for asset in bundle["assets"]),
    }
    for name in expected_files:
        _safe_bundle_path(name)
    actual_files = {name[len(tag) + 1 :] for name in entries}
    _require(actual_files == expected_files, "archive inventory mismatch")
    _require(
        [item["id"] for item in components] == list(COMPONENT_ORDER),
        "declared component inventory mismatch",
    )
    component_files: set[str] = set()
    for record, component_id in zip(components, COMPONENT_ORDER, strict=True):
        _validate_component_record(record, component_id)
        name = record["file"]
        _safe_bundle_path(name)
        _require(name not in component_files, f"duplicate component file: {name}")
        component_files.add(name)
        data = entries.get(f"{tag}/{name}")
        _require(data is not None, f"component graph missing: {name}")
        _require(
            record["size"] == len(data) and record["sha256"] == hashlib.sha256(data).hexdigest(),
            f"component digest/size mismatch: {name}",
        )
        _validate_onnx(component_id, record, data)

    _validate_assets(bundle, entries, tag)
    _validate_parity(bundle, entries, tag)

    parity_data = entries[f"{tag}/{bundle['parity']['file']}"]
    _require(
        bundle["parity"]["sha256"] == hashlib.sha256(parity_data).hexdigest()
        and bundle["parity"]["size"] == len(parity_data),
        "parity digest/size mismatch",
    )
    _check_candidate_tree(root, archive_name, expected_files, entries, tag)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--catalog", default=str(ROOT / "catalog/releases.json"))
    args = parser.parse_args(argv)
    try:
        verify_candidate(args.candidate, catalog=args.catalog)
    except CandidateVerificationError as exc:
        raise SystemExit(str(exc)) from exc
    print("candidate verified")
    return 0


if __name__ == "__main__":
    main()
