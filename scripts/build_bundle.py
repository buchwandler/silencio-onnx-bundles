from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_silencio import export  # noqa: E402
from scripts.silencio_contracts import CACHE_ABI, COMPONENT_ORDER  # noqa: E402
from silencio_onnx_bundles.manifest import load_release_spec, sha256_file, validate_snapshot_pin  # noqa: E402

_PROJECT_NOTICE_FILES = ("LICENSE", "MODEL_LICENSES.md", "NOTICE")
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth", ".safetensors"}
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


def _copy_license_notices(snapshot: Path, bundle_dir: Path) -> list[str]:
    notice_files = []
    for name in _PROJECT_NOTICE_FILES:
        source = ROOT / name
        if not source.is_file():
            raise RuntimeError(f"required license notice is missing: {source}")
        shutil.copy2(source, bundle_dir / name)
        notice_files.append(name)

    upstream_root = bundle_dir / "upstream-notices"
    prefixes = ("license", "notice", "copying")
    for source in sorted(snapshot.rglob("*")):
        if not source.is_file() or not source.name.lower().startswith(prefixes):
            continue
        relative = source.relative_to(snapshot)
        destination = upstream_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        notice_files.append((Path("upstream-notices") / relative).as_posix())
    return sorted(notice_files)


def _asset_record(bundle_dir: Path, name: str, kind: str) -> dict[str, Any]:
    path = bundle_dir / name
    return {
        "file": name,
        "kind": kind,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _runtime_semantics(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    dit = config["dit"]
    vocos = config["vocos"]
    n_fft = int(vocos["n_fft"])
    return {
        "batch_size": 1,
        "dit": {
            "dimension": int(dit["dim"]),
            "layers": int(dit["depth"]),
            "heads": int(dit["heads"]),
            "mel_channels": int(vocos["input_channels"]),
        },
        "cache_semantics": {
            "initial_frames": 0,
            "updated_frames": "past_frames + window_frames - block_size",
            "trim_trailing_lookahead_frames": int(config["block_size"]),
            "reuse_chunk_entry_cache_for_solver_substeps": True,
            "persist_final_substep_cache": True,
        },
        "duration_expansion": {
            "rounding": "nearest, ties to even",
            "minimum_frames_per_token": 1,
            "operation": "repeat each token feature by its rounded predicted duration",
        },
        "bnf_alignment": {
            "text_frames_per_second": 40,
            "sample_rate": int(config["sample_rate"]),
            "fbank_n_fft": 400,
            "fbank_hop_length": 160,
            "minimum_fbank_frames": 19,
            "subsampling_factor": 4,
            "formula": "((fbank_frames - 19) // 16 + 1) * 4; zero below 19 frames",
        },
        "condition_interpolation": {
            "factor": 4,
            "mode": "linear",
            "align_corners": True,
        },
        "solver": {
            "method": "Euler",
            "steps": int(spec["runtime"]["solver_steps"]),
            "schedule": "linear from 1 to 0, float32",
            "update": "x = x - (t - r) * velocity",
            "noise": "host-generated float32 tensor supplied to both parity paths",
        },
        "chunking": {
            "chunk_size": int(config["chunk_size"]),
            "lookahead_frames": int(config["block_size"]),
            "reuse_chunk_entry_cache_for_solver_substeps": True,
            "persist_final_substep_cache": True,
            "reset_cache_each_utterance": True,
        },
        "mel_scaling": {
            "operation": "(noisy_mel.transpose(0, 2, 1) + 1) / 2",
            "clamp": False,
        },
        "speaker_embedding": {
            "dimension": 256,
            "default_voice": str(config["default_voice"]),
            "raw_audio_enrollment_in_bundle": False,
        },
        "vocos": {
            "mel_channels": int(vocos["input_channels"]),
            "layers": int(vocos["num_layers"]),
            "backbone_dimension": int(vocos["dim"]),
            "spectrum_bins": n_fft // 2 + 1,
            "magnitude_transform": "exp then clamp maximum to 100",
            "istft": {
                "n_fft": n_fft,
                "win_length": n_fft,
                "hop_length": int(vocos["hop_length"]),
                "window": "periodic Hann",
                "center": vocos["padding"] == "center",
                "normalized": False,
                "onesided": True,
                "window_envelope": "overlap-add sum of squared window values",
                "minimum_window_envelope": 1e-11,
                "output_trim": "remove n_fft // 2 samples from each centered edge",
                "output_samples": "(mel_frames - 1) * hop_length",
            },
        },
    }


def _validate_component_artifacts(bundle_dir: Path, components: list[dict[str, Any]]) -> list[str]:
    component_ids = [item["id"] for item in components]
    if component_ids != list(COMPONENT_ORDER):
        raise RuntimeError(f"component inventory {component_ids} does not match {list(COMPONENT_ORDER)}")
    files = []
    for component in components:
        name = str(component["file"])
        relative = Path(name)
        if relative.is_absolute() or relative.name != name or relative.suffix != ".onnx":
            raise RuntimeError(f"invalid ONNX component filename: {name}")
        path = bundle_dir / relative
        if not path.is_file():
            raise RuntimeError(f"exported component is missing: {name}")
        if int(component["size"]) != path.stat().st_size:
            raise RuntimeError(f"exported component size differs from its metadata: {name}")
        if str(component["sha256"]) != sha256_file(path):
            raise RuntimeError(f"exported component digest differs from its metadata: {name}")
        files.append(name)
    return files


def _validate_parity_report(bundle_dir: Path, export_result: dict[str, Any]) -> dict[str, Any]:
    path = bundle_dir / "parity-report.json"
    if not path.is_file():
        raise RuntimeError("exporter did not write parity-report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("end_to_end", {}).get("status") != "passed":
        raise RuntimeError("candidate parity report does not pass end-to-end")
    if report.get("consumer_determinism", {}).get("status") != "passed":
        raise RuntimeError("candidate consumer pipeline is not deterministic")
    component_statuses = report.get("components", {})
    if not _REQUIRED_PARITY_COMPONENTS.issubset(component_statuses):
        raise RuntimeError("candidate parity report is missing component stages")
    if any(component_statuses[name].get("status") != "passed" for name in _REQUIRED_PARITY_COMPONENTS):
        raise RuntimeError("candidate parity report has a failing component stage")
    if export_result.get("parity_report") != report:
        raise RuntimeError("export result parity report differs from parity-report.json")
    return report


def _assert_bundle_inventory(bundle_dir: Path, expected_files: set[str]) -> None:
    actual_files = {
        path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*") if path.is_file()
    }
    checkpoint_files = sorted(
        name for name in actual_files if Path(name).suffix.lower() in _CHECKPOINT_SUFFIXES
    )
    if checkpoint_files:
        raise RuntimeError(f"build-time checkpoint assets are forbidden: {checkpoint_files}")
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise RuntimeError(f"bundle inventory mismatch; missing={missing}, unexpected={unexpected}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("release_key")
    parser.add_argument("--catalog", default="catalog/releases.json")
    parser.add_argument("--dist", default="dist")
    args = parser.parse_args(argv)
    catalog = ROOT / args.catalog if not Path(args.catalog).is_absolute() else Path(args.catalog)
    spec = dict(load_release_spec(catalog, args.release_key))
    upstream = spec["upstream"]
    snapshot = Path(
        snapshot_download(repo_id=upstream["repo"], revision=upstream.get("revision", "main"))
    )
    resolved = validate_snapshot_pin(snapshot, spec)
    tag = spec["tag"]
    candidate = ROOT / args.dist / tag
    if candidate.exists():
        shutil.rmtree(candidate)
    bundle_dir = candidate / "bundle"
    bundle_dir.mkdir(parents=True)

    export_result = export(snapshot, bundle_dir, spec)
    components = export_result.get("components", [])
    if not components:
        raise SystemExit("Exporter returned no components")
    component_files = _validate_component_artifacts(bundle_dir, components)
    parity_report = _validate_parity_report(bundle_dir, export_result)
    notice_files = _copy_license_notices(snapshot, bundle_dir)
    upstream_notice_files = [
        name for name in notice_files if name.startswith("upstream-notices/")
    ]
    assets = [
        _asset_record(
            bundle_dir,
            "parity-report.json",
            "parity_report",
        ),
        *(
            _asset_record(
                bundle_dir,
                name,
                "upstream_license_notice" if name in upstream_notice_files else "license_notice",
            )
            for name in notice_files
        ),
    ]
    upstream_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    bundle = {
        "schema": 2,
        "abi": {
            "version": 1,
            "component_order": list(COMPONENT_ORDER),
            "cache": CACHE_ABI,
        },
        "model_id": spec["model_id"],
        "version": spec["model_version"],
        "sample_rate": spec["sample_rate"],
        "frontend": spec["frontend"],
        "runtime": spec["runtime"],
        "runtime_semantics": _runtime_semantics(upstream_config, spec),
        "components": components,
        "assets": assets,
        "parity": {
            "file": "parity-report.json",
            "status": parity_report["status"],
            "consumer_determinism": parity_report["consumer_determinism"]["status"],
            "sha256": sha256_file(bundle_dir / "parity-report.json"),
            "size": (bundle_dir / "parity-report.json").stat().st_size,
        },
        "provenance": {
            "repository": upstream["repo"],
            "requested_revision": upstream.get("revision", "main"),
            "resolved_revision": resolved,
            "exporter": spec["exporter"],
            "upstream_notice_files": upstream_notice_files,
        },
        "license": spec["license"],
        "licensing": {
            "spdx": spec["license"],
            "redistribution": spec["redistribution"],
            "notice_files": notice_files,
            "upstream_notice_files": upstream_notice_files,
        },
    }
    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    expected_files = {
        *component_files,
        "bundle.json",
        "parity-report.json",
        *notice_files,
    }
    _assert_bundle_inventory(bundle_dir, expected_files)

    archive = candidate / f"{tag}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, Path(tag) / path.relative_to(bundle_dir))
    manifest = {
        "schema": 1,
        "release_key": args.release_key,
        "tag": tag,
        "model_id": spec["model_id"],
        "model_version": spec["model_version"],
        "release_version": spec["release_version"],
        "sample_rate": spec["sample_rate"],
        "runtime": spec["runtime"],
        "frontend": spec["frontend"],
        "license": spec["license"],
        "redistribution": spec["redistribution"],
        "licensing": bundle["licensing"],
        "abi": bundle["abi"],
        "parity": bundle["parity"],
        "upstream": {
            "repo": upstream["repo"],
            "requested_revision": upstream.get("revision", "main"),
            "resolved_revision": resolved,
        },
        "bundle": {
            "filename": archive.name,
            "sha256": sha256_file(archive),
            "size": archive.stat().st_size,
        },
    }
    (candidate / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (candidate / "release-notes.md").write_text(
        f"# {tag}\n\nSilencio ONNX bundle for `{spec['model_id']}`.\n\n"
        f"Upstream: `{upstream['repo']}@{resolved}`.\n",
        encoding="utf-8",
    )
    (candidate / "SHA256SUMS").write_text(
        f"{manifest['bundle']['sha256']}  {archive.name}\n",
        encoding="utf-8",
    )
    print(candidate)


if __name__ == "__main__":
    main()
