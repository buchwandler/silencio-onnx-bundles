import json
from pathlib import Path

import pytest

from scripts.inspect_upstream_contract import inspect_snapshot
from silencio_onnx_bundles.manifest import validate_snapshot_pin

PIN = "e200e2f9ef1887bb7b878045e0872d79a241f128"


def test_release_catalog_uses_immutable_upstream_pin():
    catalog = json.loads(Path("catalog/releases.json").read_text(encoding="utf-8"))
    assert catalog["releases"]["silencio-v1"]["upstream"]["revision"] == PIN


def test_validate_snapshot_pin_accepts_exact_public_release_snapshot(tmp_path):
    snapshot = tmp_path / PIN
    snapshot.mkdir()
    spec = {"upstream": {"revision": PIN}, "redistribution": "allowed"}

    assert validate_snapshot_pin(snapshot, spec) == PIN


def test_validate_snapshot_pin_rejects_mismatched_commit(tmp_path):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    spec = {"upstream": {"revision": PIN}, "redistribution": "allowed"}

    with pytest.raises(ValueError, match="does not match configured pin"):
        validate_snapshot_pin(snapshot, spec)


def test_validate_snapshot_pin_rejects_moving_public_revision(tmp_path):
    snapshot = tmp_path / "resolved-commit"
    snapshot.mkdir()
    spec = {"upstream": {"revision": "main"}, "redistribution": "allowed"}

    with pytest.raises(ValueError, match="must be a full commit hash"):
        validate_snapshot_pin(snapshot, spec)


def test_inspect_snapshot_reports_contract_discovery_symbols(tmp_path):
    package = tmp_path / "snapshot" / "silencio"
    package.mkdir(parents=True)
    (package / "model.py").write_text(
        """\
MODEL_WIDTH = 7

class Encoder:
    def forward(self, tokens):
        return tokens.repeat_interleave(2)

def infer_audio(text):
    for step in range(3):
        text = decode(text)
    return text


def stream_audio(sentences, steps):
    for sentence in sentences:
        decode(sentence, steps=steps)
    return sentences
""",
        encoding="utf-8",
    )

    report = inspect_snapshot(tmp_path / "snapshot")
    module = report["modules"][0]

    assert module["module"] == "silencio.model"
    assert module["top_level_constants"] == [{"name": "MODEL_WIDTH", "line": 1, "value": "7"}]
    assert module["forward_signatures"][0]["signature"] == "forward(self, tokens)"
    assert module["inference_generation_functions"][0]["name"] == "infer_audio"
    assert module["references"]["repeat_interleave"] == [5]
    assert module["references"]["decode"] == [9, 15]
    assert module["loops"][0]["solver_related"] is True
    assert module["loops"][1]["solver_related"] is False
