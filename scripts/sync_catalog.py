from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("--repo", default="buchwandler/silencio-onnx-bundles")
    p.add_argument("--catalog", default="catalog/models.json")
    a = p.parse_args()
    root = Path(a.candidate)
    m = json.loads((root / "release-manifest.json").read_text())
    path = Path(a.catalog)
    data = json.loads(path.read_text())
    model_id = m["model_id"]
    filename = m["bundle"]["filename"]
    url = f"https://github.com/{a.repo}/releases/download/{m['tag']}/{filename}"
    data["models"][model_id] = {
        "version": m["model_version"],
        "release_version": m["release_version"],
        "release_tag": m["tag"],
        "sample_rate": m["sample_rate"],
        "license": m["license"],
        "frontend": m["frontend"],
        "runtime": m["runtime"],
        "provenance": m["upstream"],
        "bundle": {"url": url, "sha256": m["bundle"]["sha256"], "size": m["bundle"]["size"]},
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
