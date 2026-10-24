import json
from pathlib import Path


def test_runtime_catalog_shape():
    data = json.loads(Path("catalog/models.json").read_text())
    assert data["schema"] == 1
    assert isinstance(data["models"], dict)
