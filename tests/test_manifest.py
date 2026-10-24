import json
from silencio_onnx_bundles.manifest import load_release_spec

def test_release_catalog_has_silencio():
    spec=load_release_spec('catalog/releases.json','silencio-v1')
    assert spec['upstream']['repo']=='BlueRoseLabs/Silencio'
    assert spec['runtime']['kind']=='silencio-split-v1'
