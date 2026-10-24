import json

def test_runtime_catalog_shape():
    data=json.load(open('catalog/models.json'))
    assert data['schema']==1
    assert isinstance(data['models'],dict)
