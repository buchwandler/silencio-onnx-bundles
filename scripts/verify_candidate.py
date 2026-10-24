from __future__ import annotations
import argparse, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from silencio_onnx_bundles.manifest import sha256_file

def main():
    p=argparse.ArgumentParser(); p.add_argument('candidate'); a=p.parse_args(); root=Path(a.candidate)
    m=json.loads((root/'release-manifest.json').read_text())
    if m.get('schema')!=1: raise SystemExit('bad release manifest schema')
    if m.get('redistribution')!='allowed': raise SystemExit('release is not redistributable')
    archive=root/m['bundle']['filename']
    if sha256_file(archive)!=m['bundle']['sha256']: raise SystemExit('bundle digest mismatch')
    with zipfile.ZipFile(archive) as zf:
        names=set(zf.namelist())
        if not any(n.endswith('/bundle.json') for n in names): raise SystemExit('bundle.json missing from archive')
        onnx_names=[n for n in names if n.endswith('.onnx')]
        if not onnx_names: raise SystemExit('candidate contains no ONNX files')
        try:
            import onnx
            for name in onnx_names:
                data=zf.read(name); model=onnx.load_model_from_string(data); onnx.checker.check_model(model)
        except ModuleNotFoundError:
            pass
    print('candidate verified')
if __name__=='__main__': main()
