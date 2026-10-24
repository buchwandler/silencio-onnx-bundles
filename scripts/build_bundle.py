from __future__ import annotations

import argparse, json, shutil, sys, zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silencio_onnx_bundles.manifest import load_release_spec, sha256_file
from scripts.export_silencio import export


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('release_key'); p.add_argument('--catalog',default='catalog/releases.json'); p.add_argument('--dist',default='dist')
    a=p.parse_args(argv)
    spec=dict(load_release_spec(ROOT/a.catalog if not Path(a.catalog).is_absolute() else a.catalog, a.release_key))
    upstream=spec['upstream']
    snapshot=Path(snapshot_download(repo_id=upstream['repo'], revision=upstream.get('revision','main')))
    resolved=snapshot.name
    tag=spec['tag']; candidate=ROOT/a.dist/tag
    if candidate.exists(): shutil.rmtree(candidate)
    work=candidate/'bundle'; work.mkdir(parents=True)
    export_result=export(snapshot, work, spec)
    components=export_result.get('components', [])
    if not components: raise SystemExit('Exporter returned no components')
    bundle={
      'schema':1,'model_id':spec['model_id'],'version':spec['model_version'],'sample_rate':spec['sample_rate'],
      'frontend':spec['frontend'],'runtime':spec['runtime'],'components':components,
      'provenance':{'repository':upstream['repo'],'requested_revision':upstream.get('revision','main'),'resolved_revision':resolved},
      'license':spec['license'],
    }
    (work/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    archive=candidate/f"{tag}.zip"
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work.rglob('*')):
            if path.is_file(): zf.write(path, Path(tag)/path.relative_to(work))
    manifest={
      'schema':1,'release_key':a.release_key,'tag':tag,'model_id':spec['model_id'],'model_version':spec['model_version'],
      'release_version':spec['release_version'],'sample_rate':spec['sample_rate'],'runtime':spec['runtime'],'frontend':spec['frontend'],
      'license':spec['license'],'redistribution':spec['redistribution'],'upstream':{'repo':upstream['repo'],'requested_revision':upstream.get('revision','main'),'resolved_revision':resolved},
      'bundle':{'filename':archive.name,'sha256':sha256_file(archive),'size':archive.stat().st_size},
    }
    (candidate/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (candidate/'release-notes.md').write_text(f"# {tag}\n\nSilencio ONNX bundle for `{spec['model_id']}`.\n\nUpstream: `{upstream['repo']}@{resolved}`.\n")
    (candidate/'SHA256SUMS').write_text(f"{manifest['bundle']['sha256']}  {archive.name}\n")
    print(candidate)

if __name__=='__main__': main()
