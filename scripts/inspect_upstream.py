from __future__ import annotations
import argparse, json
from pathlib import Path
from huggingface_hub import snapshot_download

def inspect(repo: str, revision: str) -> dict:
    root = Path(snapshot_download(repo_id=repo, revision=revision))
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    source = [x for x in files if x.startswith("silencio/") and x.endswith(".py")]
    return {"repo": repo, "requested_revision": revision, "resolved_revision": root.name, "root": str(root), "python_sources": source, "files": files}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--revision',default='main'); p.add_argument('--output')
    a=p.parse_args(); data=inspect(a.repo,a.revision); text=json.dumps(data,indent=2)
    if a.output: Path(a.output).write_text(text+'\n')
    print(text)
if __name__=='__main__': main()
