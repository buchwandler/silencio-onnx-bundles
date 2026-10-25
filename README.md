# silencio-onnx-bundles

Reproducible build/release/catalog repository for ONNX bundles derived from [BlueRoseLabs/Silencio](https://huggingface.co/BlueRoseLabs/Silencio).

The name uses **bundles** intentionally: the expected portable runtime is a group of ONNX graphs plus a `bundle.json` contract, not necessarily one monolithic model file.

No `src/` layout is used. The tooling package lives at `silencio_onnx_bundles/`. Tooling versions are derived from Git tags through `setuptools_scm`.

## Pinned model and bundle contract

`catalog/releases.json` pins `silencio-v1` to `BlueRoseLabs/Silencio` commit `e200e2f9ef1887bb7b878045e0872d79a241f128`. The builder checks that the resolved Hugging Face snapshot matches that commit before exporting.

The bundle contains five ONNX components: `text_encoder`, `text_prosody`, `adapter`, `dit_step`, and `vocos`. Component tensor contracts and native inference evidence are documented in [`docs/SILENCIO_UPSTREAM_CONTRACT.md`](docs/SILENCIO_UPSTREAM_CONTRACT.md). Host-side duration expansion, solver scheduling, chunk/cache orchestration, and exact iSTFT reconstruction remain outside ONNX as specified by that contract.

Each candidate contains `bundle.json`, the passing `parity-report.json`, required license notices, component hashes and sizes, and upstream provenance. Build-time checkpoints and unlisted files are rejected. The verifier checks archive paths and integrity, exact inventory, ONNX contracts and ONNX Runtime CPU loading, parity, licensing, and provenance.

## Local build and verification

Run these commands from the repository root. They build and verify a local candidate only. They do not publish a release.

```bash
uv run --extra build --extra test python scripts/build_bundle.py silencio-v1
uv run --extra test python scripts/verify_candidate.py dist/silencio-v1-r1
uv run --extra test python -m pytest -q
uv run --extra dev ruff check .
```

The candidate directory is `dist/silencio-v1-r1`. It contains the bundle directory, ZIP archive, release manifest, release notes, and `SHA256SUMS`. The build requires access to the pinned Hugging Face snapshot.

## Release flow

The GitHub Actions release workflow is started manually from **Actions → build-and-release**. Before publication, it runs the test and lint gates, builds the catalog-selected candidate, and invokes the strict candidate verifier. Only that explicitly dispatched workflow proceeds to publish an immutable GitHub Release and synchronize `catalog/models.json`. Local build and verification commands never publish.
