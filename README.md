# silencio-onnx-bundles

Reproducible build/release/catalog repository for ONNX bundles derived from [BlueRoseLabs/Silencio](https://huggingface.co/BlueRoseLabs/Silencio).

The name uses **bundles** intentionally: the expected portable runtime is a group of ONNX graphs plus a `bundle.json` contract, not necessarily one monolithic model file.

No `src/` layout is used. The tooling package lives at `silencio_onnx_bundles/`. Tooling versions are derived from Git tags through `setuptools_scm`.

## Release flow

`catalog/releases.json` is the build plan. A GitHub Actions release run:

1. resolves the selected release entry;
2. downloads the pinned/upstream Silencio snapshot;
3. runs the ONNX exporter adapter;
4. validates all ONNX files and creates `bundle.json`;
5. creates an immutable ZIP + SHA256 + release manifest;
6. publishes a GitHub Release;
7. synchronizes `catalog/models.json` with the published asset URL and exact digest.

Run manually from **Actions → build-and-release**.

## First required engineering task

The workflow and artifact contract are complete, but `scripts/export_silencio.py` deliberately does **not** guess internal Silencio tensor contracts. It discovers the current upstream source layout and then calls a pinned adapter. The included adapter emits a precise error showing what it found until the first Silencio graph contract is implemented.

This is intentional: the current Silencio implementation contains Python-side duration expansion, iterative DiT scheduling/cache logic, and Vocos/iSTFT behavior that should be made explicit instead of silently traced into brittle graphs.

Recommended first contract:

- `text_encoder.onnx`
- `adapter.onnx`
- `dit_step.onnx` (one solver step, explicit cache inputs/outputs)
- `vocos.onnx` (or Vocos spectral output + host iSTFT)

The host runtime (`silenciosynth`) should own sentence/chunk scheduling, the three-step solver, G2P, and any host iSTFT.

## Local commands

```bash
python -m pip install -e ".[build,test]"
python scripts/inspect_upstream.py --repo BlueRoseLabs/Silencio --revision main
python scripts/build_bundle.py silencio-v1
python scripts/verify_candidate.py dist/silencio-v1-r1
```

Before the first public release, replace `main` in `catalog/releases.json` with an exact upstream commit when possible. The build manifest always records the resolved Hugging Face snapshot revision.
