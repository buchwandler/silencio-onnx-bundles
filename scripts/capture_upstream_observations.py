from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

CASES = (
    ("short_default_voice", "həlˈoʊ.", None),
    ("short_alternate_voice", "həlˈoʊ.", "librispeech_1320"),
    ("long_alternate_voice", "ðɪs ɪz ə sɪmpəl tɛst.", "librispeech_1320"),
)


def summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        numeric = tensor.float()
        return {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "rank": tensor.ndim,
            "shape": list(tensor.shape),
            "finite": bool(torch.isfinite(numeric).all()),
            "min": float(numeric.min()) if numeric.numel() else None,
            "max": float(numeric.max()) if numeric.numel() else None,
            "mean": float(numeric.mean()) if numeric.numel() else None,
        }
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    return {"python_type": type(value).__name__, "value": str(value)}


def _hook_module(
    name: str, module: torch.nn.Module, events: dict[str, list[dict[str, Any]]]
) -> list[Any]:
    def pre_hook(_module, args):
        events[name].append({"inputs": summarize(args)})

    def post_hook(_module, _args, output):
        events[name][-1]["outputs"] = summarize(output)

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(post_hook),
    ]


def capture(snapshot: Path, output: Path, *, device: str, steps: int, seed: int) -> None:
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {snapshot}")
    sys.path.insert(0, str(snapshot))
    from silencio import SilencioTTS

    torch.set_num_threads(1)
    tts = SilencioTTS.from_pretrained(str(snapshot), device=device)
    observations: dict[str, Any] = {
        "snapshot": snapshot.name,
        "device": device,
        "steps": steps,
        "seed": seed,
        "input_kind": "phoneme strings supplied directly, without invoking G2P",
        "cases": [],
    }

    for label, phonemes, voice in CASES:
        events: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in (
                "text",
                "duration_lstm",
                "duration_projection",
                "asr_projection",
                "prosody_shared",
                "adapter",
                "vocos_backbone",
                "vocos_head",
            )
        }
        dit_calls: list[dict[str, Any]] = []
        handles = []
        predictor = tts.text.ts.predictor
        modules = (
            ("text", tts.text),
            ("duration_lstm", predictor.lstm),
            ("duration_projection", predictor.duration_proj),
            ("asr_projection", tts.text.ts.asr_proj),
            ("prosody_shared", predictor.shared),
            ("adapter", tts.adapter),
            ("vocos_backbone", tts.vocos.backbone),
            ("vocos_head", tts.vocos.head),
        )
        for name, module in modules:
            handles.extend(_hook_module(name, module, events))

        def dit_pre(_module, args, kwargs, calls=dit_calls):
            calls.append({"inputs": summarize(args), "kwargs": summarize(kwargs)})

        def dit_post(_module, _args, _kwargs, result, calls=dit_calls):
            calls[-1]["outputs"] = summarize(result)

        handles.append(tts.dit.register_forward_pre_hook(dit_pre, with_kwargs=True))
        handles.append(tts.dit.register_forward_hook(dit_post, with_kwargs=True))

        ids = tts.text.ids(phonemes)
        content = tts.content(phonemes)
        waveform = tts.render(content, voice=voice, steps=steps, seed=seed)
        observations["cases"].append(
            {
                "name": label,
                "phonemes": phonemes,
                "phoneme_token_count_including_boundary_pads": ids.shape[1],
                "voice": voice or tts.config["default_voice"],
                "speaker_embedding": summarize(tts.voice(voice)),
                "stage_calls": events,
                "dit_calls": dit_calls,
                "waveform": summarize(waveform),
            }
        )
        for handle in handles:
            handle.remove()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    call_count = sum(len(case["dit_calls"]) for case in observations["cases"])
    print(f"Captured {len(observations['cases'])} cases and {call_count} DiT calls to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture native Silencio tensor observations.")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    capture(args.snapshot, args.output, device=args.device, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
