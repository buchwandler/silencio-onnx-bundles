from __future__ import annotations

from typing import Any

from scripts.silencio_runtime_reference import (
    CACHE_ABI_VERSION,
    CACHE_HEAD_DIM,
    CACHE_HEADS,
    CACHE_LAYERS,
)

OPSET_VERSION = 17
COMPONENT_ORDER = (
    "text_encoder",
    "text_prosody",
    "adapter",
    "dit_step",
    "vocos",
)


def _tensor(dtype: str, *shape: int | str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape)}


def _cache_contract(prefix: str, *, output: bool = False) -> dict[str, dict[str, Any]]:
    name_prefix = f"updated_{prefix}" if output else prefix
    frame_axis = "updated_cache_frames" if output else "past_frames"
    return {
        f"{name_prefix}_l{layer}": _tensor(
            "float32", 1, CACHE_HEADS, frame_axis, CACHE_HEAD_DIM
        )
        for layer in range(CACHE_LAYERS)
    }


_CACHE_INPUTS = {
    **_cache_contract("cache_key"),
    **_cache_contract("cache_value"),
}
_CACHE_OUTPUTS = {
    **_cache_contract("cache_key", output=True),
    **_cache_contract("cache_value", output=True),
}
_CACHE_DYNAMIC_AXES = {
    name: {
        2: "updated_cache_frames" if name.startswith("updated_") else "past_frames"
    }
    for name in (*_CACHE_INPUTS, *_CACHE_OUTPUTS)
}

COMPONENT_CONTRACTS: dict[str, dict[str, Any]] = {
    "text_encoder": {
        "file": "text_encoder.onnx",
        "inputs": {"phoneme_ids": _tensor("int64", 1, "tokens")},
        "outputs": {
            "asr_token_features": _tensor("float32", 1, "tokens", 512),
            "duration_state": _tensor("float32", 1, "tokens", 224),
            "raw_durations": _tensor("float32", 1, "tokens"),
        },
        "dynamic_axes": {
            "phoneme_ids": {1: "tokens"},
            "asr_token_features": {1: "tokens"},
            "duration_state": {1: "tokens"},
            "raw_durations": {1: "tokens"},
        },
    },
    "text_prosody": {
        "file": "text_prosody.onnx",
        "inputs": {"duration_state_frames": _tensor("float32", 1, "text_frames", 224)},
        "outputs": {
            "pitch": _tensor("float32", 1, "prosody_values"),
            "loudness": _tensor("float32", 1, "prosody_values"),
        },
        "dynamic_axes": {
            "duration_state_frames": {1: "text_frames"},
            "pitch": {1: "prosody_values"},
            "loudness": {1: "prosody_values"},
        },
    },
    "adapter": {
        "file": "adapter.onnx",
        "inputs": {
            "asr_features": _tensor("float32", 1, 512, "text_frames"),
            "pitch": _tensor("float32", 1, "prosody_values"),
            "loudness": _tensor("float32", 1, "prosody_values"),
            "bnf_length": _tensor("int64", 1),
        },
        "outputs": {"content_features": _tensor("float32", 1, "bnf_frames", 256)},
        "dynamic_axes": {
            "asr_features": {2: "text_frames"},
            "pitch": {1: "prosody_values"},
            "loudness": {1: "prosody_values"},
            "content_features": {1: "bnf_frames"},
        },
    },
    "dit_step": {
        "file": "dit_step.onnx",
        "inputs": {
            "noisy_mel": _tensor("float32", 1, "window", 80),
            "t": _tensor("float32", 1),
            "r": _tensor("float32", 1),
            "condition": _tensor("float32", 1, "window", 256),
            "speaker_embedding": _tensor("float32", 1, 256),
            "position_offset": _tensor("int64"),
            **_CACHE_INPUTS,
        },
        "outputs": {
            "velocity": _tensor("float32", 1, "window", 80),
            **_CACHE_OUTPUTS,
        },
        "dynamic_axes": {
            "noisy_mel": {1: "window"},
            "condition": {1: "window"},
            "velocity": {1: "window"},
            **_CACHE_DYNAMIC_AXES,
        },
    },
    "vocos": {
        "file": "vocos.onnx",
        "inputs": {"mel": _tensor("float32", 1, 80, "mel_frames")},
        "outputs": {
            "magnitude": _tensor("float32", 1, 321, "mel_frames"),
            "phase": _tensor("float32", 1, 321, "mel_frames"),
        },
        "dynamic_axes": {
            "mel": {2: "mel_frames"},
            "magnitude": {2: "mel_frames"},
            "phase": {2: "mel_frames"},
        },
    }
}

CACHE_ABI = {
    "version": CACHE_ABI_VERSION,
    "layers": CACHE_LAYERS,
    "heads": CACHE_HEADS,
    "head_dim": CACHE_HEAD_DIM,
    "initial_frames": 0,
    "input_names": list(_CACHE_INPUTS),
    "output_names": list(_CACHE_OUTPUTS),
}
