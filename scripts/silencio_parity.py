from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.silencio_contracts import CACHE_ABI, COMPONENT_CONTRACTS, COMPONENT_ORDER
from scripts.silencio_runtime_reference import (
    bnf_frames_for_text,
    durations_from_raw,
    expand_text_outputs,
    kv_cache_from_outputs,
    render_dit_mel,
    solver_update,
    upsample_condition,
    validate_kv_cache,
    vocos_istft_from_spectrum,
)


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def _copy_arrays(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in values.items()}


def _component_outputs(
    sessions: Mapping[str, Any], component: str, inputs: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    names = list(COMPONENT_CONTRACTS[component]["outputs"])
    values = sessions[component].run(names, dict(inputs))
    return dict(zip(names, values, strict=True))


def run_consumer_pipeline(
    component_dir: Path,
    phoneme_ids: np.ndarray,
    noise: np.ndarray,
    speaker_embedding: np.ndarray,
    *,
    steps: int,
    chunk_size: int,
    block_size: int,
    n_fft: int,
    hop_length: int,
    sessions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only ONNX Runtime and NumPy, with tokens, noise and speaker supplied by the caller."""
    if phoneme_ids.dtype != np.int64 or phoneme_ids.ndim != 2 or phoneme_ids.shape[0] != 1:
        raise ValueError("phoneme_ids must be int64 [1,tokens]")
    if noise.dtype != np.float32 or noise.ndim != 3 or noise.shape[0] != 1:
        raise ValueError("noise must be float32 [1,frames,80]")
    if speaker_embedding.dtype != np.float32 or speaker_embedding.shape != (1, 256):
        raise ValueError("speaker_embedding must be float32 [1,256]")

    if sessions is None:
        import onnxruntime as ort

        sessions = {
            component: ort.InferenceSession(
                str(component_dir / COMPONENT_CONTRACTS[component]["file"]),
                providers=["CPUExecutionProvider"],
            )
            for component in COMPONENT_ORDER
        }

    text_encoder_outputs = _component_outputs(
        sessions, "text_encoder", {"phoneme_ids": phoneme_ids}
    )
    durations = durations_from_raw(text_encoder_outputs["raw_durations"])
    asr_features, duration_state_frames = expand_text_outputs(
        text_encoder_outputs["asr_token_features"],
        text_encoder_outputs["duration_state"],
        durations,
    )
    pitch_outputs = _component_outputs(
        sessions,
        "text_prosody",
        {"duration_state_frames": duration_state_frames},
    )
    bnf_length = bnf_frames_for_text(asr_features.shape[2])
    if bnf_length < 1:
        raise ValueError("phoneme IDs expand to too few frames for the pinned adapter")
    adapter_outputs = _component_outputs(
        sessions,
        "adapter",
        {
            "asr_features": asr_features,
            "pitch": pitch_outputs["pitch"],
            "loudness": pitch_outputs["loudness"],
            "bnf_length": np.asarray([bnf_length], dtype=np.int64),
        },
    )
    content_features = adapter_outputs["content_features"]
    condition = upsample_condition(content_features)

    step_events: list[dict[str, Any]] = []

    def onnx_step(
        noisy_mel: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        condition_window: np.ndarray,
        speaker: np.ndarray,
        offset: int,
        cache: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        past_frames = validate_kv_cache(cache)
        inputs = (
            noisy_mel,
            t,
            r,
            condition_window,
            speaker,
            np.asarray(offset, dtype=np.int64),
            *(cache[name] for name in CACHE_ABI["input_names"]),
        )
        input_names = list(COMPONENT_CONTRACTS["dit_step"]["inputs"])
        outputs = _component_outputs(
            sessions,
            "dit_step",
            dict(zip(input_names, inputs, strict=True)),
        )
        velocity = outputs["velocity"]
        updated_cache = kv_cache_from_outputs(outputs)
        updated_frames = validate_kv_cache(updated_cache)
        if updated_frames != past_frames + noisy_mel.shape[1] - block_size:
            raise RuntimeError("consumer DiT returned an unexpected cache length")
        if not np.isfinite(velocity).all() or not all(
            np.isfinite(value).all() for value in updated_cache.values()
        ):
            raise RuntimeError("consumer DiT produced non-finite velocity or cache values")
        step_events.append(
            {
                "offset": int(offset),
                "x_in": noisy_mel.copy(),
                "t": t.copy(),
                "r": r.copy(),
                "condition": condition_window.copy(),
                "speaker": speaker.copy(),
                "cache_in": _copy_arrays(cache),
                "velocity": velocity.copy(),
                "x_out": solver_update(noisy_mel, velocity, t, r),
                "cache_out": _copy_arrays(updated_cache),
                "past_frames": past_frames,
                "updated_frames": updated_frames,
            }
        )
        return velocity, updated_cache

    mel, final_cache = render_dit_mel(
        condition,
        noise,
        speaker_embedding,
        onnx_step,
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
    )

    vocos_outputs = _component_outputs(sessions, "vocos", {"mel": mel})
    waveform = vocos_istft_from_spectrum(
        vocos_outputs["magnitude"],
        vocos_outputs["phase"],
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return {
        "text_encoder": text_encoder_outputs,
        "durations": durations,
        "asr_features": asr_features,
        "duration_state_frames": duration_state_frames,
        "text_prosody": pitch_outputs,
        "bnf_length": np.asarray([bnf_length], dtype=np.int64),
        "content_features": content_features,
        "condition": condition,
        "dit_steps": step_events,
        "mel": mel,
        "final_cache": final_cache,
        "vocos_spectrum": vocos_outputs,
        "waveform": waveform,
    }


def run_native_pipeline(
    tts: Any,
    phonemes: str,
    *,
    seed: int,
    steps: int,
    chunk_size: int,
    block_size: int,
) -> dict[str, Any]:
    """Run pinned PyTorch Silencio with the same host semantics as the consumer."""
    import torch

    from scripts.silencio_graph_wrappers import (
        TextEncoderWrapper,
        TextProsodyWrapper,
        VocosSpectrumWrapper,
    )

    phoneme_ids_tensor = tts.text.ids(phonemes).cpu()
    phoneme_ids = _as_numpy(phoneme_ids_tensor).astype(np.int64, copy=False)
    text_encoder = TextEncoderWrapper(tts.text.ts)
    text_prosody = TextProsodyWrapper(tts.text.ts)
    with torch.no_grad():
        encoder_tensors = text_encoder(phoneme_ids_tensor)
    encoder_outputs = dict(
        zip(
            COMPONENT_CONTRACTS["text_encoder"]["outputs"],
            (_as_numpy(value) for value in encoder_tensors),
            strict=True,
        )
    )
    durations = durations_from_raw(encoder_outputs["raw_durations"])
    expanded_asr, duration_state_frames = expand_text_outputs(
        encoder_outputs["asr_token_features"],
        encoder_outputs["duration_state"],
        durations,
    )
    with torch.no_grad():
        graph_pitch, graph_loudness = text_prosody(
            torch.from_numpy(np.ascontiguousarray(duration_state_frames))
        )
        native_asr, native_pitch, native_loudness = tts.text(phoneme_ids_tensor)
    native_asr_np = _as_numpy(native_asr)
    native_pitch_np = _as_numpy(native_pitch)
    native_loudness_np = _as_numpy(native_loudness)
    bnf_length = bnf_frames_for_text(native_asr_np.shape[2])
    if bnf_length < 1:
        raise ValueError(f"phonemes {phonemes!r} expand to too few adapter frames")
    bnf_tensor = torch.tensor([bnf_length], dtype=torch.int64)
    with torch.no_grad():
        native_content = tts.adapter(native_asr, native_pitch, native_loudness, bnf_tensor)
    content_features = _as_numpy(native_content).astype(np.float32, copy=False)
    condition = upsample_condition(content_features)
    noise = np.random.default_rng(seed).standard_normal(
        (1, condition.shape[1], 80), dtype=np.float32
    )
    speaker_embedding = (
        _as_numpy(tts.voice(tts.config["default_voice"]))
        .astype(np.float32, copy=False)
        .reshape(1, 256)
    )
    step_events: list[dict[str, Any]] = []

    def native_step(
        noisy_mel: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        condition_window: np.ndarray,
        speaker: np.ndarray,
        offset: int,
        cache: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        past_frames = validate_kv_cache(cache)
        kv_cache = None
        if past_frames:
            kv_cache = [
                (
                    torch.from_numpy(cache[f"cache_key_l{layer}"]),
                    torch.from_numpy(cache[f"cache_value_l{layer}"]),
                )
                for layer in range(4)
            ]
        with torch.no_grad():
            velocity_tensor, updated_tensors = tts.dit(
                torch.from_numpy(noisy_mel),
                torch.from_numpy(t),
                torch.from_numpy(r),
                cache=None,
                cond=torch.from_numpy(condition_window),
                spks=torch.from_numpy(speaker),
                offset=offset,
                is_inference=True,
                kv_cache=kv_cache,
            )
        velocity = _as_numpy(velocity_tensor)
        updated_outputs = {
            f"updated_cache_key_l{layer}": _as_numpy(updated_tensors[layer][0])
            for layer in range(4)
        }
        updated_outputs.update(
            {
                f"updated_cache_value_l{layer}": _as_numpy(updated_tensors[layer][1])
                for layer in range(4)
            }
        )
        updated_cache = kv_cache_from_outputs(updated_outputs)
        updated_frames = validate_kv_cache(updated_cache)
        step_events.append(
            {
                "offset": int(offset),
                "x_in": noisy_mel.copy(),
                "t": t.copy(),
                "r": r.copy(),
                "condition": condition_window.copy(),
                "speaker": speaker.copy(),
                "cache_in": _copy_arrays(cache),
                "velocity": velocity.copy(),
                "x_out": solver_update(noisy_mel, velocity, t, r),
                "cache_out": _copy_arrays(updated_cache),
                "past_frames": past_frames,
                "updated_frames": updated_frames,
            }
        )
        return velocity, updated_cache

    mel, final_cache = render_dit_mel(
        condition,
        noise,
        speaker_embedding,
        native_step,
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
    )
    mel_tensor = torch.from_numpy(np.ascontiguousarray(mel))
    vocos_wrapper = VocosSpectrumWrapper(tts.vocos)
    with torch.no_grad():
        magnitude, phase = vocos_wrapper(mel_tensor)
        waveform = tts.vocos.decode(mel_tensor)
    vocos_spectrum = {
        "magnitude": _as_numpy(magnitude),
        "phase": _as_numpy(phase),
    }
    return {
        "inputs": {
            "phonemes": phonemes,
            "phoneme_ids": phoneme_ids,
            "noise": noise,
            "speaker_embedding": speaker_embedding,
            "seed": seed,
        },
        "text_encoder": encoder_outputs,
        "durations": durations,
        "expanded_asr": expanded_asr,
        "duration_state_frames": duration_state_frames,
        "graph_pitch": _as_numpy(graph_pitch),
        "graph_loudness": _as_numpy(graph_loudness),
        "native_asr": native_asr_np,
        "native_pitch": native_pitch_np,
        "native_loudness": native_loudness_np,
        "bnf_length": np.asarray([bnf_length], dtype=np.int64),
        "content_features": content_features,
        "condition": condition,
        "dit_steps": step_events,
        "mel": mel,
        "final_cache": final_cache,
        "vocos_spectrum": vocos_spectrum,
        "waveform": _as_numpy(waveform),
    }


def _array_metric(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(f"parity shape mismatch: {actual.shape} != {expected.shape}")
    errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if np.issubdtype(actual.dtype, np.integer) and np.issubdtype(expected.dtype, np.integer):
        passed = np.array_equal(actual, expected)
    else:
        passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    if not passed:
        max_abs = float(np.max(errors, initial=0.0))
        raise RuntimeError(f"parity exceeded tolerance (max_abs={max_abs:.8g})")
    return {
        "status": "passed",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_error": float(np.max(errors, initial=0.0)),
        "mean_abs_error": float(np.mean(errors)) if errors.size else 0.0,
        "rtol": rtol,
        "atol": atol,
    }


def _mapping_metrics(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    if set(actual) != set(expected):
        raise RuntimeError("parity output names differ")
    return {
        name: _array_metric(actual[name], expected[name], rtol=rtol, atol=atol)
        for name in actual
    }


def _waveform_health(waveform: np.ndarray) -> dict[str, float | int]:
    if waveform.ndim != 2 or not np.isfinite(waveform).all():
        raise RuntimeError("waveform must be finite with shape [1,samples]")
    values = waveform.astype(np.float64, copy=False)
    return {
        "samples": int(waveform.shape[-1]),
        "rms": float(np.sqrt(np.mean(values**2))),
        "peak": float(np.max(np.abs(values))),
        "dc_offset": float(np.mean(values)),
    }


def _spectral_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual_magnitude = np.abs(np.fft.rfft(actual.astype(np.float64), axis=-1))
    expected_magnitude = np.abs(np.fft.rfft(expected.astype(np.float64), axis=-1))
    difference = actual_magnitude - expected_magnitude
    return {
        "spectral_convergence": float(
            np.linalg.norm(difference) / max(np.linalg.norm(expected_magnitude), 1e-12)
        ),
        "log_spectral_mae": float(
            np.mean(
                np.abs(
                    np.log(actual_magnitude + 1e-7)
                    - np.log(expected_magnitude + 1e-7)
                )
            )
        ),
    }


def _compare_pipelines(native: Mapping[str, Any], consumer: Mapping[str, Any]) -> dict[str, Any]:
    native_text_outputs = native["text_encoder"]
    consumer_text_outputs = consumer["text_encoder"]
    component_metrics = {
        "text_encoder": {
            "status": "passed",
            "outputs": _mapping_metrics(consumer_text_outputs, native_text_outputs),
        },
        "duration_expansion": {
            "status": "passed",
            "outputs": _mapping_metrics(
                {
                    "durations": consumer["durations"],
                    "asr_features": consumer["asr_features"],
                    "duration_state_frames": consumer["duration_state_frames"],
                    "bnf_length": consumer["bnf_length"],
                },
                {
                    "durations": native["durations"],
                    "asr_features": native["native_asr"],
                    "duration_state_frames": native["duration_state_frames"],
                    "bnf_length": native["bnf_length"],
                },
            ),
            "native_source_parity": _mapping_metrics(
                {"expanded_asr": native["expanded_asr"]},
                {"expanded_asr": native["native_asr"]},
            ),
        },
        "text_prosody": {
            "status": "passed",
            "outputs": _mapping_metrics(
                consumer["text_prosody"],
                {"pitch": native["native_pitch"], "loudness": native["native_loudness"]},
            ),
            "native_source_parity": _mapping_metrics(
                {"pitch": native["graph_pitch"], "loudness": native["graph_loudness"]},
                {"pitch": native["native_pitch"], "loudness": native["native_loudness"]},
            ),
        },
        "adapter": {
            "status": "passed",
            "outputs": _mapping_metrics(
                {"content_features": consumer["content_features"]},
                {"content_features": native["content_features"]},
            ),
        },
        "condition_interpolation": {
            "status": "passed",
            "outputs": _mapping_metrics(
                {"condition": consumer["condition"]},
                {"condition": native["condition"]},
            ),
        },
    }

    native_steps = native["dit_steps"]
    consumer_steps = consumer["dit_steps"]
    if len(native_steps) != len(consumer_steps):
        raise RuntimeError("native and consumer DiT call counts differ")
    step_metrics = []
    for index, (native_step, consumer_step) in enumerate(
        zip(native_steps, consumer_steps, strict=True)
    ):
        for name in ("offset", "past_frames", "updated_frames"):
            if native_step[name] != consumer_step[name]:
                raise RuntimeError(f"DiT step {index} {name} differs")
        inputs = _mapping_metrics(
            {
                name: consumer_step[name]
                for name in ("x_in", "t", "r", "condition", "speaker")
            },
            {name: native_step[name] for name in ("x_in", "t", "r", "condition", "speaker")},
        )
        cache_inputs = _mapping_metrics(
            consumer_step["cache_in"],
            native_step["cache_in"],
            rtol=1e-4,
            atol=2e-5,
        )
        outputs = _mapping_metrics(
            {name: consumer_step[name] for name in ("velocity", "x_out")},
            {name: native_step[name] for name in ("velocity", "x_out")},
            rtol=1e-4,
            atol=2e-5,
        )
        cache_outputs = _mapping_metrics(
            consumer_step["cache_out"], native_step["cache_out"],
            rtol=1e-4,
            atol=5e-5,
        )
        step_metrics.append(
            {
                "index": index,
                "offset": native_step["offset"],
                "past_frames": native_step["past_frames"],
                "updated_frames": native_step["updated_frames"],
                "inputs": inputs,
                "cache_inputs": cache_inputs,
                "outputs": outputs,
                "cache_outputs": cache_outputs,
            }
        )
    component_metrics["dit_step"] = {"status": "passed", "calls": step_metrics}
    component_metrics["final_mel"] = {
        "status": "passed",
        "outputs": _mapping_metrics({"mel": consumer["mel"]}, {"mel": native["mel"]}),
    }
    vocos_spectrum_metrics = {
        "magnitude": _array_metric(
            consumer["vocos_spectrum"]["magnitude"],
            native["vocos_spectrum"]["magnitude"],
            rtol=1e-4,
            atol=2.5e-4,
        ),
        "phase": _array_metric(
            consumer["vocos_spectrum"]["phase"],
            native["vocos_spectrum"]["phase"],
            rtol=1e-4,
            atol=1.5e-2,
        ),
    }
    component_metrics["vocos"] = {
        "status": "passed",
        "spectrum_outputs": vocos_spectrum_metrics,
    }
    native_waveform = native["waveform"]
    consumer_waveform = consumer["waveform"]
    if native_waveform.shape[-1] != consumer_waveform.shape[-1]:
        raise RuntimeError("native and consumer waveform lengths differ")
    waveform_parity = _array_metric(
        consumer_waveform,
        native_waveform,
        rtol=1e-4,
        atol=1e-4,
    )
    native_health = _waveform_health(native_waveform)
    consumer_health = _waveform_health(consumer_waveform)
    health_parity = {
        name: _array_metric(
            np.asarray([consumer_health[name]], dtype=np.float64),
            np.asarray([native_health[name]], dtype=np.float64),
            rtol=1e-4,
            atol=1e-5,
        )
        for name in ("rms", "peak", "dc_offset")
    }
    component_metrics["vocos"]["waveform"] = waveform_parity
    component_metrics["vocos"]["waveform_health"] = {
        "native": native_health,
        "consumer": consumer_health,
        "parity": health_parity,
    }
    component_metrics["vocos"]["spectral_metrics"] = _spectral_metrics(
        consumer_waveform, native_waveform
    )
    return {
        "status": "passed",
        "components": component_metrics,
        "end_to_end": {
            "status": "passed",
            "dit_calls": len(step_metrics),
            "mel": component_metrics["final_mel"]["outputs"]["mel"],
            "waveform": waveform_parity,
            "waveform_samples": int(consumer_waveform.shape[-1]),
        },
    }


def _compare_consumer_repeats(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    array_outputs = (
        "text_encoder",
        "asr_features",
        "duration_state_frames",
        "text_prosody",
        "content_features",
        "condition",
        "mel",
        "vocos_spectrum",
        "final_cache",
    )
    metrics = {
        name: _mapping_metrics(second[name], first[name], rtol=0.0, atol=0.0)
        if isinstance(first[name], Mapping)
        else _array_metric(second[name], first[name], rtol=0.0, atol=0.0)
        for name in array_outputs
    }
    for name in ("durations", "bnf_length", "waveform"):
        metrics[name] = _array_metric(second[name], first[name], rtol=0.0, atol=0.0)
    if len(first["dit_steps"]) != len(second["dit_steps"]):
        raise RuntimeError("repeated consumer DiT call counts differ")
    metrics["dit_steps"] = []
    for index, (first_step, second_step) in enumerate(
        zip(first["dit_steps"], second["dit_steps"], strict=True)
    ):
        for name in ("offset", "past_frames", "updated_frames"):
            if first_step[name] != second_step[name]:
                raise RuntimeError(f"repeated consumer DiT step {index} {name} differs")
        metrics["dit_steps"].append(
            {
                "index": index,
                "inputs": _mapping_metrics(
                    {name: second_step[name] for name in ("x_in", "t", "r", "condition", "speaker")},
                    {name: first_step[name] for name in ("x_in", "t", "r", "condition", "speaker")},
                    rtol=0.0,
                    atol=0.0,
                ),
                "cache_inputs": _mapping_metrics(
                    second_step["cache_in"], first_step["cache_in"], rtol=0.0, atol=0.0
                ),
                "outputs": _mapping_metrics(
                    {name: second_step[name] for name in ("velocity", "x_out")},
                    {name: first_step[name] for name in ("velocity", "x_out")},
                    rtol=0.0,
                    atol=0.0,
                ),
                "cache_outputs": _mapping_metrics(
                    second_step["cache_out"], first_step["cache_out"], rtol=0.0, atol=0.0
                ),
            }
        )
    return {"status": "passed", "outputs": metrics}


def run_parity_report(
    *,
    tts: Any,
    component_dir: Path,
    spec: Mapping[str, Any],
    phonemes: str = "həlˈoʊ.",
    seed: int = 20261024,
) -> dict[str, Any]:
    """Run native and consumer paths on shared deterministic inputs and persist the report."""
    runtime = spec["runtime"]
    vocos_config = tts.config["vocos"]
    steps = int(runtime["solver_steps"])
    chunk_size = int(tts.config["chunk_size"])
    block_size = int(tts.config["block_size"])
    n_fft = int(vocos_config["n_fft"])
    hop_length = int(vocos_config["hop_length"])
    native = run_native_pipeline(
        tts,
        phonemes,
        seed=seed,
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
    )
    inputs = native["inputs"]
    consumer = run_consumer_pipeline(
        component_dir,
        inputs["phoneme_ids"],
        inputs["noise"],
        inputs["speaker_embedding"],
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    consumer_repeat = run_consumer_pipeline(
        component_dir,
        inputs["phoneme_ids"],
        inputs["noise"],
        inputs["speaker_embedding"],
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    consumer_determinism = _compare_consumer_repeats(consumer, consumer_repeat)
    comparison = _compare_pipelines(native, consumer)
    report = {
        "schema": 1,
        "status": comparison["status"],
        "upstream": {
            "repository": spec["upstream"]["repo"],
            "revision": spec["upstream"]["revision"],
        },
        "input": {
            "phonemes": phonemes,
            "phoneme_tokens": int(inputs["phoneme_ids"].shape[1]),
            "seed": seed,
            "noise_shape": list(inputs["noise"].shape),
        },
        "runtime": {
            "solver_steps": steps,
            "chunk_size": chunk_size,
            "block_size": block_size,
            "vocos_n_fft": n_fft,
            "vocos_hop_length": hop_length,
        },
        "components": comparison["components"],
        "end_to_end": comparison["end_to_end"],
        "consumer_determinism": consumer_determinism,
    }
    (component_dir / "parity-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
