from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.onnx_export import compare_onnx_probe, export_component
from scripts.silencio_contracts import CACHE_ABI, COMPONENT_CONTRACTS, OPSET_VERSION
from scripts.silencio_graph_wrappers import (
    AdapterWrapper,
    DiTStepWrapper,
    TextEncoderWrapper,
    TextProsodyWrapper,
    VocosSpectrumWrapper,
)
from scripts.silencio_runtime_reference import (
    bnf_frames_for_text,
    durations_from_raw,
    empty_kv_cache,
    expand_text_outputs,
    kv_cache_from_outputs,
    render_dit_mel,
    solver_schedule,
    upsample_condition,
    validate_kv_cache,
    vocos_istft_from_spectrum,
)
from silencio_onnx_bundles.manifest import validate_snapshot_pin

_TEXT_CASES = (
    "həlˈoʊ.",
    "həlˈoʊ həlˈoʊ.",
    "ðɪs ɪz ə sɪmpəl tɛst.",
)


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _as_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value))


def _check_close(
    actual: np.ndarray,
    expected: torch.Tensor,
    label: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    expected_array = _as_numpy(expected)
    try:
        np.testing.assert_allclose(actual, expected_array, rtol=rtol, atol=atol)
    except AssertionError as exc:
        error = np.max(np.abs(actual - expected_array), initial=0.0)
        raise RuntimeError(f"{label} differs from pinned PyTorch output (max_abs={error:.8g})") from exc


def _prepare_cases(
    tts: Any,
    text_encoder: TextEncoderWrapper,
    text_prosody: TextProsodyWrapper,
    adapter: AdapterWrapper,
) -> list[dict[str, Any]]:
    cases = []
    for phonemes in _TEXT_CASES:
        phoneme_ids = tts.text.ids(phonemes).cpu()
        with torch.no_grad():
            asr_tokens, duration_state, raw_durations = text_encoder(phoneme_ids)
        durations = durations_from_raw(_as_numpy(raw_durations).astype(np.float32, copy=False))
        asr_frames, state_frames = expand_text_outputs(
            _as_numpy(asr_tokens).astype(np.float32, copy=False),
            _as_numpy(duration_state).astype(np.float32, copy=False),
            durations,
        )
        state_tensor = _as_tensor(state_frames)
        with torch.no_grad():
            pitch, loudness = text_prosody(state_tensor)
            native_asr, native_pitch, native_loudness = tts.text(phoneme_ids)
        _check_close(asr_frames, native_asr, "host-expanded ASR features")
        _check_close(_as_numpy(pitch), native_pitch, "text prosody pitch")
        _check_close(_as_numpy(loudness), native_loudness, "text prosody loudness")

        bnf_length = bnf_frames_for_text(asr_frames.shape[2])
        if bnf_length < 1:
            raise ValueError(
                f"probe phonemes {phonemes!r} expand to too few frames for the pinned adapter"
            )
        adapter_inputs = (
            _as_tensor(asr_frames),
            pitch,
            loudness,
            torch.tensor([bnf_length], dtype=torch.int64),
        )
        with torch.no_grad():
            graph_content = adapter(*adapter_inputs)
            native_content = tts.adapter(*adapter_inputs)
        _check_close(_as_numpy(graph_content), native_content, "adapter output")

        cases.append(
            {
                "phoneme_ids": phoneme_ids,
                "text_frames": int(asr_frames.shape[2]),
                "bnf_length": bnf_length,
                "content_features": native_content,
                "prosody_inputs": (state_tensor,),
                "adapter_inputs": adapter_inputs,
            }
        )
    token_lengths = {int(case["phoneme_ids"].shape[1]) for case in cases}
    if len(token_lengths) != len(cases):
        raise ValueError("text parity cases must exercise distinct token lengths")
    return cases


def _check_dynamic_parity(
    out_dir: Path,
    wrappers: dict[str, torch.nn.Module],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import onnxruntime as ort

    results = []
    for component in ("text_encoder", "text_prosody", "adapter"):
        path = out_dir / COMPONENT_CONTRACTS[component]["file"]
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for case in cases:
            if component == "text_encoder":
                args = (case["phoneme_ids"],)
            elif component == "text_prosody":
                args = case["prosody_inputs"]
            else:
                args = case["adapter_inputs"]
            metrics = compare_onnx_probe(
                wrappers[component],
                path,
                {**COMPONENT_CONTRACTS[component], "id": component},
                args,
                session=session,
                rtol=1e-4,
                atol=1e-5,
            )
            results.append(
                {
                    "component": component,
                    "phoneme_tokens": int(case["phoneme_ids"].shape[1]),
                    "text_frames": case["text_frames"],
                    "bnf_frames": case["bnf_length"],
                    "outputs": metrics,
                }
            )
        del session
    return results


def _native_dit_step(
    tts: Any,
    noisy_mel: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    condition: np.ndarray,
    speaker_embedding: np.ndarray,
    offset: int,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    past_frames = validate_kv_cache(cache)
    kv_cache = None
    if past_frames:
        kv_cache = [
            (
                _as_tensor(cache[f"cache_key_l{layer}"]),
                _as_tensor(cache[f"cache_value_l{layer}"]),
            )
            for layer in range(4)
        ]
    with torch.no_grad():
        velocity, updated_cache = tts.dit(
            _as_tensor(noisy_mel),
            _as_tensor(t),
            _as_tensor(r),
            cache=None,
            cond=_as_tensor(condition),
            spks=_as_tensor(speaker_embedding),
            offset=offset,
            is_inference=True,
            kv_cache=kv_cache,
        )
    outputs = {}
    for layer in range(4):
        outputs[f"updated_cache_key_l{layer}"] = _as_numpy(updated_cache[layer][0])
        outputs[f"updated_cache_value_l{layer}"] = _as_numpy(updated_cache[layer][1])
    return _as_numpy(velocity), kv_cache_from_outputs(outputs)


def _dit_graph_args(
    noisy_mel: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    condition: np.ndarray,
    speaker_embedding: np.ndarray,
    offset: int,
    cache: dict[str, np.ndarray],
) -> tuple[torch.Tensor, ...]:
    return (
        _as_tensor(noisy_mel),
        _as_tensor(t),
        _as_tensor(r),
        _as_tensor(condition),
        _as_tensor(speaker_embedding),
        torch.tensor(offset, dtype=torch.int64),
        *(_as_tensor(cache[name]) for name in CACHE_ABI["input_names"]),
    )


def _check_dit_parity(
    tts: Any,
    out_dir: Path,
    wrapper: DiTStepWrapper,
    case: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    import onnxruntime as ort

    component = "dit_step"
    contract = {**COMPONENT_CONTRACTS[component], "id": component}
    path = out_dir / contract["file"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    block_size = int(tts.config["block_size"])
    chunk_size = int(tts.config["chunk_size"])
    steps = int(spec["runtime"]["solver_steps"])
    schedule = solver_schedule(steps)
    content = _as_numpy(case["content_features"]).astype(np.float32, copy=False)
    condition = upsample_condition(content)
    window = chunk_size + block_size
    noise = np.random.default_rng(20261024).standard_normal(
        (1, condition.shape[1], 80), dtype=np.float32
    )
    speaker = _as_numpy(tts.voice(tts.config["default_voice"])).astype(np.float32).reshape(1, 256)
    empty_cache = empty_kv_cache()

    initial_args = _dit_graph_args(
        noise[:, :window],
        schedule[:1],
        schedule[1:2],
        condition[:, :window],
        speaker,
        0,
        empty_cache,
    )
    initial_metrics = compare_onnx_probe(
        wrapper,
        path,
        contract,
        initial_args,
        session=session,
        rtol=1e-4,
        atol=2e-5,
    )
    with torch.no_grad():
        eager_initial = wrapper(*initial_args)
    native_velocity, first_cache = _native_dit_step(
        tts,
        noise[:, :window],
        schedule[:1],
        schedule[1:2],
        condition[:, :window],
        speaker,
        0,
        empty_cache,
    )
    _check_close(_as_numpy(eager_initial[0]), _as_tensor(native_velocity), "initial DiT velocity")
    initial_outputs = dict(
        zip(contract["outputs"], (_as_numpy(value) for value in eager_initial), strict=True)
    )
    eager_first_cache = kv_cache_from_outputs(initial_outputs)
    for name in first_cache:
        _check_close(eager_first_cache[name], _as_tensor(first_cache[name]), f"initial {name}")
    expected_first_frames = window - block_size
    if validate_kv_cache(first_cache) != expected_first_frames:
        raise RuntimeError("initial DiT call returned an unexpected cache length")

    offset = chunk_size
    nonempty_args = _dit_graph_args(
        noise[:, offset : offset + window],
        schedule[1:2],
        schedule[2:3],
        condition[:, offset : offset + window],
        speaker,
        offset,
        first_cache,
    )
    nonempty_metrics = compare_onnx_probe(
        wrapper,
        path,
        contract,
        nonempty_args,
        session=session,
        rtol=1e-4,
        atol=2e-5,
    )
    with torch.no_grad():
        eager_nonempty = wrapper(*nonempty_args)
    native_velocity, second_cache = _native_dit_step(
        tts,
        noise[:, offset : offset + window],
        schedule[1:2],
        schedule[2:3],
        condition[:, offset : offset + window],
        speaker,
        offset,
        first_cache,
    )
    _check_close(_as_numpy(eager_nonempty[0]), _as_tensor(native_velocity), "cached DiT velocity")
    nonempty_outputs = dict(
        zip(contract["outputs"], (_as_numpy(value) for value in eager_nonempty), strict=True)
    )
    eager_second_cache = kv_cache_from_outputs(nonempty_outputs)
    for name in second_cache:
        _check_close(eager_second_cache[name], _as_tensor(second_cache[name]), f"cached {name}")
    expected_second_frames = expected_first_frames + window - block_size
    if validate_kv_cache(second_cache) != expected_second_frames:
        raise RuntimeError("cached DiT call did not append current frames before lookahead trim")

    cache_observations = []

    def onnx_step(x, t, r, cond, spks, chunk_offset, cache):
        past = validate_kv_cache(cache)
        inputs = (
            x,
            t,
            r,
            cond,
            spks,
            np.asarray(chunk_offset, dtype=np.int64),
            *(cache[name] for name in CACHE_ABI["input_names"]),
        )
        feed = dict(zip(contract["inputs"], inputs, strict=True))
        values = session.run(list(contract["outputs"]), feed)
        outputs = dict(zip(contract["outputs"], values, strict=True))
        velocity = outputs["velocity"]
        updated = kv_cache_from_outputs(outputs)
        updated_frames = validate_kv_cache(updated)
        expected_frames = past + x.shape[1] - block_size
        if updated_frames != expected_frames:
            raise RuntimeError(
                f"DiT cache grew to {updated_frames}, expected {expected_frames} at offset {chunk_offset}"
            )
        if not np.isfinite(velocity).all() or not all(
            np.isfinite(value).all() for value in updated.values()
        ):
            raise RuntimeError("DiT ONNX produced non-finite velocity or cache values")
        cache_observations.append(
            (chunk_offset, past, x.shape[1], updated_frames)
        )
        return velocity, updated

    def native_step(x, t, r, cond, spks, chunk_offset, cache):
        return _native_dit_step(tts, x, t, r, cond, spks, chunk_offset, cache)

    frame_count = min(32, condition.shape[1])
    condition_window = condition[:, :frame_count]
    noise_window = noise[:, :frame_count]
    native_mel, native_final_cache = render_dit_mel(
        condition_window,
        noise_window,
        speaker,
        native_step,
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
    )
    onnx_mel, onnx_final_cache = render_dit_mel(
        condition_window,
        noise_window,
        speaker,
        onnx_step,
        steps=steps,
        chunk_size=chunk_size,
        block_size=block_size,
    )
    mel_error = np.abs(onnx_mel - native_mel)
    if not np.allclose(onnx_mel, native_mel, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"host solver DiT mel parity failed (max_abs={mel_error.max(initial=0.0):.8g})"
        )
    for name in native_final_cache:
        _check_close(
            onnx_final_cache[name],
            _as_tensor(native_final_cache[name]),
            f"final {name}",
            rtol=1e-4,
            atol=2e-5,
        )

    observed_by_offset = {}
    for observation in cache_observations:
        offset, past, window_frames, updated_frames = observation
        current = (past, window_frames, updated_frames)
        if offset in observed_by_offset and observed_by_offset[offset] != current:
            raise RuntimeError("solver substeps did not reuse the chunk-entry cache shape")
        observed_by_offset[offset] = current
    progression = [
        {"offset": offset, "past_frames": past, "window_frames": window_frames,
         "updated_frames": updated_frames}
        for offset, (past, window_frames, updated_frames) in sorted(observed_by_offset.items())
    ]
    return {
        "component_cases": [
            {
                "cache_state": "initial",
                "past_frames": 0,
                "updated_frames": validate_kv_cache(first_cache),
                "outputs": initial_metrics,
            },
            {
                "cache_state": "nonempty",
                "past_frames": validate_kv_cache(first_cache),
                "updated_frames": validate_kv_cache(second_cache),
                "outputs": nonempty_metrics,
            },
        ],
        "host_solver": {
            "steps": steps,
            "frames": frame_count,
            "mel_max_abs_error": float(mel_error.max(initial=0.0)),
            "cache_progression": progression,
        },
    }


def _vocos_mel(frames: int) -> np.ndarray:
    noise = np.random.default_rng(20261024 + frames).standard_normal(
        (1, 80, frames), dtype=np.float32
    )
    return (noise + np.float32(1.0)) / np.float32(2.0)


def _waveform_health(waveform: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(waveform).all():
        raise RuntimeError("Vocos waveform contains non-finite values")
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
    convergence = np.linalg.norm(difference) / max(np.linalg.norm(expected_magnitude), 1e-12)
    log_mae = np.mean(
        np.abs(np.log(actual_magnitude + 1e-7) - np.log(expected_magnitude + 1e-7))
    )
    return {
        "spectral_convergence": float(convergence),
        "log_spectral_mae": float(log_mae),
    }


def _check_vocos_parity(
    tts: Any,
    out_dir: Path,
    wrapper: VocosSpectrumWrapper,
) -> list[dict[str, Any]]:
    import onnxruntime as ort

    component = "vocos"
    contract = {**COMPONENT_CONTRACTS[component], "id": component}
    path = out_dir / contract["file"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    vocos_config = tts.config["vocos"]
    n_fft = int(vocos_config["n_fft"])
    hop_length = int(vocos_config["hop_length"])
    results = []
    for frames in (96, 192):
        mel = _vocos_mel(frames)
        args = (_as_tensor(mel),)
        parity = compare_onnx_probe(
            wrapper,
            path,
            contract,
            args,
            session=session,
            rtol=1e-4,
            atol=2.5e-4,
        )
        actual_magnitude, actual_phase = session.run(list(contract["outputs"]), {"mel": mel})
        repeated_spectrum = session.run(list(contract["outputs"]), {"mel": mel})
        if any(
            not np.array_equal(first, second)
            for first, second in zip((actual_magnitude, actual_phase), repeated_spectrum, strict=True)
        ):
            raise RuntimeError("Vocos ONNX spectrum is not deterministic")
        actual = vocos_istft_from_spectrum(
            actual_magnitude,
            actual_phase,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        with torch.no_grad():
            expected = _as_numpy(tts.vocos.decode(args[0]))
        expected_samples = (frames - 1) * hop_length
        if actual.shape != expected.shape or actual.shape != (1, expected_samples):
            raise RuntimeError(
                f"Vocos produced {actual.shape}, expected {(1, expected_samples)} samples"
            )
        max_abs_error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-4, atol=7e-5):
            raise RuntimeError(
                f"Vocos host iSTFT differs from native decode, max_abs_error={max_abs_error}"
            )
        actual_health = _waveform_health(actual)
        expected_health = _waveform_health(expected)
        for name in ("rms", "peak", "dc_offset"):
            if not np.isclose(actual_health[name], expected_health[name], rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"Vocos waveform {name} differs from native decode")
        results.append(
            {
                "mel_frames": frames,
                "waveform_samples": expected_samples,
                "outputs": parity,
                "max_abs_error": max_abs_error,
                "native_health": expected_health,
                "onnx_health": actual_health,
                "spectral_metrics": _spectral_metrics(actual, expected),
            }
        )
    return results


def export(*, snapshot: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Export pinned text, adapter, one-step DiT and Vocos boundaries."""
    from silencio import SilencioTTS

    resolved = validate_snapshot_pin(snapshot, spec)
    tts = SilencioTTS.from_pretrained(str(snapshot), device="cpu")
    text_encoder = TextEncoderWrapper(tts.text.ts)
    text_prosody = TextProsodyWrapper(tts.text.ts)
    adapter = AdapterWrapper(tts.adapter)
    dit_step = DiTStepWrapper(tts.dit)
    vocos_wrapper = VocosSpectrumWrapper(tts.vocos)
    wrappers = {
        "text_encoder": text_encoder,
        "text_prosody": text_prosody,
        "adapter": adapter,
        "dit_step": dit_step,
        "vocos": vocos_wrapper,
    }
    cases = _prepare_cases(tts, text_encoder, text_prosody, adapter)
    probe = cases[0]
    vocos_probe = (_as_tensor(_vocos_mel(192)),)

    dit_schedule = solver_schedule(int(spec["runtime"]["solver_steps"]))
    dit_condition = upsample_condition(
        _as_numpy(probe["content_features"]).astype(np.float32, copy=False)
    )
    dit_window = int(tts.config["chunk_size"] + tts.config["block_size"])
    dit_noise = np.random.default_rng(20261024).standard_normal(
        (1, dit_condition.shape[1], 80), dtype=np.float32
    )
    dit_speaker = _as_numpy(tts.voice(tts.config["default_voice"])).astype(np.float32).reshape(1, 256)
    dit_args = _dit_graph_args(
        dit_noise[:, :dit_window],
        dit_schedule[:1],
        dit_schedule[1:2],
        dit_condition[:, :dit_window],
        dit_speaker,
        0,
        empty_kv_cache(),
    )

    components = [
        export_component(
            text_encoder,
            (probe["phoneme_ids"],),
            out_dir / COMPONENT_CONTRACTS["text_encoder"]["file"],
            component="text_encoder",
            contract=COMPONENT_CONTRACTS["text_encoder"],
            opset=OPSET_VERSION,
        ),
        export_component(
            text_prosody,
            probe["prosody_inputs"],
            out_dir / COMPONENT_CONTRACTS["text_prosody"]["file"],
            component="text_prosody",
            contract=COMPONENT_CONTRACTS["text_prosody"],
            opset=OPSET_VERSION,
        ),
        export_component(
            adapter,
            probe["adapter_inputs"],
            out_dir / COMPONENT_CONTRACTS["adapter"]["file"],
            component="adapter",
            contract=COMPONENT_CONTRACTS["adapter"],
            opset=OPSET_VERSION,
        ),
        export_component(
            dit_step,
            dit_args,
            out_dir / COMPONENT_CONTRACTS["dit_step"]["file"],
            component="dit_step",
            contract=COMPONENT_CONTRACTS["dit_step"],
            opset=OPSET_VERSION,
            rtol=1e-4,
            atol=2e-5,
        ),
        export_component(
            vocos_wrapper,
            vocos_probe,
            out_dir / COMPONENT_CONTRACTS["vocos"]["file"],
            component="vocos",
            contract=COMPONENT_CONTRACTS["vocos"],
            opset=OPSET_VERSION,
            rtol=1e-4,
            atol=2.5e-4,
        ),
    ]
    dynamic_parity = _check_dynamic_parity(out_dir, wrappers, cases)
    dit_parity = _check_dit_parity(tts, out_dir, dit_step, probe, spec)
    vocos_parity = _check_vocos_parity(tts, out_dir, vocos_wrapper)
    from scripts.silencio_parity import run_parity_report

    parity_report = run_parity_report(tts=tts, component_dir=out_dir, spec=spec)
    return {
        "components": components,
        "parity_report": parity_report,
        "exporter": {
            "name": "silencio-current",
            "upstream_revision": resolved,
            "opset": OPSET_VERSION,
            "text_parity_cases": dynamic_parity,
            "dit_parity": dit_parity,
            "vocos_parity": vocos_parity,
        },
    }
