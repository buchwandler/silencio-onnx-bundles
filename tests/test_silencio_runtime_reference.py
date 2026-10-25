import numpy as np
import torch

from scripts.silencio_runtime_reference import (
    CACHE_HEAD_DIM,
    CACHE_HEADS,
    CACHE_LAYERS,
    CONTENT_DIM,
    MEL_DIM,
    bnf_frames_for_text,
    durations_from_raw,
    empty_kv_cache,
    expand_text_outputs,
    iter_render_chunks,
    kv_cache_from_outputs,
    mel_for_vocos,
    render_dit_mel,
    solver_schedule,
    solver_update,
    upsample_condition,
    validate_kv_cache,
    vocos_istft_from_spectrum,
)


def test_bnf_frames_for_text_matches_observed_upstream_lengths():
    assert bnf_frames_for_text(45) == 24
    assert bnf_frames_for_text(79) == 48
    assert bnf_frames_for_text(0) == 0


def test_duration_rounding_and_host_feature_expansion():
    raw = np.array([[0.49, 0.5, 1.5, 2.5]], dtype=np.float32)
    durations = durations_from_raw(raw)
    np.testing.assert_array_equal(durations, np.array([1, 1, 2, 2], dtype=np.int64))

    asr = np.zeros((1, 3, 512), dtype=np.float32)
    state = np.zeros((1, 3, 224), dtype=np.float32)
    asr[0, :, 0] = [1, 2, 3]
    state[0, :, 0] = [10, 20, 30]
    aligned_asr, expanded_state = expand_text_outputs(
        asr, state, np.array([1, 2, 1], dtype=np.int64)
    )

    assert aligned_asr.shape == (1, 512, 4)
    assert expanded_state.shape == (1, 4, 224)
    np.testing.assert_array_equal(aligned_asr[0, 0], [1, 2, 2, 3])
    np.testing.assert_array_equal(expanded_state[0, :, 0], [10, 20, 20, 30])


def test_condition_interpolation_uses_align_corners_geometry():
    content = np.zeros((1, 2, CONTENT_DIM), dtype=np.float32)
    content[0, 1, 0] = 7

    upsampled = upsample_condition(content)

    assert upsampled.shape == (1, 8, CONTENT_DIM)
    np.testing.assert_allclose(
        upsampled[0, :, 0], np.linspace(0, 7, 8, dtype=np.float32), rtol=1e-6
    )


def test_solver_schedule_update_and_vocoder_mel_layout():
    np.testing.assert_allclose(
        solver_schedule(), np.array([1, 2 / 3, 1 / 3, 0], dtype=np.float32), rtol=1e-6
    )
    noisy = np.full((1, 2, MEL_DIM), 2, dtype=np.float32)
    velocity = np.full_like(noisy, 3)
    updated = solver_update(
        noisy,
        velocity,
        np.array([0.5], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
    )
    np.testing.assert_array_equal(updated, np.full_like(noisy, 1.25))

    mel = mel_for_vocos(updated)
    assert mel.shape == (1, MEL_DIM, 2)
    np.testing.assert_array_equal(mel, np.full((1, MEL_DIM, 2), 1.125, dtype=np.float32))


def test_chunk_windows_include_lookahead_but_commit_only_chunk_frames():
    condition = np.broadcast_to(
        np.arange(31, dtype=np.float32).reshape(1, 31, 1), (1, 31, CONTENT_DIM)
    ).copy()
    noise = np.broadcast_to(
        np.arange(31, dtype=np.float32).reshape(1, 31, 1), (1, 31, MEL_DIM)
    ).copy()

    chunks = list(iter_render_chunks(condition, noise))

    assert [
        (chunk.offset, chunk.committed_frames, chunk.noise.shape[1]) for chunk in chunks
    ] == [(0, 12, 16), (12, 12, 16), (24, 7, 7)]
    assert chunks[1].condition[0, 0, 0] == 12
    assert chunks[1].condition[0, -1, 0] == 27
    assert chunks[2].noise[0, -1, 0] == 30


def test_named_cache_abi_has_empty_initial_state_and_maps_step_outputs():
    cache = empty_kv_cache()

    assert len(cache) == 2 * CACHE_LAYERS
    assert validate_kv_cache(cache) == 0
    assert all(tensor.shape == (1, CACHE_HEADS, 0, CACHE_HEAD_DIM) for tensor in cache.values())

    outputs = {
        f"updated_{name}": np.ones((1, CACHE_HEADS, 5, CACHE_HEAD_DIM), dtype=np.float32)
        for name in cache
    }
    next_cache = kv_cache_from_outputs(outputs)
    assert set(next_cache) == set(cache)
    assert validate_kv_cache(next_cache) == 5
    reset_cache = empty_kv_cache()
    assert validate_kv_cache(reset_cache) == 0
    assert reset_cache["cache_key_l0"] is not cache["cache_key_l0"]


def test_render_keeps_chunk_cache_constant_across_solver_substeps():
    condition = np.zeros((1, 31, CONTENT_DIM), dtype=np.float32)
    noise = np.zeros((1, 31, MEL_DIM), dtype=np.float32)
    speaker = np.zeros((1, CONTENT_DIM), dtype=np.float32)
    calls = []

    def dit_step(noisy, t, r, cond, spks, offset, cache):
        assert spks.shape == (1, CONTENT_DIM)
        cache_frames = validate_kv_cache(cache)
        marker = None if cache_frames == 0 else float(cache["cache_key_l0"][0, 0, 0, 0])
        calls.append((offset, float(t[0]), float(r[0]), cache_frames, marker, noisy.shape[1], cond.shape[1]))
        next_frames = cache_frames + noisy.shape[1] - 4
        marker_value = np.float32(len(calls))
        next_cache = {
            name: np.full(
                (1, CACHE_HEADS, next_frames, CACHE_HEAD_DIM), marker_value, dtype=np.float32
            )
            for name in cache
        }
        return np.zeros_like(noisy), next_cache

    mel, final_cache = render_dit_mel(condition, noise, speaker, dit_step)

    assert mel.shape == (1, MEL_DIM, 31)
    np.testing.assert_array_equal(mel, np.full_like(mel, 0.5))
    assert [call[0] for call in calls] == [0] * 3 + [12] * 3 + [24] * 3
    expected_times = np.tile(
        np.array([[1, 2 / 3], [2 / 3, 1 / 3], [1 / 3, 0]], dtype=np.float32),
        (3, 1),
    )
    np.testing.assert_allclose(
        np.array([[call[1], call[2]] for call in calls], dtype=np.float32),
        expected_times,
        rtol=1e-6,
    )
    assert [call[3] for call in calls] == [0] * 3 + [12] * 3 + [24] * 3
    assert [call[4] for call in calls] == [None] * 3 + [3.0] * 3 + [6.0] * 3
    assert [call[5] for call in calls] == [16] * 6 + [7] * 3
    assert [call[6] for call in calls] == [16] * 6 + [7] * 3
    assert validate_kv_cache(final_cache) == 27
    assert final_cache["cache_key_l0"][0, 0, 0, 0] == 9



def test_vocos_istft_matches_torch_for_multiple_dynamic_lengths():
    rng = np.random.default_rng(20261024)
    window = torch.hann_window(640)

    for frame_count in (96, 192):
        magnitude = np.exp(rng.normal(0.0, 0.1, (1, 321, frame_count))).astype(np.float32)
        phase = rng.normal(0.0, 1.0, magnitude.shape).astype(np.float32)
        actual = vocos_istft_from_spectrum(magnitude, phase)

        torch_magnitude = torch.from_numpy(magnitude)
        torch_phase = torch.from_numpy(phase)
        spectrum = torch_magnitude * (torch.cos(torch_phase) + 1j * torch.sin(torch_phase))
        expected = torch.istft(
            spectrum,
            n_fft=640,
            hop_length=160,
            win_length=640,
            window=window,
            center=True,
        )

        assert actual.shape == (1, 160 * (frame_count - 1))
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)
