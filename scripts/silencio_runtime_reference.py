from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

TEXT_ASR_DIM = 512
DURATION_STATE_DIM = 224
CONTENT_DIM = 256
MEL_DIM = 80
DURATION_BINS = 50
CONDITION_UPSAMPLE_FACTOR = 4
CACHE_ABI_VERSION = 1
CACHE_LAYERS = 4
CACHE_HEADS = 2
CACHE_HEAD_DIM = 64

KVCache = Mapping[str, np.ndarray]
DiTStep = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, KVCache],
    tuple[np.ndarray, KVCache],
]


def bnf_frames_for_text(text_frames: int) -> int:
    """Apply the pinned adapter's Fast-U2++ target-length formula to 40-Hz text frames."""
    if text_frames < 0:
        raise ValueError("text_frames must be non-negative")
    seconds = text_frames / 40
    samples = int(round(seconds * 16000))
    fbank_frames = 1 + (samples - 400) // 160 if samples >= 400 else 0
    if fbank_frames < 19:
        return 0
    return ((fbank_frames - 19) // 16 + 1) * 4


@dataclass(frozen=True)
class RenderChunk:
    offset: int
    committed_frames: int
    noise: np.ndarray
    condition: np.ndarray


def durations_from_raw(raw_durations: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Match TextEncoder's round(duration / speed), clamp-to-one, and int64 conversion."""
    if raw_durations.dtype != np.float32 or raw_durations.ndim != 2 or raw_durations.shape[0] != 1:
        raise ValueError("raw_durations must be float32 [1,tokens]")
    if raw_durations.shape[1] == 0:
        raise ValueError("raw_durations must contain at least one token")
    speed_value = np.float32(speed)
    if not np.isfinite(speed_value) or speed_value <= 0:
        raise ValueError("speed must be finite and greater than zero")
    rounded = np.rint(raw_durations / speed_value)
    return np.maximum(rounded, 1).astype(np.int64)[0]


def expand_text_outputs(
    asr_token_features: np.ndarray,
    duration_state: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat token features by duration and return ASR channel-first plus prosody state."""
    if asr_token_features.dtype != np.float32 or asr_token_features.ndim != 3:
        raise ValueError("asr_token_features must be float32 [1,tokens,512]")
    if asr_token_features.shape[0] != 1 or asr_token_features.shape[2] != TEXT_ASR_DIM:
        raise ValueError("asr_token_features must be float32 [1,tokens,512]")
    if duration_state.dtype != np.float32 or duration_state.ndim != 3:
        raise ValueError("duration_state must be float32 [1,tokens,224]")
    if duration_state.shape[0] != 1 or duration_state.shape[2] != DURATION_STATE_DIM:
        raise ValueError("duration_state must be float32 [1,tokens,224]")
    if durations.dtype != np.int64 or durations.ndim != 1:
        raise ValueError("durations must be int64 [tokens]")
    tokens = asr_token_features.shape[1]
    if tokens == 0 or duration_state.shape[1] != tokens or durations.shape[0] != tokens:
        raise ValueError("text feature and duration token axes must match and be non-empty")
    if np.any(durations < 1):
        raise ValueError("durations must be positive")

    asr_frames = np.repeat(asr_token_features, durations, axis=1).transpose(0, 2, 1)
    state_frames = np.repeat(duration_state, durations, axis=1)
    return asr_frames, state_frames


def upsample_condition(content: np.ndarray) -> np.ndarray:
    """Match linear interpolation to 4x length with align_corners=True."""
    if content.dtype != np.float32 or content.ndim != 3:
        raise ValueError("content must be float32 [1,frames,256]")
    if content.shape[0] != 1 or content.shape[2] != CONTENT_DIM:
        raise ValueError("content must be float32 [1,frames,256]")
    frames = content.shape[1]
    if frames == 0:
        raise ValueError("content must contain at least one frame")
    output_frames = frames * CONDITION_UPSAMPLE_FACTOR
    scale = np.float32(frames - 1) / np.float32(output_frames - 1)
    positions = np.arange(output_frames, dtype=np.float32) * scale
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, frames - 1)
    weight = (positions - left.astype(np.float32)).reshape(1, output_frames, 1)
    return content[:, left, :] * (np.float32(1) - weight) + content[:, right, :] * weight


def solver_schedule(steps: int = 3) -> np.ndarray:
    """Return the float32 Euler schedule used by SilencioTTS.render."""
    if steps < 1:
        raise ValueError("steps must be at least one")
    return np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)


def solver_update(
    noisy_mel: np.ndarray,
    velocity: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Apply upstream's Euler update, x = x - (t-r)*u."""
    if (
        noisy_mel.dtype != np.float32
        or noisy_mel.ndim != 3
        or noisy_mel.shape[0] != 1
        or noisy_mel.shape[2] != MEL_DIM
    ):
        raise ValueError("noisy_mel must be float32 [1,window,80]")
    if velocity.dtype != np.float32 or velocity.shape != noisy_mel.shape:
        raise ValueError("velocity must be float32 with the same shape as noisy_mel")
    if t.dtype != np.float32 or t.shape != (1,) or r.dtype != np.float32 or r.shape != (1,):
        raise ValueError("t and r must be float32 [1]")
    return noisy_mel - (t - r) * velocity


def iter_render_chunks(
    condition: np.ndarray,
    noise: np.ndarray,
    *,
    chunk_size: int = 12,
    block_size: int = 4,
) -> Iterator[RenderChunk]:
    """Yield source-equivalent chunk and lookahead slices with committed frame counts."""
    if condition.dtype != np.float32 or condition.ndim != 3:
        raise ValueError("condition must be float32 [1,frames,256]")
    if condition.shape[0] != 1 or condition.shape[2] != CONTENT_DIM:
        raise ValueError("condition must be float32 [1,frames,256]")
    if noise.dtype != np.float32 or noise.ndim != 3:
        raise ValueError("noise must be float32 [1,frames,80]")
    if noise.shape[0] != 1 or noise.shape[2] != MEL_DIM:
        raise ValueError("noise must be float32 [1,frames,80]")
    if noise.shape[1] != condition.shape[1] or condition.shape[1] == 0:
        raise ValueError("condition and noise must have the same non-empty frame axis")
    if chunk_size < 1 or block_size < 0:
        raise ValueError("chunk_size must be positive and block_size must be non-negative")

    frames = condition.shape[1]
    for offset in range(0, frames, chunk_size):
        committed_end = min(offset + chunk_size, frames)
        window_end = min(committed_end + block_size, frames)
        yield RenderChunk(
            offset=offset,
            committed_frames=committed_end - offset,
            noise=noise[:, offset:window_end],
            condition=condition[:, offset:window_end],
        )


def _cache_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_l{layer}" for layer in range(CACHE_LAYERS))


def empty_kv_cache() -> dict[str, np.ndarray]:
    """Create the zero-frame cache tensors used for the initial ONNX step."""
    return {
        name: np.empty((1, CACHE_HEADS, 0, CACHE_HEAD_DIM), dtype=np.float32)
        for prefix in ("cache_key", "cache_value")
        for name in _cache_names(prefix)
    }


def validate_kv_cache(cache: KVCache) -> int:
    """Validate the four-layer named cache ABI and return its past-frame length."""
    expected = set(_cache_names("cache_key")) | set(_cache_names("cache_value"))
    if set(cache) != expected:
        raise ValueError(f"cache keys must be exactly {sorted(expected)}")

    past_frames: int | None = None
    for name in sorted(expected):
        tensor = cache[name]
        if tensor.dtype != np.float32 or tensor.ndim != 4:
            raise ValueError(f"{name} must be float32 [1,2,cache_frames,64]")
        if tensor.shape[:2] != (1, CACHE_HEADS) or tensor.shape[3] != CACHE_HEAD_DIM:
            raise ValueError(f"{name} must be float32 [1,2,cache_frames,64]")
        if past_frames is None:
            past_frames = tensor.shape[2]
        elif tensor.shape[2] != past_frames:
            raise ValueError("all key/value cache tensors must have the same frame length")
    return 0 if past_frames is None else past_frames


def kv_cache_from_outputs(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename DiT cache outputs to the matching next-step cache inputs."""
    cache = {
        input_name: outputs[f"updated_{input_name}"]
        for prefix in ("cache_key", "cache_value")
        for input_name in _cache_names(prefix)
    }
    validate_kv_cache(cache)
    return cache


def mel_for_vocos(noisy_mel_frames: np.ndarray) -> np.ndarray:
    """Transpose final solver frames and apply the source's exact (mel + 1) / 2 scale."""
    if noisy_mel_frames.dtype != np.float32 or noisy_mel_frames.ndim != 3:
        raise ValueError("noisy_mel_frames must be float32 [1,frames,80]")
    if noisy_mel_frames.shape[0] != 1 or noisy_mel_frames.shape[2] != MEL_DIM:
        raise ValueError("noisy_mel_frames must be float32 [1,frames,80]")
    return (noisy_mel_frames.transpose(0, 2, 1) + 1.0) / 2.0



def vocos_istft_from_spectrum(
    magnitude: np.ndarray,
    phase: np.ndarray,
    *,
    n_fft: int = 640,
    hop_length: int = 160,
) -> np.ndarray:
    """Reconstruct centered Vocos iSTFT output from its real magnitude and phase tensors."""
    if magnitude.dtype != np.float32 or magnitude.ndim != 3:
        raise ValueError("magnitude must be float32 [1,n_fft/2+1,frames]")
    if phase.dtype != np.float32 or phase.shape != magnitude.shape:
        raise ValueError("phase must be float32 with the same shape as magnitude")
    if magnitude.shape[0] != 1 or magnitude.shape[1] != n_fft // 2 + 1:
        raise ValueError("spectrum must be float32 [1,n_fft/2+1,frames]")
    if n_fft < 2 or n_fft % 2 or hop_length < 1 or hop_length > n_fft:
        raise ValueError("n_fft must be even and hop_length must be within [1,n_fft]")

    frame_count = magnitude.shape[2]
    window_positions = np.arange(n_fft, dtype=np.float32)
    window = np.float32(0.5) - np.float32(0.5) * np.cos(
        np.float32(2 * np.pi / n_fft) * window_positions
    )
    spectrum = magnitude * (np.cos(phase) + np.complex64(1j) * np.sin(phase))
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1, norm="backward")
    frames = frames * window[None, :, None]

    padded_length = (frame_count - 1) * hop_length + n_fft
    waveform = np.zeros((1, padded_length), dtype=np.float32)
    envelope = np.zeros(padded_length, dtype=np.float32)
    window_envelope = window * window
    for frame in range(frame_count):
        start = frame * hop_length
        end = start + n_fft
        waveform[:, start:end] += frames[:, :, frame]
        envelope[start:end] += window_envelope

    padding = n_fft // 2
    waveform = waveform[:, padding : padded_length - padding]
    envelope = envelope[padding : padded_length - padding]
    return waveform / np.maximum(envelope[None, :], np.float32(1e-11))


def render_dit_mel(
    condition: np.ndarray,
    noise: np.ndarray,
    speaker_embedding: np.ndarray,
    dit_step: DiTStep,
    *,
    steps: int = 3,
    chunk_size: int = 12,
    block_size: int = 4,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run host-owned chunking and Euler steps around a one-step DiT callable."""
    if speaker_embedding.dtype != np.float32 or speaker_embedding.shape != (1, CONTENT_DIM):
        raise ValueError("speaker_embedding must be float32 [1,256]")
    times = solver_schedule(steps)
    cache = empty_kv_cache()
    rendered_chunks = []
    offset = 0

    for chunk in iter_render_chunks(
        condition, noise, chunk_size=chunk_size, block_size=block_size
    ):
        noisy_mel = chunk.noise
        for step in range(steps):
            t = times[step : step + 1]
            r = times[step + 1 : step + 2]
            velocity, cache_after_step = dit_step(
                noisy_mel,
                t,
                r,
                chunk.condition,
                speaker_embedding,
                offset,
                cache,
            )
            validate_kv_cache(cache_after_step)
            noisy_mel = solver_update(noisy_mel, velocity, t, r)
        cache = dict(cache_after_step)
        rendered_chunks.append(noisy_mel[:, : chunk.committed_frames])
        offset += chunk.committed_frames

    return mel_for_vocos(np.concatenate(rendered_chunks, axis=1)), cache
