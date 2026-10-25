import subprocess
import sys
from pathlib import Path


def test_consumer_pipeline_uses_numpy_host_path_without_torch_or_upstream_imports():
    code = r'''
import sys
from pathlib import Path
import numpy as np
from scripts.silencio_parity import run_consumer_pipeline

class FakeSession:
    def __init__(self, component):
        self.component = component
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        if self.component == "text_encoder":
            tokens = feed["phoneme_ids"].shape[1]
            values = {
                "asr_token_features": np.zeros((1, tokens, 512), dtype=np.float32),
                "duration_state": np.zeros((1, tokens, 224), dtype=np.float32),
                "raw_durations": np.full((1, tokens), 15, dtype=np.float32),
            }
        elif self.component == "text_prosody":
            frames = feed["duration_state_frames"].shape[1]
            values = {
                "pitch": np.zeros((1, frames), dtype=np.float32),
                "loudness": np.zeros((1, frames), dtype=np.float32),
            }
        elif self.component == "adapter":
            frames = int(feed["bnf_length"][0])
            values = {"content_features": np.zeros((1, frames, 256), dtype=np.float32)}
        elif self.component == "dit_step":
            noisy = feed["noisy_mel"]
            past = feed["cache_key_l0"].shape[2]
            updated = past + noisy.shape[1] - 4
            values = {"velocity": np.zeros_like(noisy)}
            for name in names:
                if name.startswith("updated_cache_"):
                    values[name] = np.zeros((1, 2, updated, 64), dtype=np.float32)
        else:
            frames = feed["mel"].shape[2]
            values = {
                "magnitude": np.zeros((1, 321, frames), dtype=np.float32),
                "phase": np.zeros((1, 321, frames), dtype=np.float32),
            }
        return [values[name] for name in names]

sessions = {
    name: FakeSession(name)
    for name in ("text_encoder", "text_prosody", "adapter", "dit_step", "vocos")
}
result = run_consumer_pipeline(
    Path("unused"),
    np.array([[1, 2, 3]], dtype=np.int64),
    np.zeros((1, 96, 80), dtype=np.float32),
    np.zeros((1, 256), dtype=np.float32),
    steps=3,
    chunk_size=12,
    block_size=4,
    n_fft=640,
    hop_length=160,
    sessions=sessions,
)
assert result["mel"].shape == (1, 80, 96)
assert result["waveform"].shape == (1, 15200)
assert len(result["dit_steps"]) == 24
assert [step["past_frames"] for step in result["dit_steps"][:6]] == [0, 0, 0, 12, 12, 12]
assert "torch" not in sys.modules
assert "silencio" not in sys.modules
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
