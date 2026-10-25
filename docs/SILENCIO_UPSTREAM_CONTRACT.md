# Silencio upstream inference contract

## Snapshot and evidence

This contract is based on `BlueRoseLabs/Silencio` at the immutable commit
`e200e2f9ef1887bb7b878045e0872d79a241f128`. The AST inventory covered all 16
Python files under `silencio/`. It is reproducible with:

```bash
python scripts/inspect_upstream_contract.py \
  --snapshot "$SNAPSHOT" \
  --output /tmp/silencio-upstream-contract.json

uv run --extra build \
  --with torchaudio \
  --with transformers \
  --with x-transformers==2.2.11 \
  --with einops \
  --with librosa \
  --with safetensors \
  python scripts/capture_upstream_observations.py \
  --snapshot "$SNAPSHOT" \
  --output /tmp/silencio-observations-full.json
```

The native trace loaded the exact snapshot's `config.json`, `model.safetensors`
and `voices.safetensors` on CPU. It used deterministic phoneme strings directly,
three solver steps and seed `20261024`; no reference recordings or external G2P
were used. Forward hooks recorded tensor dtype, rank, shape, finiteness and
summary statistics. The compact, source-development observation record is
[`SILENCIO_UPSTREAM_OBSERVATIONS.json`](SILENCIO_UPSTREAM_OBSERVATIONS.json).
It is not copied into the runtime bundle.

## Source map

Line numbers below refer to files at the pinned snapshot.

| Stage | Source and relevant code |
| --- | --- |
| Package/model loading | `silencio/tts.py:28-64`; `from_pretrained` loads `config.json`, strict-loads `model.safetensors`, and loads the fixed voice vectors from `voices.safetensors`. |
| Phonemes and token IDs | `silencio/text.py:28-43`; `phonemize` lazily uses Misaki, and `ids` maps known symbols, truncates at 510 phonemes, then adds ID 0 at both ends. |
| Text prediction and alignment | `silencio/text.py:46-67`; `silencio/paradee/student.py:19-51`; `silencio/paradee/kokoro_modules.py:175-218`. |
| Adapter and BNF length | `silencio/adapter.py:13-80`; Paradee text features are 40 Hz and adapter output is 25 Hz. |
| Voice conditioning and DiT | `silencio/meanvc2/dit.py:36-131,154-331`; global speaker memory and frame-level timbre conditioning feed the DiT. |
| DiT attention cache | `silencio/meanvc2/kv_modules.py:540-666,940-971`; block attention concatenates key/value history and omits the trailing lookahead block from the returned cache. |
| Host renderer and solver | `silencio/tts.py:89-111`; chunk scheduling, noise, timestep schedule, solver updates, cache persistence and mel scaling are all here. |
| Vocos and iSTFT | `silencio/vocos.py:25-81`; the head predicts magnitude and phase, then reconstructs waveform with a center-padded iSTFT. |
| Reference voice enrollment | `silencio/enroll.py:1-39`; WavLM-Large plus ECAPA emits the 256-dimensional vector, downloads external weights on first use and requires `s3prl`. Its module doc identifies about 1.3 GB of CC BY-SA 3.0 weights, separate from this snapshot. |
| Model dimensions/runtime defaults | pinned `config.json:121-148`; text model, DiT, Vocos, chunk size, block size, solver steps and default voice. |

## Native path and ownership

1. **Frontend and IDs.** The model-side text frontend accepts phonemes. The
   catalog's configured `kokorog2p` frontend, or upstream Misaki in its own
   Python API, belongs in the host. `TextEncoder.ids` maps symbols to int64
   IDs, truncates to 510 phonemes, and inserts leading/trailing zero IDs. Batch
   is one. No text normalization, G2P, sentence splitting or tokenization belongs
   in ONNX.
2. **Text encoder.** `TextEncoder.forward` produces token-conditioned ASR
   features and duration/prosody state. The predicted raw duration is
   `sum(sigmoid(duration_proj(bidirectional_lstm(state))))` over 50 duration
   bins. It rounds `duration / speed`, clamps each token to at least one frame,
   constructs a cumulative-duration alignment matrix in Python, and aligns both
   ASR features and the state used for pitch/loudness prediction. The
   `int(end[-1])`, `arange`, boolean alignment construction and matrix products
   are variable-length expansion logic, not a stable graph boundary.
3. **Host duration expansion and prosody.** Keep duration rounding and
   `repeat_interleave`/alignment in the host. The duration-state sequence is
   expanded before the source's `F0Ntrain` shared bidirectional LSTM and F0/N
   blocks. This requires a narrow prosody graph after host expansion if that
   operation remains host-owned. Pitch and loudness have two values per
   40-Hz text frame.
4. **Adapter.** The adapter combines aligned 512-channel ASR features with
   pitch/loudness, resamples each valid text span, then emits 256-channel BNF
   content. The host computes `out_len` with `bnf_frames` at
   `silencio/adapter.py:19-25`: `seconds=F/40`, `samples=round(seconds*16000)`,
   and `fbanks=1+(samples-400)//160` when `samples>=400`. It returns 0 below 19
   fbank frames, otherwise `((fbanks-19)//16+1)*4`. This is not
   `round(F*25/40)`. Batch is one in v1, so padding and per-item Python loops are
   unnecessary in the exported wrapper.
5. **Renderer preparation.** `render` converts `[T,256]` content to float32,
   linearly interpolates time to `4*T` with `align_corners=True`, and treats the
   result as a 100-Hz condition. It creates external `[1,4*T,80]` Gaussian noise.
   The ONNX step receives the noisy state, not a random seed or random operator.
6. **DiT and solver.** The host creates `steps+1` values with
   `linspace(1,0,steps+1)`. For each chunk it runs the one-step DiT three times
   by default and applies `x = x - (t-r)*u`. The DiT itself returns one velocity
   and per-layer updated attention cache; it does not own the solver loop.
7. **Chunk state.** `chunk_size=12`, `block_size=4`. Each non-final input window
   contains 12 current frames plus 4 lookahead frames. The host commits only the
   first 12 output frames. The final window can contain only 12 frames. `offset`
   advances by committed frames. Every solver substep in a chunk receives the
   same cache from the previous chunk; only the cache from the final substep is
   persisted for the next chunk. The separate DiT `cache` argument is always
   `None` in `SilencioTTS.render`; the persistent inference state is `kv_cache`.
8. **Speaker/reference conditioning.** The DiT consumes a precomputed float32
   speaker vector of width 256. It maps that vector to 32 global timbre memory
   slots and combines frame-level content/timbre features with the expanded
   global speaker vector. `voices.safetensors` contains eight precomputed
   float32 `[256]` vectors. Raw reference audio is not consumed by this graph
   path. Upstream `enroll.py` delegates extraction to WavLM-Large and ECAPA
   weights outside `model.safetensors`; its module doc identifies about 1.3 GB
   of CC BY-SA 3.0 weights downloaded from external repositories at runtime, with
   `s3prl` required. Source names `s3prl/converted_ckpts/wavlm_large.pt` and
   `bezzam/wavlm_large_finetune_seed_tts_eval/wavlm_large_finetune.pth`; neither
   external revision is pinned by this Silencio snapshot.
   Therefore the base ONNX ABI accepts a speaker embedding
   and can use the eight bundled voices. A consumer that promises arbitrary
   raw-audio enrollment must separately pin and package that encoder's weights,
   license notices and ONNX/host contract; it must not silently import upstream
   PyTorch or imply that the base bundle is self-contained for raw reference audio.
9. **Vocoder.** The host transposes the final DiT output and applies exactly
   `(mel + 1) / 2`, then calls `Vocos.decode`. The configured Vocos has 80 mel
   channels, 320 hidden channels, 8 backbone layers, `n_fft=640`,
   `hop_length=160` and `padding="center"`. The head predicts 321 log-magnitude
   and 321 phase channels, exponentiates and caps magnitude at 100, forms a
   complex spectrum, and calls `torch.istft` with a Hann window and center
   padding. The observed waveform length is `(mel_frames - 1) * 160`.

## Observed native tensor boundaries

All tensors below were observed as float32 except phoneme IDs and BNF target
lengths, which were int64. Batch was one throughout. The text and long-time axes
changed between the 9-token and 23-token runs.

| Boundary | Short trace | Longer trace | Meaning |
| --- | --- | --- | --- |
| Phoneme IDs | int64 `[1,9]` | int64 `[1,23]` | Includes two boundary IDs. |
| Duration-LSTM input state | float32 `[1,9,224]` | float32 `[1,23,224]` | Token state including the learned style features, before duration LSTM. |
| Duration-LSTM result | float32 `[1,9,192]` | float32 `[1,23,192]` | Bidirectional token features. |
| Duration projection | float32 `[1,9,50]` | float32 `[1,23,50]` | Sigmoid and sum across 50 channels yields raw duration per token. |
| ASR token features | float32 `[1,9,512]` | float32 `[1,23,512]` | Before host duration alignment. |
| Aligned text outputs | ASR `[1,512,45]`; pitch and loudness `[1,90]` | ASR `[1,512,79]`; pitch and loudness `[1,158]` | Text features at 40 Hz; pitch/loudness are 2x that frame count. |
| Expanded prosody-state input | float32 `[1,45,224]` | float32 `[1,79,224]` | Input to the shared F0/N bidirectional LSTM after duration alignment. |
| Prosody shared-LSTM output | float32 `[1,45,192]` | float32 `[1,79,192]` | Source intermediate before F0 and loudness heads. |
| Adapter inputs | ASR float32 `[1,512,45]`, pitch/loudness float32 `[1,90]`, length int64 `[1]` | ASR float32 `[1,512,79]`, pitch/loudness float32 `[1,158]`, length int64 `[1]` | `out_len` is 24 and 48 respectively. |
| Adapter content | float32 `[1,24,256]` | float32 `[1,48,256]` | 25-Hz BNF features. |
| Upsampled renderer condition | float32 `[1,96,256]` | float32 `[1,192,256]` | Host linear interpolation at 4x. |
| DiT input/output | float32 `[1,12 or 16,80]` | same | Final axis is 80 mel channels. Non-final windows include lookahead. |
| DiT condition | float32 `[1,12 or 16,256]` | same | Aligned content plus four lookahead frames in non-final windows. |
| DiT times | float32 `[1]` each for `t` and `r` | same | Values are 1, 2/3, 1/3, 0 across the three default updates. |
| DiT offset | Host Python integer, 0 then increments by 12 per full chunk | same | Passed to rotary-position construction; proposed ONNX ABI is an int64 scalar. |
| Speaker input | float32 `[1,256]` | same | Static global speaker vector; values change by selected voice. |
| Per-layer cache K/V | float32 `[1,2,P,64]` for each key and value | same | Four transformer layers; `P` is the explicit cache-frame axis. |
| Vocos input | float32 `[1,80,96]` | float32 `[1,80,192]` | Already rescaled by `(mel+1)/2`; values are not clamped to `[0,1]`. |
| Vocos backbone output | float32 `[1,96,320]` | float32 `[1,192,320]` | Input to ISTFT head. |
| Vocos waveform | float32 `[1,15200]` | float32 `[1,30560]` | `160 * (mel_frames - 1)` samples; finite in both runs. |

The short input used the default `librispeech_6930` voice and the same phonemes,
seed and shapes were rerun with `librispeech_1320`. Their speaker-vector
statistics and first DiT velocity statistics differed, confirming that voice
conditioning enters the DiT. Full measured values and cache progression are in
the observation JSON.

## Proposed ONNX ABI

The source has a real boundary missing from the starter's four-graph guess:
keeping duration expansion in the host requires the duration-state sequence to
be returned before alignment and fed to F0/N prediction after host expansion.
The proposed v1 graph set is therefore five components:

1. **`text_encoder.onnx`**
   - Input: `phoneme_ids`, int64 `[1, tokens]`.
   - Outputs: `asr_token_features`, float32 `[1, tokens, 512]`;
     `duration_state`, float32 `[1, tokens, 224]`; and `raw_durations`, float32
     `[1, tokens]`.
   - `raw_durations` is the source's sigmoid-and-sum result before speed
     adjustment. The learned text style and model weights are graph constants.
2. **Host duration expansion**
   - Compute `durations = max(round(raw_durations / speed), 1)` exactly as
     upstream.
   - Repeat both token-level outputs by those integer durations. Transpose the
     ASR features to `[1,512,F]`; keep duration state `[1,F,224]`.
   - Calculate `bnf_length` using the pinned `bnf_frames` implementation.
3. **`text_prosody.onnx`**
   - Input: `duration_state_frames`, float32 `[1,F,224]`.
   - Outputs: `pitch`, float32 `[1,2F]`, and `loudness`, float32 `[1,2F]`.
   - The trained 32-value text style is a frozen graph parameter. This graph is
     necessary because the native F0/N recurrent stack operates after duration
     expansion.
4. **`adapter.onnx`**
   - Inputs: `asr_features`, float32 `[1,512,F]`; `pitch` and `loudness`, each
     float32 `[1,2F]`; and `bnf_length`, int64 `[1]`.
   - Output: `content_features`, float32 `[1,T,256]`.
5. **Host renderer plus `dit_step.onnx` and `vocos.onnx`**
   - Host owns 4x condition interpolation, noise generation, chunk/lookahead
     slicing, the `t/r` schedule, Euler updates, offset, cache persistence, and
     final mel scaling.
   - `dit_step.onnx` represents one DiT evaluation. Inputs are `noisy_mel`,
     float32 `[1,window,80]`; `t` and `r`, float32 `[1]`; `condition`, float32
     `[1,window,256]`; `speaker_embedding`, float32 `[1,256]`;
     `position_offset`, int64 scalar; and four layer pairs of `cache_key_lN` and
     `cache_value_lN`, float32 `[1,2,past_frames,64]`. Outputs are `velocity`,
     float32 `[1,window,80]`, and corresponding updated key/value tensors,
     float32 `[1,2,updated_cache_frames,64]`. Source cache length is
     `past_frames + window - block_size`; the final `block_size` lookahead frames
     are excluded. Both cache axes are dynamic. At utterance start each input
     cache tensor has zero frames. Inference flags are constants, and the unused
     native `cache` argument is not in the ABI.
     `position_offset` must preserve
     the exact source rotary positions; if exporter tracing cannot make the
     offset dynamic, expose the exact rotary tensors as named host inputs rather
     than freezing a traced Python integer.
   - `vocos.onnx` accepts normalized float32 mel `[1,80,mel_frames]` and returns
     magnitude and phase, each float32 `[1,321,mel_frames]`. The ONNX graph runs
     the pinned Vocos backbone and spectrum head, including magnitude exp/clamp.
     Keep iSTFT in the host because the legacy TorchScript ONNX exporter fails on
     the complex input to `aten::istft` (`Unknown number type: complex`). Apply the
     configured periodic Hann window, centered overlap-add iSTFT with `n_fft=640`
     and `hop_length=160`; the exact waveform length is `160*(mel_frames-1)`.
     NumPy host reconstruction passes Torch iSTFT comparisons on synthetic spectra.
     Against native Vocos decode, observed max absolute errors are `3.32e-5` and
     `5.61e-5` for 96 and 192 mel frames. The parity record includes spectrum
     error, waveform health (RMS, peak and DC offset), spectral convergence and
     log-spectral MAE.

Batch remains fixed at one. `tokens`, text-frame, content-frame, DiT-window,
cache-frame and waveform axes are dynamic only where shown by source and trace.
No component accepts Python objects. All component input/output names and
shapes above are semantic ABI names, not native module parameter names.

## Contract constraints before export

- Do not trace the full `SilencioTTS.generate`/`render` orchestration.
- Preserve the host duration operation and the exact adapter target-frame
  formula. Test both short and longer text lengths.
- Flatten the four layer K/V pairs into named tensor inputs/outputs. Verify
  cache behavior on initial, subsequent and final chunks; the final cache is
  not reused by upstream after the utterance ends.
- Use the same external noise for native and ONNX parity. Do not generate noise
  inside ONNX.
- Keep G2P, sentence/chunk scheduling and the 3-step solver outside graphs.
- The model graph's speaker contract is a 256D embedding. The supplied voice
  vectors are inputs/assets, not the raw reference audio or an embedded speaker
  encoder. Any arbitrary-reference enrollment component must have its own
  pinned weights, explicit license/provenance and a separate ONNX/host contract.
- Vocos input is rescaled, not clamped. Test waveform length, finiteness, RMS,
  peak and DC offset against the native implementation.
