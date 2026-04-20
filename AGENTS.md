# AGENTS.md

## Project Overview

qwen3-asr.cpp is a C++ GGML implementation of Qwen3-ASR and Qwen3-ForcedAligner.
It performs speech recognition and forced alignment with word-level timestamps, with full support for F16, Q8_0, and NVFP4 quantized models.

## Architecture

### Core Components

- `src/tools/asr_cli.cpp` — CLI entry point, mode dispatch (transcribe-only, align-only, combined)
- `src/asr/transcribe/encoder.cpp/h/pp` — ASR audio encoder (conv frontend via im2col+mul_mat, chunked processing, 18-layer transformer)
- `src/asr/transcribe/decoder.cpp/h/pp` — ASR text decoder (Qwen2, 28 layers, GQA, KV cache, flash attention, audio embedding injection)
- `src/asr/aligner/encoder.cpp/h/pp` — Aligner audio encoder (conv frontend via im2col+mul_mat, chunked batched processing, 24-layer windowed-attention transformer)
- `src/asr/aligner/decoder.cpp/h/pp` — Aligner decoder (Qwen2, classify_head, causal attention, timestamp extraction, Korean/CJK tokenization)
- `src/asr/mel/mel.cpp/h` — Mel spectrogram computation (n_fft=400, hop=160, n_mels=128)
- `src/asr/codec/codec.cpp/h` + `wav_loader.cpp/h` — Audio loading (WAV/ffmpeg), normalization
- `src/asr/common/types.hpp` — Shared types (MelSpectrum, AudioFeatures, AlignedWord, ErrorInfo)
- `src/asr/common/hf_tokenizer.hpp` — HF JSON tokenizer fallback (vocab/merges/added_tokens parsing)
- `src/tools/quantize.cpp` — GGUF quantization tool (F16 → NVFP4/Q8_0/Q4_0)
- `src/logger.cpp/h` — spdlog-based logging
- `src/timing.h` — Timing instrumentation macros (QWEN3_ASR_TIMING)

### Model Architecture

**ASR encoder** (Whisper-style):
- 3 Conv2d stride=2 (via im2col+mul_mat) → 480ch → linear proj → d_model=896
- Chunked processing (chunk_size=100 mel frames, ~13 encoder frames per chunk)
- 18-layer transformer with biases, sinusoidal PE, proj1+GELU+proj2 → hidden=1024

**ASR decoder** (Qwen2):
- 28 layers, GQA (n_heads=32, n_kv_heads=8, head_dim=128), hidden=1024
- SwiGLU FFN (intermediate=3072), Q/K norm + RoPE (theta=1M)
- Flash attention (F32 precision), F16 KV cache
- Weight tying (token_embd = output weight)
- Audio embedding injection at audio_pad positions

**Aligner encoder** (larger conv frontend):
- 3 Conv2d stride=2 (via im2col+mul_mat) → 480ch → linear proj → d_model=1024
- Batched chunk processing (all chunks in single graph call)
- 24-layer transformer with **windowed attention** (block-diagonal mask, window_aftercnn=104)
- proj1+GELU+proj2 → hidden=1024

**Aligner decoder** (Qwen2 variant):
- Same 28-layer Qwen2 architecture, but **CAUSAL attention** (mandatory, model was trained with is_causal=True)
- classify_head (5000 classes × 80ms = 400s range) instead of lm_head
- No KV cache — single forward pass, non-autoregressive
- Each word gets 2 `<|timestamp|>` tokens (start+end), argmax → class index → timestamp

### Key Design Decisions

- **GGML tensor library** (not PyTorch/ONNX) for minimal dependencies
- **Multi-backend GPU support** (Metal on macOS, CUDA on Linux) with ggml_backend_sched
- **Device selection** via `--device <name>` for multi-GPU (CUDA0, CUDA1, Metal)
- **mmap weight loading** with zero-copy GPU transfer
- **Conv2d via im2col+mul_mat**: ggml_conv_2d op doesn't support quantized kernels; manual im2col(F16)+mul_mat lets quantized kernels compute directly via mul_mat's quantized kernel path
- **F16 KV cache** to reduce memory bandwidth
- **Flash attention** (`ggml_flash_attn_ext`, F32 precision) for decode speedup
- **Weight tying** (token_embd = output weight) to save memory
- **GGUF key name fallback**: supports both `qwen3-asr.*` (hyphen) and `qwen3_asr.*` (underscore) GGUF naming
- **HF tokenizer JSON fallback**: when `tokenizer.ggml.tokens` is absent, parses `tokenizer.huggingface.json` (11MB HF blob) for vocab/merges/added_tokens
- **attn_output.weight excluded from NVFP4**: quantizing this layer to NVFP4 causes severe inference degradation (repetitive generation); must remain F16/Q8_0
- **NVFP4 block_size=64 constraint**: tensor ne[0] must be multiple of 64; conv weights and other small-dimension tensors stay F16
- **Korean word splitting** (soynlp LTokenizer, 17,968-word dictionary)
- **CJK tokenization**: CJK chars split individually, Latin sequences preserved (e.g., "我是AI工程师" → ["我", "是", "AI", "工", "程", "师"])

### Build System

- CMake 3.14+, C++17
- GGML as git submodule at `./ggml` (auto-initialized if missing)
- spdlog fetched via FetchContent
- **GPU backends**: macOS Metal (Accelerate/vDSP for mel), Linux CUDA (auto-detected at `/usr/local/cuda`)
- Build targets: `asr-cli` (CLI), `ggml-quantize` (quantization tool)
- **Test builds**: `QWEN3_ASR_BUILD_TESTS=ON` (default OFF)

#### Build Options

```bash
# Standard build (with CUDA)
cmake -B build && cmake --build build -j$(nproc)

# Disable CUDA explicitly
cmake -B build -DGGML_CUDA=OFF && cmake --build build -j$(nproc)

# Enable timing instrumentation
cmake -B build -DQWEN3_ASR_TIMING=ON && cmake --build build -j$(nproc)

# Enable tests
cmake -B build -DQWEN3_ASR_BUILD_TESTS=ON && cmake --build build -j$(nproc)
```

### Quantization Tool

```bash
# Quantize F16 → NVFP4 (RTX 5080/Blackwell recommended)
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-nvfp4.gguf nvfp4

# Quantize F16 → Q8_0 (universal, all GPUs)
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-q8_0.gguf q8_0

# Quantize F16 → Q4_0
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-q4_0.gguf q4_0
```

**Quantization rules** (`should_quantize_tensor()`):
- Quantize: all F16/F32 weight matrices with nrows>1 and ne[0] divisible by block_size
- Keep F16/F32: norm weights/biases, all `.bias`, embed_tokens/token_embd, lm_head, output.weight (exact match), classify_head, ln_post, proj biases, conv biases, **attn_output.weight**
- NVFP4 additionally skips tensors where ne[0] % 64 != 0 (block_size=64)

### Testing

- **Test audio**: any 16kHz mono WAV (`ffmpeg -i input.wav -ar 16000 -ac 1 output.wav`)
- **CLI ASR**: `./build/asr-cli --input audio.wav --model models/qwen3-asr-1.7b-f16.gguf --transcribe-only`
- **CLI Align**: `./build/asr-cli --input audio.wav --model models/qwen3-forced-aligner-0.6b-f16.gguf --align-only --text "text" --language korean`
- **CLI Combined**: `./build/asr-cli --input audio.wav --model models/qwen3-asr-1.7b-f16.gguf --aligner models/qwen3-forced-aligner-0.6b-f16.gguf`
- **NVFP4**: same commands with nvfp4 model paths

### Conventions

- **Namespace**: `asr::transcribe::encoder`, `asr::transcribe::decoder`, `asr::aligner::encoder`, `asr::aligner::decoder`
- **Error handling**: bool return + error_msg_ member (ErrorInfo struct)
- **Timing**: chrono-based timing in asr_cli.cpp; `QWEN3_TIMER_SCOPED` macros when `QWEN3_ASR_TIMING` defined
- **Memory**: RAII with explicit cleanup; mmap cleanup via munmap
- **Tensor naming**: GGUF keys follow GGML convention (blk.*, audio.encoder.*); fallback to HF naming (model.layers.*, thinker.*)

### Important Caveats

- The aligner decoder MUST use causal attention (model was trained with `self_attn.is_causal: True`)
- The aligner encoder uses windowed attention (block-diagonal mask, window_aftercnn=104)
- attn_output.weight must NOT be quantized to NVFP4 — causes repetitive/infinite generation (keep F16)
- NVFP4 requires block_size=64: any tensor with ne[0] not divisible by 64 stays F16
- im2col output type must be F16 (CUDA doesn't support Q8_0 im2col output)
- Korean word splitting requires `assets/korean_dict_jieba.dict` for optimal alignment (fallback: character-level)
- ASR output starts with "language <Name>" prefix — must be stripped before alignment
- Special token IDs: audio_start=151669, audio_end=151670, audio_pad=151676, timestamp=151705
- Audio must be 16kHz mono PCM/WAV
- Device names: CUDA0, CUDA1, Metal, CPU

## Performance Notes

Benchmark on 120-second Chinese audio, NVIDIA RTX 5080 (Blackwell):

| Stage | F16 | NVFP4 | Ratio |
|-------|-----|-------|-------|
| ASR Init | 5950ms | 1125ms | 0.19x (mmap+less data) |
| Audio encode | 352ms | 313ms | 0.89x |
| Text decode (190 tokens) | 2028ms | 1910ms | 0.94x |
| Aligner Init | 2342ms | 327ms | 0.14x |
| Aligner encode | 176ms | 139ms | 0.79x |
| Aligner decode (237 words) | 6059ms | 5937ms | 0.98x |

**Memory reduction**: ASR 4.48GB → 2.30GB (48.7%), Aligner 1.75GB → 0.81GB (53.7%)

NVFP4 on Blackwell: encode speed comparable or faster than F16; decode ~7% slower; main benefit is memory reduction.

## Development Guidelines

### When Adding Features

1. Maintain compatibility with GGML submodule API
2. Profile before optimizing using `QWEN3_ASR_TIMING` build flag
3. Follow existing patterns for error handling and memory management
4. Test on real audio (not just synthetic data)
5. For any new conv_2d usage, use `conv_2d_via_im2col` to support quantized models

### Common Pitfalls

- Don't use `ggml_conv_2d` directly — quantized kernels won't work, always use `conv_2d_via_im2col`
- Don't quantize attn_output.weight to low-bit types (NVFP4/Q4_0) — causes inference quality collapse
- Don't break mmap compatibility (weight tensors must use mmap buffer)
- Flash attention requires contiguous memory layout
- Korean dictionary must be UTF-8 encoded
- Audio must be 16kHz mono PCM (ffmpeg conversion required otherwise)
- HF tokenizer JSON fallback handles byte-to-unicode BPE encoding — match Python implementation exactly
- GGUF key names vary between F16 (GGML convention) and Q8_0 (HF convention) — both must be supported

### File Organization

```
src/
  asr/
    transcribe/    encoder.cpp/h, encoder_model.hpp, decoder.cpp/h, decoder_model.hpp
    aligner/       encoder.cpp/h, encoder_model.hpp, decoder.cpp/h, decoder_model.hpp
    mel/           mel.cpp/h
    codec/         codec.cpp/h, wav_loader.cpp/h
    common/        types.hpp, hf_tokenizer.hpp
  tools/           asr_cli.cpp, quantize.cpp
  logger.cpp/h
  timing.h
```

## Contributing

When making changes:
1. Run existing tests to verify no regression (`QWEN3_ASR_BUILD_TESTS=ON`)
2. Add new tests for new functionality
3. Update AGENTS.md if architecture changes
4. Benchmark on both F16 and NVFP4 models
5. Verify inference quality (especially text output) after any quantization-related changes

## Resources

- [GGML Documentation](https://github.com/ggerganov/ggml)
- [Qwen3-ASR HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- Korean dictionary: 17,968 words from soynlp corpus (`assets/korean_dict_jieba.dict`)