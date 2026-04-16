# AGENTS.md

## Project Overview

qwen3-asr.cpp is a C++ GGML implementation of Qwen3-ASR and Qwen3-ForcedAligner.
It performs speech recognition and forced alignment with word-level timestamps.

## Architecture

### Core Components

- `src/main.cpp` — CLI entry point, mode dispatch (transcription, alignment, combined)
- `src/qwen3_asr.cpp/h` — High-level ASR orchestration (mel → encoder → decoder)
- `src/forced_aligner.cpp/h` — Forced aligner (separate encoder + decoder, chunked convolution, BPE tokenizer, Korean word splitting)
- `src/text_decoder.cpp/h` — Qwen2-based text decoder with KV cache, flash attention, RoPE
- `src/audio_encoder.cpp/h` — Audio feature encoder with GPU backend support (Metal/CUDA)
- `src/mel_spectrogram.cpp/h` — Mel spectrogram computation (vDSP/Accelerate/MKL/OpenBLAS optimized)
- `src/audio_injection.cpp/h` — Audio embedding injection into token sequence
- `src/audio_utils.cpp/h` — Audio processing utilities (normalization, chunking, repetition fix, output parsing)
- `src/gguf_loader.cpp/h` — GGUF model file loading with mmap
- `src/qwen3asr_c_api.cpp/h` — C API for shared library with device selection support
- `src/http_server/` — HTTP server implementations (align_server, transcribe_server)

### Model Architecture

**ASR encoder**: Whisper-style audio encoder (conv frontend + transformer)

**ASR decoder**: Qwen2 (28 layers, GQA 16/4 heads, head_dim=64, hidden=1024, RoPE theta=1M)

**Forced aligner encoder**: Larger conv frontend (3x Conv2d stride=2 → 480ch → linear 1024) + 24-layer transformer with windowed attention

**Forced aligner decoder**: Same Qwen2 architecture but with classify_head (5000 classes × 80ms = 400s range) instead of lm_head, uses CAUSAL attention

### Key Design Decisions

- **GGML tensor library** (not PyTorch/ONNX) for minimal dependencies
- **Multi-backend GPU support** (Metal on macOS, CUDA on Linux) with ggml_backend_sched for optimal placement
- **Device selection** via `--device <name>` for multi-GPU environments (e.g., CUDA0, CUDA1, Metal)
- **mmap weight loading** with zero-copy GPU transfer via `ggml_backend_dev_buffer_from_host_ptr`
- **F16 KV cache** to reduce memory bandwidth
- **Flash attention** (`ggml_flash_attn_ext`) for decode speedup
- **Weight tying** (token_embd = output weight) to save memory
- **Korean word splitting** ported from soynlp LTokenizer with bundled dictionary (optional but recommended for Korean alignment)
- **Chinese/Japanese tokenization** matches Python implementation: CJK chars split individually, Latin sequences preserved (e.g., "我是AI工程师" → ["我", "是", "AI", "工", "程", "师"])
- **Audio normalization** (`float_range_normalize`): handles peak > 1.0 (int-scaled audio) by normalizing to [-1, 1]
- **Audio chunking** (`split_audio_into_chunks`): low-energy boundary detection for smart splitting of long audio
- **Repetition detection** (`detect_and_fix_repetitions`): fixes character and pattern repeats in ASR output
- **ASR output parsing** (`parse_asr_output`): handles `<asr_text>` tag, empty audio, forced language, and repetition fix

### Build System

- CMake 3.14+, C++17
- GGML included as git submodule at `./ggml` (auto-initialized if missing)
- **GPU backends**:
  - macOS: Metal (Accelerate framework for vDSP mel spectrogram)
  - Linux: CUDA (auto-detected at `/usr/local/cuda`, fallback to CPU-only)
- **BLAS backends** (Linux): Intel MKL > OpenBLAS > generic BLAS (auto-detected)
- **Test builds**: controlled by `QWEN3_ASR_BUILD_TESTS` option (default OFF)
- Build: `cmake --build build -j$(nproc)` (Linux) or `cmake --build build -j$(sysctl -n hw.ncpu)` (macOS)

#### Build Options

```bash
# Standard build
cmake -B build && cmake --build build -j$(nproc)

# Enable tests
cmake -B build -DQWEN3_ASR_BUILD_TESTS=ON && cmake --build build -j$(nproc)

# Disable CUDA explicitly
cmake -B build -DGGML_CUDA=OFF && cmake --build build -j$(nproc)

# Enable timing instrumentation
cmake -B build -DQWEN3_ASR_TIMING=ON && cmake --build build -j$(nproc)
```

### HTTP Servers

Two HTTP servers are provided for production deployment:

#### Transcribe Server (ASR)

```bash
./build/qwen3-transcribe-server \
    --model models/qwen3-asr-0.6b-f16.gguf \
    --port 8081 \
    --threads 4 \
    --device CUDA0  # Optional: specify GPU device

# Endpoints:
#   GET  /health     - Health check
#   POST /transcribe - Upload audio (multipart: audio, language, context, max_tokens)
```

#### Align Server (Forced Alignment)

```bash
./build/qwen3-align-server \
    --model models/qwen3-forced-aligner-0.6b-f16.gguf \
    --port 8080 \
    --threads 4 \
    --korean-dict assets/korean_dict_jieba.dict \
    --device CUDA1  # Optional: specify different GPU

# Endpoints:
#   GET  /health - Health check
#   POST /align  - Upload audio + text (multipart: audio, text, language)
```

#### Multi-GPU Deployment

Run both servers on different GPUs:

```bash
# GPU 0 for transcription
./build/qwen3-transcribe-server --model models/asr.gguf --port 8081 --device CUDA0

# GPU 1 for alignment
./build/qwen3-align-server --model models/aligner.gguf --port 8080 --device CUDA1
```

#### API Usage Example

```bash
# Transcribe
curl -X POST http://localhost:8081/transcribe \
    -F "audio=@audio.raw" \
    -F "language=chinese"

# Align (using transcription result)
TEXT=$(curl -s http://localhost:8081/transcribe -F "audio=@audio.raw" | jq -r '.text_content')
curl -X POST http://localhost:8080/align \
    -F "audio=@audio.raw" \
    -F "text=$TEXT"
```

### Testing

- **Test audio**: any 16kHz mono WAV (convert with `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav`)
- **CLI ASR**: `./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav`
- **CLI Align**: `./build/qwen3-asr-cli -m models/qwen3-forced-aligner-0.6b-f16.gguf -f audio.wav --align --text "text" --lang korean`
- **CLI Combined**: `./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf -f audio.wav --transcribe-align`
- **HTTP API**: See HTTP Servers section above

### Conventions

- **Namespace**: `qwen3_asr::`
- **Error handling**: bool return + error_msg_ member
- **Timing**: `QWEN3_TIMER_SCOPED("name")` macros from `src/timing.h`
- **Memory**: RAII with explicit cleanup in destructors; mmap cleanup via munmap
- **Tensor naming**: follows HuggingFace naming convention for weight mapping

### Important Caveats

- The forced aligner decoder MUST use causal attention (model was trained with `self_attn.is_causal: True`)
- The forced aligner encoder uses windowed attention (block-diagonal mask, window_aftercnn=104)
- Korean word splitting requires `assets/korean_dict_jieba.dict` for optimal word-level alignment (without it, falls back to character-level)
- The ASR output text starts with "language <Name>" prefix (e.g. "language Korean...") which must be stripped before alignment
- Special token IDs: audio_start=151669, audio_end=151670, audio_pad=151676, timestamp=151705
- HTTP server audio format: raw PCM (16kHz, mono, 16-bit signed), not WAV (strip header or convert)
- Device names: CUDA0, CUDA1, ... for NVIDIA GPUs; Metal for Apple Silicon; CPU for fallback

## Performance Notes

Benchmark on 92-second Korean audio:
- **Apple M2 Pro**: Mel 98ms, Audio encode 715ms, Text decode 4,194ms (323 tokens), Total ASR 5,007ms, Alignment 12,998ms (183 words)
- **NVIDIA GPU (Linux)**: Similar or better performance depending on GPU model

Memory: ~247 MB RSS, ~294 MB GPU (varies by model size)

## Development Guidelines

### When Adding Features

1. **Maintain compatibility** with GGML submodule API
2. **Profile before optimizing** using `QWEN3_ASR_TIMING` build flag
3. **Follow existing patterns** for error handling and memory management
4. **Test on real audio** (not just synthetic data)
5. **Update benchmarks** if modifying hot paths

### Common Pitfalls

- Don't break mmap compatibility (weight tensors must use mmap buffer)
- Don't change tensor layout without updating GGUF conversion script
- Flash attention requires contiguous memory layout
- Korean dictionary must be UTF-8 encoded
- Audio must be 16kHz mono PCM (conversion required otherwise)

### File Organization

- Headers (`.h`) contain public API and documentation
- Implementation (`.cpp`) contains internal logic
- Test files (`test_*.cpp`) demonstrate usage and verify correctness
- Timing macros only active when `QWEN3_ASR_TIMING` defined

## Contributing

When making changes:
1. Run existing tests to verify no regression
2. Add new tests for new functionality
3. Update AGENTS.md if architecture changes
4. Document any new conventions or caveats
5. Benchmark performance impact on Apple Silicon

## Resources

- [GGML Documentation](https://github.com/ggerganov/ggml)
- [Qwen3-ASR HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- Korean dictionary: 17,968 words from soynlp corpus
