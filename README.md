# Qwen3-ASR.cpp

A C++ GGML implementation of Qwen3-ASR and Qwen3-ForcedAligner with GPU acceleration (CUDA/Metal). Supports F16, Q8_0, and NVFP4 quantized models for speech recognition and word-level timestamp alignment.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio to text in 30+ languages
- **Forced Alignment**: Align text to audio with word-level timestamps
- **Combined Pipeline** (`--transcribe-align`): ASR → alignment with auto language detection
- **Quantized Model Support**: F16, Q8_0, NVFP4 (Blackwell), Q4_0 — up to 53% memory reduction
- **im2col+mul_mat Conv**: Quantized conv kernels compute directly via mul_mat (no dequant overhead)
- **Flash Attention**: `ggml_flash_attn_ext()` with F32 precision for fast decoding
- **F16 KV Cache**: Half-precision key-value cache
- **GPU Acceleration**: CUDA (Linux) / Metal (macOS) with multi-backend scheduling
- **mmap Weight Loading**: Zero-copy GPU transfer, fast initialization
- **Korean/CJK Word Splitting**: Soynlp LTokenizer + CJK character-level splitting
- **HF Tokenizer JSON Fallback**: Parses `tokenizer.huggingface.json` when GGUF tokens absent
- **GGUF Quantization Tool**: Built-in `ggml-quantize` for F16 → NVFP4/Q8_0/Q4_0 conversion
- **Pure C++17**: No Python runtime required

## Supported Models

| Model | F16 Size | NVFP4 Size | Q8_0 Size | Description |
|-------|----------|------------|-----------|-------------|
| `qwen3-asr-1.7b-f16.gguf` | ~4.48 GB | ~2.30 GB | ~2.4 GB | ASR model |
| `qwen3-forced-aligner-0.6b-f16.gguf` | ~1.75 GB | ~0.81 GB | — | Forced aligner |

## Requirements

- CMake 3.14+, C++17 compiler
- NVIDIA GPU with CUDA (Linux) or Apple Silicon with Metal (macOS)
- GGML library (included as submodule)

## Building

```bash
git clone --recursive https://github.com/yexiangyu/qwen3-asr.cpp.git
cd qwen3-asr.cpp

# Standard build (with CUDA)
cmake -B build && cmake --build build -j$(nproc)

# Disable CUDA
cmake -B build -DGGML_CUDA=OFF && cmake --build build -j$(nproc)

# macOS
cmake -B build && cmake --build build -j$(sysctl -n hw.ncpu)
```

## Quick Start

### Transcription (ASR only)

```bash
./build/asr-cli --input audio.wav --model models/qwen3-asr-1.7b-f16.gguf --transcribe-only
```

### Forced Alignment

```bash
./build/asr-cli --input audio.wav \
  --model models/qwen3-forced-aligner-0.6b-f16.gguf \
  --align-only --text "Hello world" --language english
```

### Combined (Transcribe + Align)

```bash
./build/asr-cli --input audio.wav \
  --model models/qwen3-asr-1.7b-f16.gguf \
  --aligner models/qwen3-forced-aligner-0.6b-f16.gguf
```

### NVFP4 Quantized Models

Same commands, just use the quantized model paths:

```bash
./build/asr-cli --input audio.wav \
  --model models/qwen3-asr-1.7b-nvfp4.gguf \
  --aligner models/qwen3-forced-aligner-0.6b-nvfp4.gguf
```

### CLI Options

```
--input <path>           Input audio file (WAV/ffmpeg-supported)
--model <path>           ASR model path
--aligner <path>         Aligner model path
--device <name>          Compute device: CUDA0, CUDA1, Metal, CPU (default: CUDA0)
--threads <n>            Thread count (default: 4)
--language <lang>        Language hint (e.g., chinese, korean, english)
--transcribe-only        Only transcription, skip alignment
--align-only             Only alignment (requires --text)
--text <text>            Text for alignment
--max-tokens <n>         Max tokens to generate (default: 512)
--output <path>          Output file path
--format <fmt>           Output format: text, json (default: text)
```

## Quantization

```bash
# F16 → NVFP4 (Blackwell RTX 5080+)
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-nvfp4.gguf nvfp4

# F16 → Q8_0 (universal, all GPUs)
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-q8_0.gguf q8_0

# F16 → Q4_0
./build/ggml-quantize models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-q4_0.gguf q4_0
```

### Quantization Rules

- **Quantized**: all weight matrices with ne[0] divisible by block_size
- **Kept as F16**: norm/bias, token embeddings, lm_head/output.weight, classify_head, ln_post, conv weights (small dims), **attn_output.weight** (critical — NVFP4/Q4_0 causes repetitive generation)
- **NVFP4**: additionally requires ne[0] % 64 == 0

## Model Conversion (HF → GGUF)

```bash
pip install -r scripts/requirements.txt

# Convert ASR model
python scripts/convert_hf_to_gguf.py \
  --input /path/to/Qwen3-ASR-1.7B \
  --output models/qwen3-asr-1.7b-f16.gguf \
  --type f16

# Convert aligner model
python scripts/convert_hf_to_gguf.py \
  --input /path/to/Qwen3-ForcedAligner-0.6B \
  --output models/qwen3-forced-aligner-0.6b-f16.gguf \
  --type f16
```

## Performance

### NVIDIA RTX 5080 (Blackwell), 120-second Chinese audio

| Stage | F16 | NVFP4 | Ratio |
|-------|-----|-------|-------|
| ASR Init | 5950 ms | 1125 ms | 0.19x |
| Audio encode | 352 ms | 313 ms | 0.89x |
| Text decode (190 tokens) | 2028 ms | 1910 ms | 0.94x |
| Aligner Init | 2342 ms | 327 ms | 0.14x |
| Aligner encode | 176 ms | 139 ms | 0.79x |
| Aligner decode (237 words) | 6059 ms | 5937 ms | 0.98x |

**Memory**: ASR 4.48GB → 2.30GB (48.7%), Aligner 1.75GB → 0.81GB (53.7%)

### Key Optimizations

- **im2col+mul_mat Conv**: Quantized conv kernels compute via mul_mat directly — no dequant overhead
- **Flash Attention** (F32 precision): Fast autoregressive decode
- **F16 KV Cache**: Half-precision for reduced memory bandwidth
- **mmap + Zero-Copy**: Fast model loading, GPU transfer without CPU copy
- **GGUF key name fallback**: Supports both GGML (`blk.*`) and HF (`model.layers.*`) naming
- **HF tokenizer JSON fallback**: Parses `tokenizer.huggingface.json` for vocab/merges when GGUF tokens absent

## Inference Pipeline

### ASR (Transcribe)

1. **Audio load**: WAV/ffmpeg → float samples, 16kHz mono (`codec::decode_file`)
2. **Mel spectrogram**: FFT → power spectrum → mel filter bank → log (`mel::compute`, n_fft=400, hop=160, n_mels=128)
3. **Audio encode**: Conv frontend (3× Conv2d stride=2 via im2col+mul_mat) → 18-layer transformer → hidden=1024 (`transcribe::encoder::encode_batch`)
4. **Text decode**: Chat template with audio_pad tokens → inject audio embeddings → prefill + autoregressive decode (Qwen2, 28 layers, GQA 32/8 heads, flash attention, F16 KV cache) → argmax → text (`transcribe::decoder::transcribe`)

### Forced Alignment

1. **Audio encode** (same mel + aligner encoder): 3× Conv2d → 24-layer transformer with **windowed attention** (window_aftercnn=104) → hidden=1024
2. **Tokenize text**: CJK per-char / Korean LTokenizer / Latin per-word → each word + 2 `<|timestamp|>` tokens
3. **Aligner decode**: Single forward pass (non-autoregressive, **causal attention**, classify_head 5000 classes) → argmax per timestamp token → class × 80ms = timestamp
4. **Timestamp fix**: LIS-based monotonic correction → word start/end timestamps

### Special Token IDs

| Token | ID |
|-------|----|
| `<|audio_start|>` | 151669 |
| `<|audio_end|>` | 151670 |
| `<|AUDIO|>` (audio pad) | 151676 |
| `<|timestamp|>` | 151705 |
| `<|im_start|>` | 151644 |
| `<|im_end|>` (EOS) | 151645 |

## Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese | chinese | English | english |
| Japanese | japanese | Korean | korean |
| German | german | French | french |
| Spanish | spanish | Italian | italian |
| Portuguese | portuguese | Russian | russian |
| Arabic | arabic | Hindi | hindi |
| Thai | thai | Vietnamese | vietnamese |

30+ languages total.

## Audio Requirements

- **Format**: WAV (PCM) or ffmpeg-supported formats
- **Sample rate**: 16 kHz
- **Channels**: Mono
- **Bit depth**: 16-bit signed integer

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Project Structure

```
qwen3-asr.cpp/
├── src/
│   ├── tools/
│   │   ├── asr_cli.cpp              # CLI entry point
│   │   └── quantize.cpp             # GGUF quantization tool
│   ├── asr/
│   │   ├── transcribe/
│   │   │   ├── encoder.cpp/h        # ASR audio encoder (im2col+mul_mat conv, 18-layer transformer)
│   │   │   ├── encoder_model.hpp    # Encoder hyperparams + model struct
│   │   │   ├── decoder.cpp/h        # ASR text decoder (Qwen2, flash attention, KV cache)
│   │   │   └── decoder_model.hpp    # Decoder hyperparams + model struct
│   │   ├── aligner/
│   │   │   ├── encoder.cpp/h        # Aligner encoder (im2col+mul_mat conv, 24-layer windowed attn)
│   │   │   ├── encoder_model.hpp
│   │   │   ├── decoder.cpp/h        # Aligner decoder (classify_head, causal attn, timestamps)
│   │   │   └── decoder_model.hpp
│   │   ├── mel/
│   │   │   ├── mel.cpp/h            # Mel spectrogram computation
│   │   ├── codec/
│   │   │   ├── codec.cpp/h          # Audio loading (WAV + ffmpeg fallback)
│   │   │   ├── wav_loader.cpp/h     # WAV parsing + normalization
│   │   ├── common/
│   │   │   ├── types.hpp            # MelSpectrum, AudioFeatures, AlignedWord, ErrorInfo
│   │   │   ├── hf_tokenizer.hpp     # HF JSON tokenizer fallback (vocab/merges parsing)
│   ├── logger.cpp/h                 # spdlog-based logging
│   ├── timing.h                     # Timing instrumentation macros
├── scripts/
│   ├── convert_hf_to_gguf.py        # HF → GGUF model conversion
│   └── requirements.txt
├── assets/
│   └── korean_dict_jieba.dict       # Korean word dictionary (17,968 words)
├── models/                          # Model files (.gguf)
├── tests/                           # Test programs
├── docs/                            # Documentation
└── ggml/                            # GGML library (submodule)
```

## Important Caveats

- **Don't use `ggml_conv_2d` directly** — always use `conv_2d_via_im2col` for quantized model support
- **Don't quantize `attn_output.weight`** to NVFP4/Q4_0 — causes repetitive/infinite generation
- **Aligner decoder must use causal attention** (model trained with `is_causal=True`)
- **Aligner encoder uses windowed attention** (block-diagonal mask, window=104 frames)
- **NVFP4 block_size=64**: tensors with ne[0] not divisible by 64 stay F16
- **im2col output must be F16** (CUDA doesn't support Q8_0 im2col output)
- **GGUF key names** vary: F16 uses GGML convention (`blk.*`), Q8_0 uses HF convention (`model.layers.*`)
- **ASR output** starts with "language <Name>" prefix — must strip before alignment
- **Audio must be 16kHz mono PCM/WAV**

## License

MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) — Tensor library
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — Original ASR model
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) — Aligner model