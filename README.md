# Qwen3-ASR.cpp

A high-performance C++ implementation of Qwen3-ASR and Qwen3-ForcedAligner using the GGML tensor library. Optimized for Apple Silicon with Metal GPU acceleration, providing fast speech recognition and word-level timestamp alignment.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio files to text in 30+ languages
- **Forced Alignment**: Align reference text to audio with utterance-level timestamps
- **Combined Pipeline** (`--transcribe-align`): Automatically runs ASR then alignment with auto language detection
- **HTTP Server Services**: Production-ready HTTP servers for transcription and alignment
- **C API Library**: C-compatible API for multi-language integration (Python, Go, Rust, etc.)
- **Flash Attention**: Uses `ggml_flash_attn_ext()` for fast decoding (3.7x speedup)
- **Metal GPU Acceleration**: Optimized for Apple Silicon with dual CPU+Metal backend
- **Accelerate/vDSP**: Highly optimized mel spectrogram computation (45x speedup)
- **mmap Weight Loading**: Zero-copy GPU transfer for fast model initialization
- **F16 KV Cache**: Reduced memory bandwidth with half-precision key-value cache
- **Korean Word Splitting**: Soynlp LTokenizer algorithm with 18K-word dictionary
- **Quantization Support**: Q8_0 quantization for reduced memory usage (~40% smaller)
- **Pure C++17**: No Python runtime required for inference

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b-f16.gguf` | ~1.8 GB | ASR model, F16 precision |
| `qwen3-asr-0.6b-q8_0.gguf` | ~1.3 GB | ASR model, Q8_0 quantized |
| `qwen3-forced-aligner-0.6b-f16.gguf` | ~1.8 GB | Forced alignment model |

## Requirements

- CMake 3.14+
- C++17 compatible compiler (Clang 7+, GCC 8+, MSVC 2019+)
- Apple Silicon recommended (Metal GPU support)
- GGML library (included as submodule)

## Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yexiangyu/qwen3-asr.cpp.git
cd qwen3-asr.cpp

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

On Linux, replace `$(sysctl -n hw.ncpu)` with `$(nproc)`.

## Quick Start

### CLI Tools

**Transcription (ASR):**
```bash
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav
```

**Forced Alignment:**
```bash
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "Hello world" \
  --lang english
```

**Combined Pipeline (Transcribe + Align):**
```bash
./build/qwen3-asr-cli \
  -m models/qwen3-asr-0.6b-f16.gguf \
  --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --transcribe-align
```

### HTTP Servers

**Transcription Server:**
```bash
./build/qwen3-transcribe-server \
  --model models/qwen3-asr-0.6b-f16.gguf \
  --port 8081
```

**Alignment Server:**
```bash
./build/qwen3-align-server \
  --model models/qwen3-forced-aligner-0.6b-f16.gguf \
  --port 8080
```

## HTTP Server Services

Two independent HTTP servers for production deployment.

### qwen3-transcribe-server (Transcription Service)

Transcription HTTP server with REST API.

**Startup:**
```bash
./build/qwen3-transcribe-server \
  --model models/qwen3-asr-0.6b-f16.gguf \
  --port 8081 \
  --threads 4 \
  --max-tokens 1024
```

**Command-line Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | - | ASR model path (required) |
| `--port <num>` | 8081 | HTTP port |
| `--threads <num>` | 4 | Processing threads |
| `--max-tokens <num>` | 1024 | Maximum output tokens |
| `--default-language <lang>` | - | Default language |

**API Endpoints:**

**GET /health** - Health check
```json
{"status": "ok", "model_loaded": true}
```

**POST /transcribe** - Transcription request (multipart/form-data)

Request fields:
| Field | Required | Description |
|-------|----------|-------------|
| `audio` | Yes | PCM int16 data (16kHz mono) |
| `language` | No | Language code (e.g., `english`) |
| `context` | No | Context/prompt text |
| `max_tokens` | No | Maximum tokens override |

Example:
```bash
# Extract PCM from WAV (skip 44-byte header)
dd if=audio.wav bs=1 skip=44 of=audio.pcm

# Send request
curl -X POST http://localhost:8081/transcribe \
  -F "audio=@audio.pcm;type=application/octet-stream" \
  -F "language=english"
```

Response:
```json
{
  "success": true,
  "text": "language English Hello world, this is a test.",
  "text_content": "Hello world, this is a test.",
  "processing_time_ms": 8045,
  "n_tokens": 297,
  "tokens": [
    {"id": 151704, "confidence": 0.9947},
    {"id": 32313, "confidence": 0.9973}
  ]
}
```

Error response:
```json
{
  "success": false,
  "error": "Error description"
}
```

**Python Client:**
```python
import requests

# Read PCM data
pcm_data = open('audio.pcm', 'rb').read()

# Send request
response = requests.post(
    'http://localhost:8081/transcribe',
    files={'audio': ('audio.pcm', pcm_data, 'application/octet-stream')},
    data={
        'language': 'english',
        'context': 'Optional context prompt'
    }
)

result = response.json()
if result['success']:
    print(result['text_content'])
else:
    print(f"Error: {result['error']}")
```

**Concurrency & Performance:**
- Crow framework supports multithreaded request reception
- Single model instance with mutex protection for thread safety
- Requests processed serially to avoid memory contention
- PCM direct transmission avoids WAV parsing overhead
- Model pre-loaded at startup reduces first-request latency

### qwen3-align-server (Alignment Service)

Alignment HTTP server with utterance-level output.

**Startup:**
```bash
./build/qwen3-align-server \
  --model models/qwen3-forced-aligner-0.6b-f16.gguf \
  --port 8080 \
  --threads 4
```

**Command-line Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | - | Aligner model path (required) |
| `--port <num>` | 8080 | HTTP port |
| `--threads <num>` | 4 | Processing threads |
| `--korean-dict <path>` | - | Korean dictionary path |
| `--default-language <lang>` | - | Default language |

**API Endpoints:**

**GET /health** - Health check
```json
{"status": "ok", "model_loaded": true}
```

**POST /align** - Alignment request (multipart/form-data)

Request fields:
| Field | Required | Description |
|-------|----------|-------------|
| `audio` | Yes | PCM int16 data (16kHz mono) |
| `text` | Yes | Reference text to align |
| `language` | No | Language code (e.g., `english`) |

Example:
```bash
curl -X POST http://localhost:8080/align \
  -F "text=Hello world. This is a test." \
  -F "audio=@audio.pcm;type=application/octet-stream" \
  -F "language=english"
```

Response (utterance-level, sentences split by `.!?。！？`):
```json
{
  "success": true,
  "processing_time_ms": 6216,
  "n_utterances": 2,
  "utterances": [
    {
      "start": 0.000,
      "end": 1.360,
      "text": "Hello world.",
      "n_words": 4,
      "words": [
        {"word": "Hello", "start": 0.000, "end": 0.320, "conf_word": 0.95},
        {"word": "world", "start": 0.340, "end": 0.640, "conf_word": 0.92},
        {"word": ".", "start": 0.640, "end": 0.680}
      ]
    },
    {
      "start": 1.360,
      "end": 2.500,
      "text": "This is a test.",
      "n_words": 5,
      "words": [...]
    }
  ]
}
```

**Python Client:**
```python
import requests

pcm_data = open('audio.pcm', 'rb').read()
text = "Hello world. This is a test."

response = requests.post(
    'http://localhost:8080/align',
    files={'audio': ('audio.pcm', pcm_data, 'application/octet-stream')},
    data={'text': text, 'language': 'english'}
)

result = response.json()
for utt in result['utterances']:
    print(f"{utt['start']:.3f} -> {utt['end']:.3f}: {utt['text']}")
```

### Audio Format for HTTP Servers

HTTP servers accept raw PCM data, not WAV files:

**PCM format:**
- Format: int16 (2 bytes per sample)
- Sample rate: 16kHz mono
- No header (pure PCM bytes)

**Extract PCM from WAV:**
```bash
# WAV has 44-byte header, skip it
dd if=audio.wav bs=1 skip=44 of=audio.pcm
```

**Convert audio to PCM:**
```bash
# Convert any format to PCM
ffmpeg -i input.mp3 -f s16le -acodec pcm_s16le -ar 16000 -ac 1 audio.pcm
```

## C API Library

`libqwen3asr` provides a C-compatible API for integration with Python, Go, Rust, and other languages.

**Basic Usage:**
```c
#include "qwen3asr_c_api.h"

// Initialize handle
qwen3asr_handle handle;
qwen3asr_init(&handle);
qwen3asr_load_model(handle, "model.gguf");

// Transcribe
qwen3asr_params params = {
    .max_tokens = 1024,
    .language = "english",
    .n_threads = 4
};
qwen3asr_result result;
qwen3asr_transcribe_pcm(handle, pcm_data, n_samples, &params, &result);

// Use result
printf("Text: %s\n", result.text);
printf("Tokens: %d\n", result.n_tokens);

// Cleanup
qwen3asr_free_result(&result);
qwen3asr_free(handle);
```

**Device Enumeration:**
```c
int n_devices = qwen3_get_device_count();
for (int i = 0; i < n_devices; i++) {
    qwen3_device_info info;
    qwen3_get_device_info(i, &info);
    printf("Device %d: %s\n", i, info.name);
    qwen3_free_device_info(&info);
}
```

**Alignment API:**
```c
qwen3aligner_handle aligner;
qwen3aligner_init(&aligner);
qwen3aligner_load_model(aligner, "aligner.gguf");

qwen3alignment_result align_result;
qwen3aligner_params align_params = {.language = "english"};
qwen3aligner_align_pcm(aligner, pcm_data, n_samples, "text", &align_params, &align_result);

// Process utterances
for (int i = 0; i < align_result.n_utterances; i++) {
    // ...
}

qwen3aligner_free_result(&align_result);
qwen3aligner_free(aligner);
```

See header file: `src/qwen3asr_c_api.h`

## Output Formats

### ASR Output

**Plain text (default):**
```
language English Hello world, this is a test.
```

**JSON format (`--json`):**
```json
{
  "text": "language English Hello world, this is a test.",
  "text_prefix": "language English",
  "text_content": "Hello world, this is a test.",
  "tokens": [{"id": ..., "string": "...", "confidence": ...}]
}
```

### Alignment Output

Utterance-level JSON with word-level timestamps:
```json
{
  "utterances": [
    {
      "start": 0.000,
      "end": 1.360,
      "text": "Hello world.",
      "words": [
        {"word": "Hello", "start": 0.000, "end": 0.320, "conf_word": 0.95},
        {"word": "world", "start": 0.340, "end": 0.640, "conf_word": 0.92}
      ]
    }
  ]
}
```

Sentences are split by: `.`, `!`, `?`, `。`, `！`, `？`

## Performance

Benchmark on 92-second Korean audio, Apple M2 Pro:

| Stage | Time |
|-------|------|
| Mel spectrogram | 98 ms |
| Audio encoding | 715 ms |
| Text decoding (323 tokens) | 4,194 ms |
| **ASR Total** | **5,007 ms** |
| Forced alignment (183 words) | 12,998 ms |
| **Combined Total** | **18,005 ms** |

**Memory Usage:** ~247 MB RSS, ~294 MB Metal

### Key Optimizations

- **Flash Attention**: 3.7x decode speedup
- **Metal GPU Dual Backend**: Automatic CPU/GPU scheduling
- **mmap + Zero-Copy**: Fast model loading
- **F16 KV Cache**: Half-precision cache
- **vDSP Mel Spectrogram**: 45x speedup on Apple Silicon
- **Korean Word Splitting**: LTokenizer with 18K-word dictionary

## Model Conversion

Convert HuggingFace models to GGUF:

```bash
pip install -r scripts/requirements.txt

# Convert ASR model
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-f16.gguf \
    --type f16

# Convert Aligner model
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ForcedAligner-0.6B \
    --output models/qwen3-forced-aligner-0.6b-f16.gguf \
    --type f16
```

## Supported Languages

30+ languages supported:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese | chinese | English | english |
| Japanese | japanese | Korean | korean |
| German | german | French | french |
| Spanish | spanish | Italian | italian |
| Portuguese | portuguese | Russian | russian |
| Arabic | arabic | Hindi | hindi |
| Thai | thai | Vietnamese | vietnamese |

## Audio Requirements

- **Format**: WAV (PCM) or raw PCM
- **Sample rate**: 16 kHz
- **Channels**: Mono
- **Bit depth**: 16-bit signed integer

Convert with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Documentation

- [CLI Usage Guide](docs/usage.md) - Complete CLI documentation
- [Mel Preprocessing](docs/mel_preprocessing.md) - Mel spectrogram details

## Project Structure

```
qwen3-asr.cpp/
├── src/
│   ├── main.cpp                 # CLI entry point
│   ├── qwen3_asr.cpp/h          # High-level ASR API
│   ├── forced_aligner.cpp/h     # Forced alignment
│   ├── audio_encoder.cpp/h      # Audio feature encoder
│   ├── text_decoder.cpp/h       # Text decoder
│   ├── mel_spectrogram.cpp/h    # Mel spectrogram
│   ├── audio_utils.cpp/h        # Audio utilities
│   ├── http_server/             # HTTP servers
│   │   ├── transcribe_server.*  # Transcription service
│   │   ├── align_server.*       # Alignment service
│   ├── qwen3asr_c_api.*         # C API
│   └── logger.*                 # Logging (spdlog)
├── tests/                       # Test programs
├── scripts/                     # Conversion scripts
├── assets/                      # Korean dictionary
├── models/                      # Model files
├── docs/                        # Documentation
└── ggml/                        # GGML library
```

## License

MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) - Tensor library
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Original model
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) - Aligner model

---