# Qwen3-ASR CLI Usage Guide

Complete documentation for the `qwen3-asr-cli` command-line interface.

## Synopsis

```
qwen3-asr-cli [options]
```

## Options

### Required Options

| Option | Description |
|--------|-------------|
| `-f, --audio <path>` | Path to input audio file (WAV, 16kHz mono) |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model <path>` | `models/qwen3-asr-0.6b-f16.gguf` | Path to GGUF model file |
| `--aligner-model <path>` | - | Path to forced aligner model (required for `--transcribe-align`) |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <path>` | stdout | Output file path |
| `--json` | off | Output in JSON format |

### Transcription Options

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --lang <lang>` | auto-detect | Language code (e.g., `english`, `chinese`, `korean`) |
| `-t, --threads <n>` | 4 | Number of CPU threads |
| `--max-tokens <n>` | 1024 | Maximum tokens to generate |
| `--context <text>` | - | Context/prompt text to inject before transcription |
| `--arabic-numbers` | off | Convert Chinese numbers to Arabic numerals |
| `--progress` | off | Print progress during transcription |
| `--no-timing` | off | Suppress timing information |

### Forced Alignment Options

| Option | Description |
|--------|-------------|
| `--align` | Enable forced alignment mode |
| `--text <text>` | Reference transcript for alignment (required with `--align`) |
| `--transcribe-align` | Combined mode: transcribe then align |

### Debug Options

| Option | Description |
|--------|-------------|
| `--debug-input` | Print input tokens to stderr |
| `--debug-output` | Print output tokens to stderr |

### Help

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |

## Transcription Mode

### Basic Usage

```bash
# Transcribe audio file
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav
```

### Output Formats

**Plain text (default):**
```
language English Hello world, this is a test transcription.
```

**JSON format (`--json`):**
```json
{
  "text": "language English Hello world, this is a test.",
  "text_prefix": "language English",
  "text_content": "Hello world, this is a test.",
  "tokens": [
    {"id": 151704, "string": "language", "confidence": 0.9947},
    {"id": 32313, "string": "Hello", "confidence": 0.9973}
  ]
}
```

### Examples

```bash
# Specify language for better accuracy
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --lang english

# Use context/prompt
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --context "This is a lecture about mathematics."

# JSON output
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --json

# Convert Chinese numbers to Arabic
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f chinese_audio.wav \
    --arabic-numbers

# Multi-threaded processing with progress
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    -t 8 \
    --progress

# Save to file
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    -o transcript.txt

# Debug mode
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --debug-input \
    --debug-output
```

## Forced Alignment Mode

Forced alignment synchronizes a reference transcript with audio, producing utterance-level timestamps with word-level details.

### Basic Usage

```bash
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world. This is a test." \
    --lang english
```

### Output Format (Utterance-level JSON)

```json
{
  "utterances": [
    {
      "start": 0.000,
      "end": 1.360,
      "text": "Hello world.",
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
      "words": [
        {"word": "This", "start": 1.360, "end": 1.520, "conf_word": 0.98},
        {"word": "is", "start": 1.540, "end": 1.620, "conf_word": 0.99},
        {"word": "a", "start": 1.640, "end": 1.680, "conf_word": 0.99},
        {"word": "test", "start": 1.700, "end": 2.000, "conf_word": 0.97},
        {"word": ".", "start": 2.000, "end": 2.040}
      ]
    }
  ]
}
```

Utterances are split by sentence-ending punctuation: `.`, `!`, `?`, `。`, `！`, `？`

### Examples

```bash
# Basic alignment with language
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world" \
    --lang english

# Korean alignment (requires dictionary)
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f korean_audio.wav \
    --align \
    --text "안녕하세요" \
    --lang korean

# Save alignment to file
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world" \
    -o alignment.json

# JSON output with progress
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Test sentence" \
    --json \
    --progress
```

## Combined Pipeline Mode

The `--transcribe-align` mode automatically runs transcription then alignment, ideal for processing audio with unknown content.

### Basic Usage

```bash
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --transcribe-align
```

This mode:
1. Runs ASR to transcribe the audio
2. Detects language from ASR output
3. Runs forced alignment with the transcript
4. Outputs alignment JSON

### Examples

```bash
# Combined mode with progress
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --transcribe-align \
    --progress

# Force language override
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --transcribe-align \
    --lang english

# Save output to file
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --transcribe-align \
    -o result.json
```

## Audio Requirements

### Supported Format

- **Format**: WAV (PCM)
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit signed integer

### Converting Audio

Use ffmpeg to convert audio to the required format:

```bash
# Convert MP3 to WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# Convert M4A to WAV
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# Convert stereo to mono
ffmpeg -i stereo.wav -ar 16000 -ac 1 -c:a pcm_s16le mono.wav

# Extract audio from video
ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
```

## Performance Tips

### Thread Count

Optimal thread count depends on your CPU:

```bash
# Use all available cores (Linux)
./build/qwen3-asr-cli -m model.gguf -f audio.wav -t $(nproc)

# Use all available cores (macOS)
./build/qwen3-asr-cli -m model.gguf -f audio.wav -t $(sysctl -n hw.ncpu)
```

### Quantized Models

Q8_0 quantized models offer:
- ~40% smaller file size
- Faster inference on CPU
- Minimal quality loss

```bash
# Use quantized model
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav
```

### Memory Usage

| Model | Memory (approx) |
|-------|-----------------|
| ASR F16 | ~2.5 GB |
| ASR Q8_0 | ~1.8 GB |
| Aligner F16 | ~1.8 GB |

### Batch Processing

For multiple files, use a shell loop:

```bash
for f in *.wav; do
    ./build/qwen3-asr-cli \
        -m models/qwen3-asr-0.6b-q8_0.gguf \
        -f "$f" \
        -o "${f%.wav}.txt" \
        --no-timing
done
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (model loading, audio loading, processing failure) |

## Timing Output

When timing is enabled (default), the following metrics are printed to stderr:

**Transcription mode:**
```
Timing:
  Mel spectrogram: 45 ms
  Audio encoding:  120 ms
  Text decoding:   850 ms
  Total:           1015 ms
```

**Alignment mode:**
```
Timing:
  Mel spectrogram: 45 ms
  Audio encoding:  120 ms
  Decoder:         350 ms
  Total:           515 ms
  Utterances aligned: 2
```

**Combined mode:**
```
Phase 1: Transcription
  Timing: ...
Phase 2: Forced Alignment
  Timing: ...
Combined Timing:
  ASR:           5007 ms
  Alignment:     12998 ms
  Total:         18005 ms
```

## Troubleshooting

### Common Errors

**"Error: Audio file path is required"**
```bash
# Solution: Provide audio file with -f
./build/qwen3-asr-cli -m model.gguf -f audio.wav
```

**"Error: Reference text is required for alignment mode"**
```bash
# Solution: Provide text with --text when using --align
./build/qwen3-asr-cli -m model.gguf -f audio.wav --align --text "Your text"
```

**"Error: Aligner model is required for --transcribe-align"**
```bash
# Solution: Provide aligner model
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --transcribe-align
```

**"Error: Failed to load model"**
```bash
# Check model path exists
ls -la models/qwen3-asr-0.6b-f16.gguf

# Ensure model is valid GGUF format
file models/qwen3-asr-0.6b-f16.gguf
```

**"Error: Could not load audio file"**
```bash
# Check audio format
ffprobe audio.wav

# Convert to correct format
ffmpeg -i audio.wav -ar 16000 -ac 1 -c:a pcm_s16le audio_fixed.wav
```

### Debug Mode

Enable debug output for troubleshooting:

```bash
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --debug-input \
    --debug-output
```

This prints input/output tokens which can help diagnose issues.