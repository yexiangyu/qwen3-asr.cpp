#include "asr/codec/codec.hpp"
#include "asr/mel/mel.hpp"
#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/decoder.hpp"
#include "asr/aligner/encoder.hpp"
#include "asr/aligner/decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace asr;

struct CliConfig {
    std::string input_path;
    std::string model_path = "models/qwen3-asr-1.7b-f16.gguf";
    std::string aligner_model_path = "models/qwen3-forced-aligner-0.6b-f16.gguf";
    std::string output_path;
    std::string language;
    std::string context;
    std::string hotwords;
    std::string text_for_align;
    std::string device = "CUDA0";
    int threads = 4;
    int max_tokens = 512;
    bool transcribe_only = false;
    bool align_only = false;
    bool json_output = false;
};

void print_usage(const char* prog_name) {
    printf("Usage: %s [options] --input <media_file>\n\n", prog_name);
    printf("Options:\n");
    printf("  --input <path>           Input media file (audio/video)\n");
    printf("  --model <path>           ASR model (default: %s)\n", "models/qwen3-asr-1.7b-f16.gguf");
    printf("  --aligner <path>         Aligner model (default: %s)\n", "models/qwen3-forced-aligner-0.6b-f16.gguf");
    printf("  --device <name>          Device (default: CUDA0)\n");
    printf("  --threads <n>            Number of threads (default: 4)\n");
    printf("  --language <lang>        Language hint (e.g., chinese, korean)\n");
    printf("  --context <text>         Previous transcription for streaming\n");
    printf("  --hotwords <words>       Words to emphasize (comma-separated)\n");
    printf("  --max-tokens <n>         Max tokens to generate (default: 512)\n");
    printf("  --output <path>          Output file path\n");
    printf("  --format <fmt>           Output format: text, json (default: text)\n");
    printf("  --transcribe-only        Only transcription, skip alignment\n");
    printf("  --align-only             Only alignment (requires --text)\n");
    printf("  --text <text>            Text for alignment (when --align-only)\n");
    printf("  --help                   Show this help message\n");
}

bool parse_args(int argc, char** argv, CliConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        
        if (arg == "--input" && i + 1 < argc) {
            config.input_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--aligner" && i + 1 < argc) {
            config.aligner_model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::atoi(argv[++i]);
        } else if (arg == "--language" && i + 1 < argc) {
            config.language = argv[++i];
        } else if (arg == "--context" && i + 1 < argc) {
            config.context = argv[++i];
        } else if (arg == "--hotwords" && i + 1 < argc) {
            config.hotwords = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            std::string fmt = argv[++i];
            config.json_output = (fmt == "json");
        } else if (arg == "--transcribe-only") {
            config.transcribe_only = true;
        } else if (arg == "--align-only") {
            config.align_only = true;
        } else if (arg == "--text" && i + 1 < argc) {
            config.text_for_align = argv[++i];
        }
    }
    
    if (config.input_path.empty()) {
        fprintf(stderr, "Error: --input is required\n");
        print_usage(argv[0]);
        return false;
    }
    
    if (config.align_only && config.text_for_align.empty()) {
        fprintf(stderr, "Error: --align-only requires --text\n");
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    CliConfig config;
    
    if (!parse_args(argc, argv, config)) {
        return 1;
    }
    
    printf("=== ASR CLI ===\n\n");
    printf("Input: %s\n", config.input_path.c_str());
    printf("Model: %s\n", config.model_path.c_str());
    printf("Aligner: %s\n", config.aligner_model_path.c_str());
    printf("Device: %s\n", config.device.c_str());
    printf("Threads: %d\n", config.threads);
    
    asr::ErrorInfo error;
    
    printf("\n--- Step 1: Load audio ---\n");
    
    std::vector<float> audio_samples;
    int sample_rate;
    
    if (!codec::decode_file(config.input_path.c_str(), audio_samples, sample_rate, &error)) {
        fprintf(stderr, "Error: Failed to load audio: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Audio: %zu samples, %d Hz, %.2f seconds\n", 
           audio_samples.size(), sample_rate, 
           (float)audio_samples.size() / sample_rate);
    
    printf("\n--- Step 2: Compute mel spectrogram ---\n");
    
    mel::Config mel_config;
    mel_config.n_threads = config.threads;
    
    mel::MelSpectrum mel_spec;
    
    if (!mel::compute_from_file(config.input_path.c_str(), mel_spec, mel_config, &error)) {
        fprintf(stderr, "Error: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Mel: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    
    printf("\n--- Step 3: Initialize modules ---\n");
    
    // Initialize transcribe encoder
    transcribe::encoder::Config enc_config;
    enc_config.model_path = config.model_path;
    enc_config.device_name = config.device;
    enc_config.n_threads = config.threads;
    
    auto enc_state = transcribe::encoder::init(enc_config);
    if (!enc_state) {
        fprintf(stderr, "Error: Failed to init transcribe encoder\n");
        return 1;
    }
    
    printf("Transcribe encoder initialized on %s\n", transcribe::encoder::get_device_name(enc_state));
    
    // Initialize transcribe decoder
    transcribe::decoder::Config dec_config;
    dec_config.model_path = config.model_path;
    dec_config.device_name = config.device;
    dec_config.n_threads = config.threads;
    dec_config.max_ctx_length = 4096;
    
    auto dec_state = transcribe::decoder::init(dec_config);
    if (!dec_state) {
        fprintf(stderr, "Error: Failed to init transcribe decoder\n");
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    auto dec_hparams = transcribe::decoder::get_hparams(dec_state);
    printf("Transcribe decoder initialized (vocab=%d, hidden=%d)\n", 
           dec_hparams.vocab_size, dec_hparams.hidden_size);
    
    // Initialize aligner modules (if needed)
    aligner::encoder::EncoderState* align_enc_state = nullptr;
    aligner::decoder::State* align_dec_state = nullptr;
    
    if (!config.transcribe_only) {
        aligner::encoder::Config align_enc_config;
        align_enc_config.model_path = config.aligner_model_path;
        align_enc_config.device_name = config.device;
        align_enc_config.n_threads = config.threads;
        
        align_enc_state = aligner::encoder::init(align_enc_config);
        if (!align_enc_state) {
            fprintf(stderr, "Error: Failed to init aligner encoder\n");
            transcribe::decoder::free(dec_state);
            transcribe::encoder::free(enc_state);
            return 1;
        }
        
        aligner::decoder::Config align_dec_config;
        align_dec_config.model_path = config.aligner_model_path;
        align_dec_config.device_name = config.device;
        align_dec_config.n_threads = config.threads;
        
        align_dec_state = aligner::decoder::init(align_dec_config);
        if (!align_dec_state) {
            fprintf(stderr, "Error: Failed to init aligner decoder\n");
            aligner::encoder::free(align_enc_state);
            transcribe::decoder::free(dec_state);
            transcribe::encoder::free(enc_state);
            return 1;
        }
        
        printf("Aligner modules initialized\n");
    }
    
    printf("\n--- Step 4: Transcribe ---\n");
    
    // Encode audio
    transcribe::encoder::BatchInput enc_input;
    enc_input.mel_data.push_back(mel_spec.data.data());
    enc_input.n_frames.push_back(mel_spec.n_frames);
    enc_input.n_mels = mel_spec.n_mels;
    enc_input.max_frames = mel_spec.n_frames;
    
    transcribe::encoder::BatchOutput enc_output;
    
    if (!transcribe::encoder::encode_batch(enc_state, enc_input, enc_output, &error)) {
        fprintf(stderr, "Error: Encoder failed: %s\n", error.message.c_str());
        if (align_dec_state) aligner::decoder::free(align_dec_state);
        if (align_enc_state) aligner::encoder::free(align_enc_state);
        transcribe::decoder::free(dec_state);
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    auto& audio_features = enc_output.features[0];
    printf("Audio features: hidden=%d, frames=%d\n", 
           audio_features.hidden_size, audio_features.n_frames);
    
    // Use new transcribe API
    transcribe::decoder::TranscribeInput transcribe_in;
    transcribe_in.audio_features = audio_features.data.data();
    transcribe_in.n_audio_frames = audio_features.n_frames;
    transcribe_in.audio_feature_dim = audio_features.hidden_size;
    transcribe_in.max_tokens = config.max_tokens;
    transcribe_in.language = config.language;
    transcribe_in.context = config.context;
    transcribe_in.hotwords = config.hotwords;
    
    transcribe::decoder::TranscribeOutput transcribe_out;
    
    if (!transcribe::decoder::transcribe(dec_state, transcribe_in, transcribe_out, &error)) {
        fprintf(stderr, "Error: Transcribe failed: %s\n", error.message.c_str());
        if (align_dec_state) aligner::decoder::free(align_dec_state);
        if (align_enc_state) aligner::encoder::free(align_enc_state);
        transcribe::decoder::free(dec_state);
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    printf("Generated %d tokens\n", transcribe_out.n_tokens);
    
    if (config.json_output) {
        printf("\n--- Output (JSON) ---\n");
        printf("{\n");
        printf("  \"language\": \"%s\",\n", transcribe_out.language.c_str());
        printf("  \"text\": \"%s\",\n", transcribe_out.text.c_str());
        printf("  \"n_tokens\": %d\n", transcribe_out.n_tokens);
        printf("}\n");
    } else {
        printf("\n--- Output ---\n");
        printf("Language: %s\n", transcribe_out.language.empty() ? "unknown" : transcribe_out.language.c_str());
        printf("Text: %s\n", transcribe_out.text.c_str());
    }
    
    // Cleanup
    printf("\n--- Cleanup ---\n");
    if (align_dec_state) aligner::decoder::free(align_dec_state);
    if (align_enc_state) aligner::encoder::free(align_enc_state);
    transcribe::decoder::free(dec_state);
    transcribe::encoder::free(enc_state);
    
    printf("Done\n");
    
    return 0;
}