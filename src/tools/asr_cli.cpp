#include "asr/codec/codec.hpp"
#include "asr/mel/mel.hpp"
#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/decoder.hpp"
#include "asr/aligner/encoder.hpp"
#include "asr/aligner/decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::string tokens_to_text(const std::vector<int>& tokens, const transcribe::decoder::HyperParams& hparams) {
    return "Generated text placeholder";  // TODO: implement proper tokenizer
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
    
    // Build token sequence for prefill (matching original qwen3_asr.cpp format)
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    std::vector<int> tokens;
    
    // System message
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // User message with audio
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // Audio tokens
    tokens.push_back(dec_hparams.audio_start_token);
    for (int i = 0; i < audio_features.n_frames; ++i) {
        tokens.push_back(dec_hparams.audio_pad_token);
    }
    tokens.push_back(dec_hparams.audio_end_token);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // Assistant message
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    // Language prompt (optional)
    if (!config.language.empty()) {
        // TODO: tokenize language prompt properly
        // For now, just add a generic prompt
    }
    
    printf("Token sequence: %zu tokens\n", tokens.size());
    printf("Audio frames in sequence: %d\n", audio_features.n_frames);
    
    // Prefill
    int audio_start_pos = 9;  // Position of first audio_pad token (audio_start_token at pos 8 stays as embedding)
    
    transcribe::decoder::PrefillInput prefill_input;
    prefill_input.tokens = tokens.data();
    prefill_input.n_tokens = tokens.size();
    prefill_input.audio_features = audio_features.data.data();
    prefill_input.n_audio_frames = audio_features.n_frames;
    prefill_input.audio_feature_dim = audio_features.hidden_size;
    prefill_input.audio_start_pos = audio_start_pos;
    
    transcribe::decoder::DecoderOutput prefill_output;
    
    if (!transcribe::decoder::prefill(dec_state, prefill_input, prefill_output, &error)) {
        fprintf(stderr, "Error: Decoder prefill failed: %s\n", error.message.c_str());
        if (align_dec_state) aligner::decoder::free(align_dec_state);
        if (align_enc_state) aligner::encoder::free(align_enc_state);
        transcribe::decoder::free(dec_state);
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    printf("Prefill complete, KV cache: %d\n", transcribe::decoder::get_kv_cache_used(dec_state));
    
    // Generate tokens
    std::vector<int> generated_tokens;
    int n_past = tokens.size();
    int next_token = 0;
    
    // Get first token from prefill output
    int max_idx = 0;
    float max_val = prefill_output.logits[0];
    for (int i = 1; i < prefill_output.vocab_size; ++i) {
        if (prefill_output.logits[i] > max_val) {
            max_val = prefill_output.logits[i];
            max_idx = i;
        }
    }
    next_token = max_idx;
    
    printf("First token from prefill: %d (logit=%.3f)\n", next_token, max_val);
    printf("EOS token ID: %d\n", dec_hparams.eos_token);
    
    // Show top 10 tokens
    std::vector<std::pair<float, int>> logits_sorted;
    for (int i = 0; i < prefill_output.vocab_size; ++i) {
        logits_sorted.push_back({prefill_output.logits[i], i});
    }
    std::sort(logits_sorted.begin(), logits_sorted.end(), std::greater<std::pair<float,int>>());
    printf("Top 10 tokens:\n");
    for (int i = 0; i < 10; ++i) {
        printf("  token %d: id=%d, logit=%.3f\n", i, logits_sorted[i].second, logits_sorted[i].first);
    }
    
    // Generate tokens
    if (next_token != dec_hparams.eos_token && next_token != im_end) {
        generated_tokens.push_back(next_token);  // Add first generated token
        
        printf("Decoding...\n");
        
        for (int step = 1; step < config.max_tokens; ++step) {
            transcribe::decoder::DecodeInput decode_input;
            decode_input.tokens = &next_token;
            decode_input.n_tokens = 1;
            decode_input.n_past = n_past;
            
            transcribe::decoder::DecoderOutput decode_output;
            
            if (!transcribe::decoder::decode(dec_state, decode_input, decode_output, &error)) {
                fprintf(stderr, "Warning: Decode step %d failed: %s\n", step, error.message.c_str());
                break;
            }
            
            max_idx = 0;
            max_val = decode_output.logits[0];
            for (int i = 1; i < decode_output.vocab_size; ++i) {
                if (decode_output.logits[i] > max_val) {
                    max_val = decode_output.logits[i];
                    max_idx = i;
                }
            }
            next_token = max_idx;
            
            if (step % 10 == 0 || step < 10) {
                printf("  Step %d: token=%d, logit=%.3f, n_past=%d\n", step, next_token, max_val, n_past);
            }
            
            if (next_token == dec_hparams.eos_token || next_token == im_end) {
                printf("  EOS/im_end token at step %d: token=%d\n", step, next_token);
                break;
            }
            
            generated_tokens.push_back(next_token);
            n_past++;
        }
    }
    
    printf("Generated %zu tokens\n", generated_tokens.size());
    
    if (config.json_output) {
        printf("\n--- Output (JSON) ---\n");
        printf("{\n");
        printf("  \"text\": \"");
        for (int t : generated_tokens) {
            printf("%d ", t);
        }
        printf("\",\n");
        printf("  \"n_tokens\": %zu\n", generated_tokens.size());
        printf("}\n");
    } else {
        printf("\n--- Output ---\n");
        printf("Generated tokens: [");
        for (size_t i = 0; i < std::min(generated_tokens.size(), (size_t)20); ++i) {
            printf("%d", generated_tokens[i]);
            if (i < generated_tokens.size() - 1) printf(", ");
        }
        if (generated_tokens.size() > 20) printf("...");
        printf("]\n");
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