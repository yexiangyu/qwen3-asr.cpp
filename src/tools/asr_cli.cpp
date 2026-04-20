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
#include <cctype>
#include <cmath>
#include <chrono>

using namespace asr;

static std::string strip_asr_language_prefix(const std::string& text) {
    if (text.empty()) return text;
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\t')) ++start;
    if (start < text.size() && text.compare(start, 8, "language") == 0) {
        size_t after_lang = start + 8;
        while (after_lang < text.size() && text[after_lang] == ' ') ++after_lang;
        size_t name_end = after_lang;
        while (name_end < text.size() && text[name_end] != ' ' && text[name_end] != '\n') ++name_end;
        size_t text_start = name_end;
        while (text_start < text.size() && (text[text_start] == ' ' || text[text_start] == '\t' || text[text_start] == '\n')) ++text_start;
        return text.substr(text_start);
    }
    return text;
}

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
    printf("  --model <path>           ASR model (default: models/qwen3-asr-1.7b-f16.gguf)\n");
    printf("  --aligner <path>         Aligner model (default: models/qwen3-forced-aligner-0.6b-f16.gguf)\n");
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
        
        if (arg == "--input" && i + 1 < argc) config.input_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) config.model_path = argv[++i];
        else if (arg == "--aligner" && i + 1 < argc) config.aligner_model_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) config.device = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) config.threads = std::atoi(argv[++i]);
        else if (arg == "--language" && i + 1 < argc) config.language = argv[++i];
        else if (arg == "--context" && i + 1 < argc) config.context = argv[++i];
        else if (arg == "--hotwords" && i + 1 < argc) config.hotwords = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) config.max_tokens = std::atoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) config.output_path = argv[++i];
        else if (arg == "--format" && i + 1 < argc) config.json_output = (std::string(argv[++i]) == "json");
        else if (arg == "--transcribe-only") config.transcribe_only = true;
        else if (arg == "--align-only") config.align_only = true;
        else if (arg == "--text" && i + 1 < argc) config.text_for_align = argv[++i];
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

std::string escape_json(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

void output_text(const std::string& language, const std::string& text, const std::vector<aligner::decoder::AlignedWord>& words) {
    printf("\n--- Output ---\n");
    printf("Language: %s\n", language.empty() ? "unknown" : language.c_str());
    printf("Text: %s\n", text.c_str());
    
    if (!words.empty()) {
        printf("\n--- Word Timestamps ---\n");
        for (const auto& w : words) {
            printf("  %.3f - %.3f: %s\n", w.start, w.end, w.word.c_str());
        }
    }
}

void output_json(const std::string& language, const std::string& text, int n_tokens, const std::vector<aligner::decoder::AlignedWord>& words, float audio_duration) {
    printf("{\n");
    printf("  \"language\": \"%s\",\n", escape_json(language).c_str());
    printf("  \"text\": \"%s\",\n", escape_json(text).c_str());
    printf("  \"n_tokens\": %d,\n", n_tokens);
    
    if (!words.empty()) {
        printf("  \"words\": [\n");
        for (size_t i = 0; i < words.size(); ++i) {
            const auto& w = words[i];
            printf("    {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f}%s\n",
                   escape_json(w.word).c_str(), w.start, w.end,
                   i < words.size() - 1 ? "," : "");
        }
        printf("  ],\n");
    }
    
    printf("  \"audio_duration\": %.3f\n", audio_duration);
    printf("}\n");
}

int main(int argc, char** argv) {
    CliConfig config;
    
    if (!parse_args(argc, argv, config)) return 1;
    
    printf("=== ASR CLI ===\n\n");
    printf("Input: %s\n", config.input_path.c_str());
    printf("Model: %s\n", config.model_path.c_str());
    printf("Aligner: %s\n", config.aligner_model_path.c_str());
    printf("Device: %s\n", config.device.c_str());
    printf("Threads: %d\n", config.threads);
    
    ErrorInfo error;
    
    printf("\n--- Step 1: Load audio ---\n");
    std::vector<float> audio_samples;
    int sample_rate;
    
    if (!codec::decode_file(config.input_path.c_str(), audio_samples, sample_rate, &error)) {
        fprintf(stderr, "Error: Failed to load audio: %s\n", error.message.c_str());
        return 1;
    }
    
    float audio_len_sec = (float)audio_samples.size() / sample_rate;
    printf("Audio: %zu samples, %d Hz, %.2f seconds\n", audio_samples.size(), sample_rate, audio_len_sec);
    
    printf("\n--- Step 2: Compute mel spectrogram ---\n");
    mel::Config mel_config;
    mel_config.n_threads = config.threads;
    
    auto mel_start = std::chrono::high_resolution_clock::now();
    mel::MelSpectrum mel_spec;
    mel::MelState mel_state;
    mel::Input mel_input;
    mel_input.samples = audio_samples.data();
    mel_input.n_samples = static_cast<int>(audio_samples.size());
    if (!mel::compute_cached(mel_state, mel_input, mel_spec, mel_config, &error)) {
        fprintf(stderr, "Error: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    auto mel_end = std::chrono::high_resolution_clock::now();
    double mel_ms = std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
    printf("Mel: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    printf("[Timing] Mel spectrogram: %.1f ms\n", mel_ms);
    
    std::string transcribe_text;
    std::string transcribe_lang;
    int transcribe_n_tokens = 0;
    std::vector<aligner::decoder::AlignedWord> aligned_words;
    float audio_duration = 0.0f;
    
    transcribe::decoder::State* dec_state = nullptr;
    
    if (!config.align_only) {
        printf("\n--- Step 3: Initialize ASR modules ---\n");
        
        auto t3_start = std::chrono::high_resolution_clock::now();
        
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
        
        transcribe::decoder::Config dec_config;
        dec_config.model_path = config.model_path;
        dec_config.device_name = config.device;
        dec_config.n_threads = config.threads;
        dec_config.max_ctx_length = 4096;
        
        dec_state = transcribe::decoder::init(dec_config);
        if (!dec_state) {
            fprintf(stderr, "Error: Failed to init transcribe decoder\n");
            transcribe::encoder::free(enc_state);
            return 1;
        }
        
        auto dec_hparams = transcribe::decoder::get_hparams(dec_state);
        printf("Transcribe decoder initialized (vocab=%d, hidden=%d)\n", dec_hparams.vocab_size, dec_hparams.hidden_size);
        
        auto t3_end = std::chrono::high_resolution_clock::now();
        double init_ms = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();
        printf("[Timing] Init (encoder+decoder load): %.1f ms\n", init_ms);
        
        printf("\n--- Step 4: Transcribe ---\n");
        
        auto t4_enc_start = std::chrono::high_resolution_clock::now();
        
        transcribe::encoder::BatchInput enc_input;
        enc_input.mel_data.push_back(mel_spec.data.data());
        enc_input.n_frames.push_back(mel_spec.n_frames);
        enc_input.n_mels = mel_spec.n_mels;
        enc_input.max_frames = mel_spec.n_frames;
        
        transcribe::encoder::BatchOutput enc_output;
        if (!transcribe::encoder::encode_batch(enc_state, enc_input, enc_output, &error)) {
            fprintf(stderr, "Error: Encoder failed: %s\n", error.message.c_str());
            transcribe::decoder::free(dec_state);
            transcribe::encoder::free(enc_state);
            return 1;
        }
        
        auto& audio_features = enc_output.features[0];
        printf("Audio features: hidden=%d, frames=%d\n", audio_features.hidden_size, audio_features.n_frames);
        
        auto t4_enc_end = std::chrono::high_resolution_clock::now();
        double enc_ms = std::chrono::duration<double, std::milli>(t4_enc_end - t4_enc_start).count();
        printf("[Timing] Audio encode: %.1f ms\n", enc_ms);
        
        auto t4_dec_start = std::chrono::high_resolution_clock::now();
        
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
            transcribe::decoder::free(dec_state);
            transcribe::encoder::free(enc_state);
            return 1;
        }
        
        transcribe_text = transcribe_out.text;
        transcribe_lang = transcribe_out.language;
        transcribe_n_tokens = transcribe_out.n_tokens;
        
        printf("Generated %d tokens\n", transcribe_n_tokens);
        
        auto t4_dec_end = std::chrono::high_resolution_clock::now();
        double dec_ms = std::chrono::duration<double, std::milli>(t4_dec_end - t4_dec_start).count();
        printf("[Timing] Text decode: %.1f ms (%.2f ms/token)\n", dec_ms, dec_ms / transcribe_n_tokens);
        
        // Free encoder immediately, but keep decoder for alignment tokenization
        transcribe::encoder::free(enc_state);
        
        if (config.transcribe_only) {
            transcribe::decoder::free(dec_state);
            dec_state = nullptr;
        }
    } else {
        transcribe_text = config.text_for_align;
        transcribe_lang = config.language;
    }
    
    if (!config.transcribe_only) {
        printf("\n--- Step 5: Initialize Aligner modules ---\n");
        auto t5_start = std::chrono::high_resolution_clock::now();
        
        aligner::encoder::Config align_enc_config;
        align_enc_config.model_path = config.aligner_model_path;
        align_enc_config.device_name = config.device;
        align_enc_config.n_threads = config.threads;
        
        auto align_enc_state = aligner::encoder::init(align_enc_config);
        if (!align_enc_state) {
            fprintf(stderr, "Error: Failed to init aligner encoder\n");
            return 1;
        }
        printf("Aligner encoder initialized on %s\n", aligner::encoder::get_device_name(align_enc_state));
        
        aligner::decoder::Config align_dec_config;
        align_dec_config.model_path = config.aligner_model_path;
        align_dec_config.device_name = config.device;
        align_dec_config.n_threads = config.threads;
        
        auto align_dec_state = aligner::decoder::init(align_dec_config);
        if (!align_dec_state) {
            fprintf(stderr, "Error: Failed to init aligner decoder\n");
            aligner::encoder::free(align_enc_state);
            return 1;
        }
        
        auto align_hparams = aligner::decoder::get_hparams(align_dec_state);
        printf("Aligner decoder initialized (classify_head_size=%d, segment_time=%dms)\n",
               align_hparams.classify_head_size, align_hparams.timestamp_segment_time_ms);
        auto t5_end = std::chrono::high_resolution_clock::now();
        printf("[Timing] Aligner init: %.1f ms\n", std::chrono::duration<double, std::milli>(t5_end - t5_start).count());
        
        printf("\n--- Step 6: Encode for alignment ---\n");
        auto t6_start = std::chrono::high_resolution_clock::now();
        
        aligner::encoder::BatchInput align_enc_input;
        align_enc_input.mel_data.push_back(mel_spec.data.data());
        align_enc_input.n_frames.push_back(mel_spec.n_frames);
        align_enc_input.n_mels = mel_spec.n_mels;
        align_enc_input.max_frames = mel_spec.n_frames;
        
        aligner::encoder::BatchOutput align_enc_output;
        if (!aligner::encoder::encode_batch(align_enc_state, align_enc_input, align_enc_output, &error)) {
            fprintf(stderr, "Error: Aligner encoder failed: %s\n", error.message.c_str());
            aligner::decoder::free(align_dec_state);
            aligner::encoder::free(align_enc_state);
            return 1;
        }
        
        auto& align_features = align_enc_output.features[0];
        printf("Aligner features: hidden=%d, frames=%d\n", align_features.hidden_size, align_features.n_frames);
        auto t6_end = std::chrono::high_resolution_clock::now();
        printf("[Timing] Aligner encode: %.1f ms\n", std::chrono::duration<double, std::milli>(t6_end - t6_start).count());
        
        printf("\n--- Step 7: Align ---\n");
        auto t7_start = std::chrono::high_resolution_clock::now();
        std::string align_text = strip_asr_language_prefix(transcribe_text);
        printf("Text to align: %s (lang: %s)\n", align_text.c_str(), transcribe_lang.c_str());
        
        aligner::decoder::AlignInput align_in;
        align_in.audio_features = align_features.data.data();
        align_in.n_audio_frames = align_features.n_frames;
        align_in.audio_feature_dim = align_features.hidden_size;
        align_in.n_mel_frames = mel_spec.n_frames;
        align_in.text = align_text;
        align_in.language = transcribe_lang;
        align_in.n_samples = (int)audio_samples.size();
        align_in.sample_rate = sample_rate;
        
        aligner::decoder::AlignOutput align_out;
        if (!aligner::decoder::align(align_dec_state, align_in, align_out, &error)) {
            fprintf(stderr, "Error: Alignment failed: %s\n", error.message.c_str());
            aligner::decoder::free(align_dec_state);
            aligner::encoder::free(align_enc_state);
            return 1;
        }
        
        aligned_words = align_out.words;
        audio_duration = align_out.audio_duration;
        printf("Aligned %zu words, audio duration: %.2f sec\n", aligned_words.size(), audio_duration);
        auto t7_end = std::chrono::high_resolution_clock::now();
        printf("[Timing] Align decode: %.1f ms\n", std::chrono::duration<double, std::milli>(t7_end - t7_start).count());
        
        aligner::decoder::free(align_dec_state);
        aligner::encoder::free(align_enc_state);
    }
    
    if (config.json_output) {
        output_json(transcribe_lang, transcribe_text, transcribe_n_tokens, aligned_words, audio_duration);
    } else {
        output_text(transcribe_lang, transcribe_text, aligned_words);
    }
    
    if (!config.output_path.empty()) {
        std::ofstream out_file(config.output_path);
        if (out_file.is_open()) {
            if (config.json_output) {
                out_file << "{\n";
                out_file << "  \"language\": \"" << escape_json(transcribe_lang) << "\",\n";
                out_file << "  \"text\": \"" << escape_json(transcribe_text) << "\",\n";
                out_file << "  \"n_tokens\": " << transcribe_n_tokens << ",\n";
                if (!aligned_words.empty()) {
                    out_file << "  \"words\": [\n";
                    for (size_t i = 0; i < aligned_words.size(); ++i) {
                        const auto& w = aligned_words[i];
                        out_file << "    {\"word\": \"" << escape_json(w.word) << "\", ";
                        out_file << "\"start\": " << std::fixed << std::setprecision(3) << w.start << ", ";
                        out_file << "\"end\": " << w.end << "}";
                        if (i < aligned_words.size() - 1) out_file << ",";
                        out_file << "\n";
                    }
                    out_file << "  ],\n";
                }
                out_file << "  \"audio_duration\": " << std::fixed << std::setprecision(3) << audio_duration << "\n";
                out_file << "}\n";
            } else {
                out_file << "Language: " << transcribe_lang << "\n";
                out_file << "Text: " << transcribe_text << "\n";
                if (!aligned_words.empty()) {
                    out_file << "\nWord Timestamps:\n";
                    for (const auto& w : aligned_words) {
                        out_file << std::fixed << std::setprecision(3) << w.start << " - " << w.end << ": " << w.word << "\n";
                    }
                }
            }
            printf("\nOutput saved to: %s\n", config.output_path.c_str());
        }
    }
    
    printf("\n--- Cleanup ---\n");
    
    if (dec_state) {
        transcribe::decoder::free(dec_state);
        dec_state = nullptr;
    }
    
    printf("Done\n");
    
    return 0;
}