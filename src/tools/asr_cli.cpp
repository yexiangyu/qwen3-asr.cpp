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

using namespace asr;

static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF);
}

static std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        uint8_t c = s[i];
        if ((c & 0x80) == 0) {
            cp = c; i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < s.size()) {
                cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
                i += 2;
            } else i += 1;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < s.size()) {
                cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
                i += 3;
            } else i += 1;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < s.size()) {
                cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
                i += 4;
            } else i += 1;
        } else {
            i += 1;
        }
        cps.push_back(cp);
    }
    return cps;
}

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s = char(cp);
    } else if (cp < 0x800) {
        s = char(0xC0 | (cp >> 6));
        s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s = char(0xE0 | (cp >> 12));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    } else {
        s = char(0xF0 | (cp >> 18));
        s += char(0x80 | ((cp >> 12) & 0x3F));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    }
    return s;
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

void output_json(const std::string& language, const std::string& text, int n_tokens, const std::vector<aligner::decoder::AlignedWord>& words) {
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
    
    printf("  \"audio_duration\": %.3f\n", words.empty() ? 0.0f : words.back().end);
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
    
    float audio_duration = (float)audio_samples.size() / sample_rate;
    printf("Audio: %zu samples, %d Hz, %.2f seconds\n", audio_samples.size(), sample_rate, audio_duration);
    
    printf("\n--- Step 2: Compute mel spectrogram ---\n");
    mel::Config mel_config;
    mel_config.n_threads = config.threads;
    
    mel::MelSpectrum mel_spec;
    if (!mel::compute_from_file(config.input_path.c_str(), mel_spec, mel_config, &error)) {
        fprintf(stderr, "Error: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    printf("Mel: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    
    std::string transcribe_text;
    std::string transcribe_lang;
    int transcribe_n_tokens = 0;
    std::vector<aligner::decoder::AlignedWord> aligned_words;
    
    transcribe::decoder::State* dec_state = nullptr;
    
    if (!config.align_only) {
        printf("\n--- Step 3: Initialize ASR modules ---\n");
        
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
        
        printf("\n--- Step 4: Transcribe ---\n");
        
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
        
        printf("\n--- Step 6: Encode for alignment ---\n");
        
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
        
        printf("\n--- Step 7: Align ---\n");
        printf("Text to align: %s\n", transcribe_text.c_str());
        
        // Use transcribe decoder's tokenizer (same tokenizer as aligner)
        std::vector<std::string> words;
        std::vector<int32_t> text_tokens;
        if (!config.align_only && dec_state) {
            // We have transcribe decoder, use its tokenizer
            // Get tokens from transcribe output and decode them
            text_tokens = transcribe::decoder::tokenize(dec_state, transcribe_text);
            
            // Build words from text (simple char-level for CJK)
            auto cps = utf8_to_codepoints(transcribe_text);
            std::string latin_seq;
            for (auto cp : cps) {
                if (is_cjk(cp)) {
                    if (!latin_seq.empty()) {
                        words.push_back(latin_seq);
                        latin_seq.clear();
                    }
                    words.push_back(codepoint_to_utf8(cp));
                } else if (cp < 0x80 && (isalpha(cp) || cp == '\'')) {
                    latin_seq += char(cp);
                } else {
                    if (!latin_seq.empty()) {
                        words.push_back(latin_seq);
                        latin_seq.clear();
                    }
                    if (cp < 0x80 && !isspace(cp)) {
                        words.push_back(codepoint_to_utf8(cp));
                    }
                }
            }
            if (!latin_seq.empty()) {
                words.push_back(latin_seq);
            }
        } else {
            // Align-only mode: need separate tokenizer
            // For now, use simple char-level tokenization
            auto cps = utf8_to_codepoints(transcribe_text);
            for (auto cp : cps) {
                std::string ch = codepoint_to_utf8(cp);
                if (!isspace(cp)) {
                    words.push_back(ch);
                }
            }
            // This will likely fail for align-only mode without proper BPE tokenizer
            fprintf(stderr, "Warning: Align-only mode may not work correctly without proper BPE tokenizer\n");
        }
        
        if (text_tokens.empty()) {
            fprintf(stderr, "Error: Failed to tokenize text for alignment\n");
            aligner::decoder::free(align_dec_state);
            aligner::encoder::free(align_enc_state);
            return 1;
        }
        
        printf("Tokenized: %zu tokens, %zu words\n", text_tokens.size(), words.size());
        
        // Build alignment input manually
        std::vector<int32_t> align_tokens = aligner::decoder::build_token_sequence(
            align_dec_state, align_features.n_frames, text_tokens);
        
        int audio_start_pos = 0;
        for (size_t i = 0; i < align_tokens.size(); ++i) {
            if (align_tokens[i] == align_hparams.audio_start_token_id) {
                audio_start_pos = i + 1;
                break;
            }
        }
        
        aligner::decoder::Input dec_input;
        dec_input.tokens = align_tokens.data();
        dec_input.n_tokens = align_tokens.size();
        dec_input.audio_features = align_features.data.data();
        dec_input.n_audio_frames = align_features.n_frames;
        dec_input.audio_start_pos = audio_start_pos;
        
        aligner::decoder::Output dec_output;
        if (!aligner::decoder::decode(align_dec_state, dec_input, dec_output, &error)) {
            fprintf(stderr, "Error: Alignment decode failed: %s\n", error.message.c_str());
            aligner::decoder::free(align_dec_state);
            aligner::encoder::free(align_enc_state);
            return 1;
        }
        
        // Convert timestamps to words
        float segment_time = align_hparams.timestamp_segment_time_ms / 1000.0f;
        int token_idx = align_features.n_frames + 2;
        
        for (size_t w = 0; w < words.size() && w < text_tokens.size(); ++w) {
            if (token_idx + 1 >= (int)dec_output.timestamp_indices.size()) break;
            
            int ts_idx = token_idx + 1;
            int ts_class = dec_output.timestamp_indices[ts_idx];
            
            aligner::decoder::AlignedWord aw;
            aw.word = words[w];
            aw.start = ts_class * segment_time;
            aw.end = aw.start + segment_time;
            
            aligned_words.push_back(aw);
            token_idx += 2;
        }
        
        printf("Aligned %zu words\n", aligned_words.size());
        
        aligner::decoder::free(align_dec_state);
        aligner::encoder::free(align_enc_state);
    }
    
    if (config.json_output) {
        output_json(transcribe_lang, transcribe_text, transcribe_n_tokens, aligned_words);
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
                out_file << "  \"audio_duration\": " << (aligned_words.empty() ? audio_duration : aligned_words.back().end) << "\n";
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