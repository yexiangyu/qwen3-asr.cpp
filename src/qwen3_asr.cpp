#include "qwen3_asr.h"
#include "audio_utils.h"
#include "timing.h"
#include "logger.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cctype>

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string extract_language_prefix(const std::string & text) {
    std::string parsed_lang, parsed_text;
    std::tie(parsed_lang, parsed_text) = parse_asr_output(text);
    
    if (!parsed_lang.empty()) {
        return "language " + parsed_lang;
    }
    return "";
}

static std::string extract_text_content(const std::string & text) {
    std::string parsed_lang, parsed_text;
    std::tie(parsed_lang, parsed_text) = parse_asr_output(text);
    return parsed_text;
}

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }
    
    if (!decoder_.load_model(model_path)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }
    
    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
    
    model_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    LOG_INFO("Model loaded in {} ms", (long long)(t_end - t_start));
    
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return transcribe_internal(samples.data(), samples.size(), params);
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        LOG_INFO("Mel spectrogram: [{}, {}]", mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    if (params.print_progress) {
        LOG_INFO("Audio features: [{}, {}]", n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.context, params.language);
    
    if (params.print_progress) {
        LOG_INFO("Input tokens: {}", input_tokens.size());
    }
    
    if (params.debug_input) {
        LOG_INFO("=== DEBUG: Input tokens ({}) ===", input_tokens.size());
        std::string decoded_input;
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            int32_t t = input_tokens[i];
            std::string token_str = decoder_.decode_token(t);
            decoded_input += token_str;
            if (i < 20 || i >= input_tokens.size() - 10) {
                LOG_INFO("  [{:4}] {:8} | {}", i, t, token_str);
            } else if (i == 20) {
                LOG_INFO("  ... (skipping {} tokens)", input_tokens.size() - 30);
            }
        }
        LOG_INFO("  Decoded input: {}", decoded_input);
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    std::vector<float> output_confs;
    if (!decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens, output_confs)) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    if (params.debug_output) {
        LOG_INFO("=== DEBUG: Output tokens ({}) ===", output_tokens.size());
        std::string decoded_output;
        for (size_t i = 0; i < output_tokens.size(); ++i) {
            int32_t t = output_tokens[i];
            std::string token_str = decoder_.decode_token(t);
            decoded_output += token_str;
            if (i < 30 || i >= output_tokens.size() - 10) {
                LOG_INFO("  [{:4}] {:8} | {} | conf={:.4f}", i, t, token_str, output_confs[i]);
            } else if (i == 30) {
                LOG_INFO("  ... (skipping {} tokens)", output_tokens.size() - 40);
            }
        }
        LOG_INFO("  Decoded output: {}", decoded_output);
    }
    
    result.tokens = output_tokens;
    result.token_confidences = output_confs;
    result.text = decoder_.decode_tokens(output_tokens);
    result.text_prefix = extract_language_prefix(result.text);
    result.text_content = extract_text_content(result.text);
    
    for (int32_t tok : output_tokens) {
        result.token_strings.push_back(decoder_.decode_token(tok));
    }
    
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        LOG_INFO("Timing:");
        LOG_INFO("  Mel spectrogram: {} ms", (long long)result.t_mel_ms);
        LOG_INFO("  Audio encoding:  {} ms", (long long)result.t_encode_ms);
        LOG_INFO("  Text decoding:   {} ms", (long long)result.t_decode_ms);
        LOG_INFO("  Total:           {} ms", (long long)result.t_total_ms);
        LOG_INFO("  Tokens generated: {}", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & context,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 100);
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    
    if (!context.empty()) {
        std::vector<int32_t> context_tokens = decoder_.tokenize(context);
        for (int32_t t : context_tokens) {
            tokens.push_back(t);
        }
    }
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    if (!language.empty()) {
        std::string lang_prompt = "language " + language + "\n";
        std::vector<int32_t> lang_tokens = decoder_.tokenize(lang_prompt);
        for (int32_t t : lang_tokens) {
            tokens.push_back(t);
        }
    }
    
    return tokens;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                               const std::vector<float> & audio_features,
                               int32_t n_audio_frames,
                               const transcribe_params & params,
                               std::vector<int32_t> & output_tokens,
                               std::vector<float> & output_confs) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    int32_t audio_start_pos = 0;
    
    if (params.context.empty()) {
        audio_start_pos = 9;
    } else {
        std::vector<int32_t> context_tokens = decoder_.tokenize(params.context);
        audio_start_pos = 9 + static_cast<int32_t>(context_tokens.size());
    }
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    auto [next_token, next_conf] = sample_greedy_with_conf(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_confs.clear();
    output_tokens.push_back(next_token);
    output_confs.push_back(next_conf);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector<int32_t> single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        auto [tok, conf] = sample_greedy_with_conf(logits.data(), vocab_size);
        next_token = tok;
        next_conf = conf;
        output_tokens.push_back(next_token);
        output_confs.push_back(next_conf);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            int percent = static_cast<int>(output_tokens.size() * 100 / params.max_tokens);
            LOG_INFO("Decoding {}/{} tokens ({}%)", output_tokens.size(), params.max_tokens, percent);
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
        output_confs.pop_back();
    }
    
    return true;
}

std::pair<int32_t, float> Qwen3ASR::sample_greedy_with_conf(const float * logits, int32_t vocab_size) {
    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    int32_t max_idx = 0;
    float max_conf = expf(logits[0] - max_logit) / sum_exp;
    for (int32_t i = 1; i < vocab_size; ++i) {
        float conf = expf(logits[i] - max_logit) / sum_exp;
        if (conf > max_conf) {
            max_conf = conf;
            max_idx = i;
        }
    }
    
    return {max_idx, max_conf};
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    return load_wav(path, samples, sample_rate);
}

} // namespace qwen3_asr
