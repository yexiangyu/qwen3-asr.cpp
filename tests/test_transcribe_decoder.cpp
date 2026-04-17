#include "asr/transcribe/decoder.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace asr::transcribe::decoder;

bool test_kv_cache_management(State* state, const Config& config) {
    fprintf(stderr, "\n=== Testing KV Cache Management ===\n");
    bool passed = true;
    
    int capacity = get_kv_cache_capacity(state);
    fprintf(stderr, "KV cache capacity: %d (expected: %d)\n", capacity, config.max_ctx_length);
    if (capacity != config.max_ctx_length) {
        fprintf(stderr, "FAIL: KV cache capacity mismatch\n");
        passed = false;
    }
    
    int used = get_kv_cache_used(state);
    fprintf(stderr, "KV cache used after init: %d (expected: 0)\n", used);
    if (used != 0) {
        fprintf(stderr, "FAIL: KV cache should be empty after init\n");
        passed = false;
    }
    
    clear_kv_cache(state);
    used = get_kv_cache_used(state);
    fprintf(stderr, "KV cache used after clear: %d (expected: 0)\n", used);
    if (used != 0) {
        fprintf(stderr, "FAIL: KV cache should be empty after clear\n");
        passed = false;
    }
    
    fprintf(stderr, "=== KV Cache Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_prefill_without_audio(State* state) {
    fprintf(stderr, "\n=== Testing Prefill (No Audio) ===\n");
    bool passed = true;
    
    HyperParams hp = get_hparams(state);
    
    std::vector<int> tokens = {hp.audio_start_token, hp.audio_end_token, 100, 101, 102};
    int n_tokens = tokens.size();
    
    PrefillInput input;
    input.tokens = tokens.data();
    input.n_tokens = n_tokens;
    input.audio_features = nullptr;
    input.n_audio_frames = 0;
    input.audio_feature_dim = hp.hidden_size;
    input.audio_start_pos = -1;
    
    DecoderOutput output;
    ErrorInfo error;
    
    bool result = prefill(state, input, output, &error);
    
    if (!result) {
        fprintf(stderr, "FAIL: Prefill failed with error: %s\n", error.message.c_str());
        return false;
    }
    
    fprintf(stderr, "Prefill succeeded\n");
    fprintf(stderr, "Output logits size: %zu\n", output.logits.size());
    fprintf(stderr, "Output vocab_size: %d (expected: %d)\n", output.vocab_size, hp.vocab_size);
    
    if (output.vocab_size != hp.vocab_size) {
        fprintf(stderr, "FAIL: vocab_size mismatch\n");
        passed = false;
    }
    
    if (output.logits.empty()) {
        fprintf(stderr, "FAIL: logits vector is empty\n");
        passed = false;
    }
    
    int kv_used = get_kv_cache_used(state);
    fprintf(stderr, "KV cache used after prefill: %d (expected: %d)\n", kv_used, n_tokens);
    if (kv_used != n_tokens) {
        fprintf(stderr, "FAIL: KV cache used mismatch\n");
        passed = false;
    }
    
    fprintf(stderr, "=== Prefill (No Audio) Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_prefill_with_audio(State* state) {
    fprintf(stderr, "\n=== Testing Prefill (With Audio) ===\n");
    bool passed = true;
    
    HyperParams hp = get_hparams(state);
    
    int n_audio_frames = 5;
    std::vector<float> audio_features(n_audio_frames * hp.hidden_size, 0.5f);
    
    std::vector<int> tokens;
    tokens.push_back(hp.audio_start_token);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(hp.audio_pad_token);
    }
    tokens.push_back(hp.audio_end_token);
    tokens.push_back(100);
    tokens.push_back(101);
    
    int n_tokens = tokens.size();
    int audio_start_pos = 1;
    
    PrefillInput input;
    input.tokens = tokens.data();
    input.n_tokens = n_tokens;
    input.audio_features = audio_features.data();
    input.n_audio_frames = n_audio_frames;
    input.audio_feature_dim = hp.hidden_size;
    input.audio_start_pos = audio_start_pos;
    
    DecoderOutput output;
    ErrorInfo error;
    
    bool result = prefill(state, input, output, &error);
    
    if (!result) {
        fprintf(stderr, "FAIL: Prefill with audio failed with error: %s\n", error.message.c_str());
        return false;
    }
    
    fprintf(stderr, "Prefill with audio succeeded\n");
    fprintf(stderr, "Output logits size: %zu\n", output.logits.size());
    fprintf(stderr, "Output vocab_size: %d (expected: %d)\n", output.vocab_size, hp.vocab_size);
    
    if (output.vocab_size != hp.vocab_size) {
        fprintf(stderr, "FAIL: vocab_size mismatch\n");
        passed = false;
    }
    
    if (output.logits.empty()) {
        fprintf(stderr, "FAIL: logits vector is empty\n");
        passed = false;
    }
    
    int kv_used = get_kv_cache_used(state);
    fprintf(stderr, "KV cache used after prefill: %d (expected: %d)\n", kv_used, n_tokens);
    if (kv_used != n_tokens) {
        fprintf(stderr, "FAIL: KV cache used mismatch\n");
        passed = false;
    }
    
    fprintf(stderr, "=== Prefill (With Audio) Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_prefill_auto_detect_audio_positions(State* state) {
    fprintf(stderr, "\n=== Testing Prefill (Auto-detect Audio Positions) ===\n");
    bool passed = true;
    
    HyperParams hp = get_hparams(state);
    
    int n_audio_frames = 3;
    std::vector<float> audio_features(n_audio_frames * hp.hidden_size, 0.3f);
    
    std::vector<int> tokens;
    tokens.push_back(hp.audio_start_token);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(hp.audio_pad_token);
    }
    tokens.push_back(hp.audio_end_token);
    
    int n_tokens = tokens.size();
    
    PrefillInput input;
    input.tokens = tokens.data();
    input.n_tokens = n_tokens;
    input.audio_features = audio_features.data();
    input.n_audio_frames = n_audio_frames;
    input.audio_feature_dim = hp.hidden_size;
    input.audio_start_pos = -1;
    
    DecoderOutput output;
    ErrorInfo error;
    
    bool result = prefill(state, input, output, &error);
    
    if (!result) {
        fprintf(stderr, "FAIL: Prefill auto-detect failed with error: %s\n", error.message.c_str());
        return false;
    }
    
    fprintf(stderr, "Prefill auto-detect succeeded\n");
    fprintf(stderr, "Output logits size: %zu\n", output.logits.size());
    
    if (output.logits.empty()) {
        fprintf(stderr, "FAIL: logits vector is empty\n");
        passed = false;
    }
    
    fprintf(stderr, "=== Prefill (Auto-detect) Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

static bool validate_logits(const std::vector<float>& logits, const char* name) {
    bool has_nan = false;
    bool all_zero = true;
    float max_val = -1e10f;
    float min_val = 1e10f;
    
    for (float v : logits) {
        if (std::isnan(v)) has_nan = true;
        if (v != 0.0f) all_zero = false;
        max_val = std::max(max_val, v);
        min_val = std::min(min_val, v);
    }
    
    if (has_nan) {
        fprintf(stderr, "FAIL: %s logits contain NaN\n", name);
        return false;
    }
    if (all_zero) {
        fprintf(stderr, "FAIL: %s logits are all zeros\n", name);
        return false;
    }
    
    fprintf(stderr, "  %s logits range: [%.4f, %.4f]\n", name, min_val, max_val);
    return true;
}

bool test_decode_after_prefill(State* state) {
    fprintf(stderr, "\n=== Testing Decode After Prefill ===\n");
    bool passed = true;
    
    clear_kv_cache(state);
    
    HyperParams hp = get_hparams(state);
    
    std::vector<int> prefill_tokens = {hp.audio_start_token, hp.audio_end_token, 100};
    PrefillInput prefill_input;
    prefill_input.tokens = prefill_tokens.data();
    prefill_input.n_tokens = prefill_tokens.size();
    prefill_input.audio_features = nullptr;
    prefill_input.n_audio_frames = 0;
    prefill_input.audio_feature_dim = hp.hidden_size;
    prefill_input.audio_start_pos = -1;
    
    DecoderOutput prefill_output;
    ErrorInfo error;
    
    if (!prefill(state, prefill_input, prefill_output, &error)) {
        fprintf(stderr, "FAIL: Prefill failed: %s\n", error.message.c_str());
        return false;
    }
    
    if (!validate_logits(prefill_output.logits, "prefill")) {
        passed = false;
    }
    
    int n_past = get_kv_cache_used(state);
    fprintf(stderr, "KV cache after prefill: %d\n", n_past);
    
    std::vector<int> decode_tokens = {101, 102};
    DecodeInput decode_input;
    decode_input.tokens = decode_tokens.data();
    decode_input.n_tokens = decode_tokens.size();
    decode_input.n_past = n_past;
    
    DecoderOutput decode_output;
    
    if (!decode(state, decode_input, decode_output, &error)) {
        fprintf(stderr, "FAIL: Decode failed: %s\n", error.message.c_str());
        return false;
    }
    
    if (!validate_logits(decode_output.logits, "decode")) {
        passed = false;
    }
    
    fprintf(stderr, "Decode succeeded\n");
    fprintf(stderr, "Output logits size: %zu\n", decode_output.logits.size());
    
    if (decode_output.logits.empty()) {
        fprintf(stderr, "FAIL: decode logits vector is empty\n");
        passed = false;
    }
    
    int kv_used_after_decode = get_kv_cache_used(state);
    int expected_kv = n_past + (int)decode_tokens.size();
    fprintf(stderr, "KV cache after decode: %d (expected: %d)\n",
            kv_used_after_decode, expected_kv);
    
    if (kv_used_after_decode != expected_kv) {
        fprintf(stderr, "FAIL: KV cache used mismatch after decode\n");
        passed = false;
    }
    
    fprintf(stderr, "=== Decode After Prefill Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_autoregressive_generation(State* state) {
    fprintf(stderr, "\n=== Testing Autoregressive Generation ===\n");
    bool passed = true;
    
    clear_kv_cache(state);
    
    HyperParams hp = get_hparams(state);
    
    std::vector<int> prefill_tokens = {hp.audio_start_token, hp.audio_end_token};
    PrefillInput prefill_input;
    prefill_input.tokens = prefill_tokens.data();
    prefill_input.n_tokens = prefill_tokens.size();
    prefill_input.audio_features = nullptr;
    prefill_input.n_audio_frames = 0;
    prefill_input.audio_feature_dim = hp.hidden_size;
    prefill_input.audio_start_pos = -1;
    
    DecoderOutput prefill_output;
    ErrorInfo error;
    
    if (!prefill(state, prefill_input, prefill_output, &error)) {
        fprintf(stderr, "FAIL: Prefill failed: %s\n", error.message.c_str());
        return false;
    }
    
    int n_past = get_kv_cache_used(state);
    fprintf(stderr, "KV cache after prefill: %d\n", n_past);
    
    const int n_gen_tokens = 5;
    std::vector<int> generated_tokens;
    
    for (int i = 0; i < n_gen_tokens; ++i) {
        int token = 100 + i;
        generated_tokens.push_back(token);
        
        DecodeInput decode_input;
        decode_input.tokens = &token;
        decode_input.n_tokens = 1;
        decode_input.n_past = n_past;
        
        DecoderOutput decode_output;
        
        if (!decode(state, decode_input, decode_output, &error)) {
            fprintf(stderr, "FAIL: Decode step %d failed: %s\n", i, error.message.c_str());
            return false;
        }
        
        if (!validate_logits(decode_output.logits, "decode_step")) {
            passed = false;
        }
        
        n_past = get_kv_cache_used(state);
        
        int expected_kv = (int)prefill_tokens.size() + i + 1;
        if (n_past != expected_kv) {
            fprintf(stderr, "FAIL: KV cache mismatch at step %d: %d vs expected %d\n",
                    i, n_past, expected_kv);
            passed = false;
        }
    }
    
    fprintf(stderr, "Generated %d tokens autoregressively\n", n_gen_tokens);
    fprintf(stderr, "Final KV cache size: %d\n", n_past);
    
    int expected_final_kv = (int)prefill_tokens.size() + n_gen_tokens;
    if (n_past != expected_final_kv) {
        fprintf(stderr, "FAIL: Final KV cache mismatch: %d vs expected %d\n",
                n_past, expected_final_kv);
        passed = false;
    }
    
    fprintf(stderr, "=== Autoregressive Generation Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    Config config;
    config.model_path = argv[1];
    config.n_threads = 4;
    config.max_ctx_length = 4096;
    
    fprintf(stderr, "Loading decoder model from: %s\n", config.model_path.c_str());
    
    State* state = init(config);
    if (!state) {
        fprintf(stderr, "Failed to initialize decoder\n");
        return 1;
    }
    
    fprintf(stderr, "Decoder initialized successfully\n");
    fprintf(stderr, "Device: %s\n", get_device_name(state));
    
    HyperParams hp = get_hparams(state);
    fprintf(stderr, "Hyperparams:\n");
    fprintf(stderr, "  vocab_size: %d\n", hp.vocab_size);
    fprintf(stderr, "  hidden_size: %d\n", hp.hidden_size);
    fprintf(stderr, "  n_layers: %d\n", hp.n_layers);
    fprintf(stderr, "  n_heads: %d\n", hp.n_heads);
    fprintf(stderr, "  n_kv_heads: %d\n", hp.n_kv_heads);
    fprintf(stderr, "  head_dim: %d\n", hp.head_dim);
    fprintf(stderr, "  intermediate_size: %d\n", hp.intermediate_size);
    fprintf(stderr, "  rms_norm_eps: %.6f\n", hp.rms_norm_eps);
    fprintf(stderr, "  rope_theta: %.1f\n", hp.rope_theta);
    
    bool all_passed = true;
    
    all_passed &= test_kv_cache_management(state, config);
    
    all_passed &= test_prefill_without_audio(state);
    
    clear_kv_cache(state);
    all_passed &= test_prefill_with_audio(state);
    
    clear_kv_cache(state);
    all_passed &= test_prefill_auto_detect_audio_positions(state);
    
    clear_kv_cache(state);
    all_passed &= test_decode_after_prefill(state);
    
    clear_kv_cache(state);
    all_passed &= test_autoregressive_generation(state);
    
    free(state);
    fprintf(stderr, "\nDecoder freed successfully\n");
    
    fprintf(stderr, "\n=== All Tests %s ===\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}