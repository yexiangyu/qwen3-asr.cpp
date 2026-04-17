#include "asr/aligner/decoder.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace qwen3_asr::asr::aligner::decoder;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [device_name]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* device_name = argc > 2 ? argv[2] : "";
    
    printf("Testing aligner decoder module...\n");
    printf("Model path: %s\n", model_path);
    printf("Device: %s\n", device_name);
    
    Config config;
    config.model_path = model_path;
    config.device_name = device_name;
    config.n_threads = 4;
    config.max_ctx_length = 4096;
    
    printf("\n=== Test 1: init ===\n");
    State* state = init(config);
    if (!state) {
        fprintf(stderr, "FAILED: init returned nullptr\n");
        return 1;
    }
    printf("PASSED: init succeeded\n");
    
    printf("\n=== Test 2: get_hparams ===\n");
    HyperParams hp = get_hparams(state);
    printf("hidden_size: %d\n", hp.hidden_size);
    printf("n_layers: %d\n", hp.n_layers);
    printf("n_heads: %d\n", hp.n_heads);
    printf("n_kv_heads: %d\n", hp.n_kv_heads);
    printf("head_dim: %d\n", hp.head_dim);
    printf("classify_head_size: %d\n", hp.classify_head_size);
    printf("rope_theta: %.1f\n", hp.rope_theta);
    printf("PASSED: get_hparams\n");
    
    printf("\n=== Test 3: get_device_name ===\n");
    printf("Device: %s\n", get_device_name(state));
    printf("PASSED: get_device_name\n");
    
    printf("\n=== Test 4: get_kv_cache_capacity ===\n");
    printf("KV cache capacity: %d\n", get_kv_cache_capacity(state));
    printf("PASSED: get_kv_cache_capacity\n");
    
    printf("\n=== Test 5: decode with synthetic input ===\n");
    const int n_tokens = 10;
    const int n_audio_frames = 5;
    const int hidden_size = hp.hidden_size;
    
    std::vector<int32_t> tokens(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        tokens[i] = hp.timestamp_token_id;
    }
    
    std::vector<float> audio_features(n_audio_frames * hidden_size);
    for (int i = 0; i < n_audio_frames * hidden_size; ++i) {
        audio_features[i] = 0.01f * (i % 100);
    }
    
    Input input;
    input.tokens = tokens.data();
    input.n_tokens = n_tokens;
    input.audio_features = audio_features.data();
    input.n_audio_frames = n_audio_frames;
    input.audio_start_pos = 2;
    
    Output output;
    ErrorInfo error;
    
    bool result = decode(state, input, output, &error);
    if (!result) {
        fprintf(stderr, "FAILED: decode returned false: %s\n", error.message.c_str());
        free(state);
        return 1;
    }
    
    printf("Output logits size: %zu\n", output.logits.size());
    printf("Output n_classes: %d\n", output.n_classes);
    printf("Output timestamp_indices size: %zu\n", output.timestamp_indices.size());
    
    if (output.n_classes != hp.classify_head_size) {
        fprintf(stderr, "FAILED: n_classes mismatch (expected %d, got %d)\n", 
                hp.classify_head_size, output.n_classes);
        free(state);
        return 1;
    }
    
    if (output.logits.size() != (size_t)n_tokens * output.n_classes) {
        fprintf(stderr, "FAILED: logits size mismatch (expected %zu, got %zu)\n",
                (size_t)n_tokens * output.n_classes, output.logits.size());
        free(state);
        return 1;
    }
    
    printf("First few timestamp indices: ");
    for (int i = 0; i < std::min(5, (int)output.timestamp_indices.size()); ++i) {
        printf("%d ", output.timestamp_indices[i]);
    }
    printf("\n");
    
    printf("Timestamp range check:\n");
    int max_ts = 0;
    for (int ts : output.timestamp_indices) {
        max_ts = std::max(max_ts, ts);
    }
    printf("Max timestamp index: %d (expected < %d)\n", max_ts, hp.classify_head_size);
    if (max_ts >= hp.classify_head_size) {
        fprintf(stderr, "FAILED: timestamp index out of range\n");
        free(state);
        return 1;
    }
    
    printf("PASSED: decode with synthetic input\n");
    
    printf("\n=== Test 6: convert_to_timestamps ===\n");
    TimestampResult ts_result = convert_to_timestamps(output, hp.timestamp_segment_time_ms);
    printf("Converted timestamps: ");
    for (int i = 0; i < std::min(5, (int)ts_result.timestamps.size()); ++i) {
        printf("%.3f ", ts_result.timestamps[i]);
    }
    printf("\n");
    printf("n_words: %d\n", ts_result.n_words);
    printf("PASSED: convert_to_timestamps\n");
    
    printf("\n=== Test 7: clear_kv_cache ===\n");
    clear_kv_cache(state);
    printf("KV cache used after clear: %d\n", get_kv_cache_used(state));
    if (get_kv_cache_used(state) != 0) {
        fprintf(stderr, "FAILED: KV cache not cleared\n");
        free(state);
        return 1;
    }
    printf("PASSED: clear_kv_cache\n");
    
    printf("\n=== Test 8: free ===\n");
    free(state);
    printf("PASSED: free\n");
    
    printf("\n=== All tests passed ===\n");
    return 0;
}