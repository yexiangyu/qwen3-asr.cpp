#include "asr/aligner/decoder.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

using namespace asr::aligner::decoder;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [ref_output.bin] [device_name]\n", argv[0]);
        fprintf(stderr, "  model.gguf: Path to forced-aligner GGUF model\n");
        fprintf(stderr, "  ref_output.bin: Optional reference output for comparison\n");
        fprintf(stderr, "  device_name: Optional device name (e.g., CUDA0, Metal)\n");
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* ref_path = argc > 2 ? argv[2] : "";
    const char* device_name = argc > 3 ? argv[3] : "";
    
    printf("=== Aligner Decoder Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Reference: %s\n", ref_path);
    printf("Device: %s\n", device_name);
    
    Config config;
    config.model_path = model_path;
    config.device_name = device_name;
    config.n_threads = 4;
    config.max_ctx_length = 4096;
    
    printf("\n--- Initializing ---\n");
    State* state = init(config);
    if (!state) {
        fprintf(stderr, "ERROR: Failed to initialize decoder\n");
        return 1;
    }
    printf("Device: %s\n", get_device_name(state));
    
    HyperParams hp = get_hparams(state);
    printf("Hyperparams:\n");
    printf("  hidden_size=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d\n",
           hp.hidden_size, hp.n_layers, hp.n_heads, hp.n_kv_heads, hp.head_dim);
    printf("  classify_head_size=%d (timestamp classes)\n", hp.classify_head_size);
    printf("  timestamp_token_id=%d, audio_pad_token_id=%d\n",
           hp.timestamp_token_id, hp.audio_pad_token_id);
    
    const int n_tokens = 100;
    const int n_audio_frames = 50;
    const int audio_start_pos = 10;
    
    printf("\n--- Creating test input ---\n");
    printf("n_tokens=%d, n_audio_frames=%d, audio_start_pos=%d\n",
           n_tokens, n_audio_frames, audio_start_pos);
    
    std::vector<int32_t> tokens(n_tokens);
    for (int i = 0; i < audio_start_pos; ++i) {
        tokens[i] = hp.pad_token_id;
    }
    for (int i = audio_start_pos; i < audio_start_pos + n_audio_frames; ++i) {
        tokens[i] = hp.audio_pad_token_id;
    }
    for (int i = audio_start_pos + n_audio_frames; i < n_tokens; ++i) {
        tokens[i] = hp.timestamp_token_id;
    }
    
    std::vector<float> audio_features(n_audio_frames * hp.hidden_size);
    for (size_t i = 0; i < audio_features.size(); ++i) {
        audio_features[i] = 0.001f * static_cast<float>(i % 1000);
    }
    
    Input input;
    input.tokens = tokens.data();
    input.n_tokens = n_tokens;
    input.audio_features = audio_features.data();
    input.n_audio_frames = n_audio_frames;
    input.audio_start_pos = audio_start_pos;
    
    printf("\n--- Running decode ---\n");
    Output output;
    ErrorInfo error;
    
    if (!decode(state, input, output, &error)) {
        fprintf(stderr, "ERROR: decode failed: %s\n", error.message.c_str());
        free(state);
        return 1;
    }
    
    printf("Output: n_classes=%d, logits_size=%zu, timestamp_indices_size=%zu\n",
           output.n_classes, output.logits.size(), output.timestamp_indices.size());
    
    if (output.n_classes != hp.classify_head_size) {
        fprintf(stderr, "ERROR: n_classes mismatch\n");
        free(state);
        return 1;
    }
    
    printf("\n--- Timestamp indices (first 20) ---\n");
    for (int i = 0; i < 20; ++i) {
        printf("  token[%d]: class=%d -> %.2fs\n", 
               i, output.timestamp_indices[i],
               output.timestamp_indices[i] * (hp.timestamp_segment_time_ms / 1000.0f));
    }
    
    printf("\n--- Timestamp indices statistics ---\n");
    int min_ts = output.timestamp_indices[0];
    int max_ts = output.timestamp_indices[0];
    for (int ts : output.timestamp_indices) {
        min_ts = std::min(min_ts, ts);
        max_ts = std::max(max_ts, ts);
    }
    printf("  min=%d, max=%d (range: %.2fs - %.2fs)\n",
           min_ts, max_ts,
           min_ts * (hp.timestamp_segment_time_ms / 1000.0f),
           max_ts * (hp.timestamp_segment_time_ms / 1000.0f));
    
    if (ref_path && ref_path[0] != '\0') {
        printf("\n--- Comparing with reference ---\n");
        std::vector<float> ref_data;
        if (load_ref_data(ref_path, ref_data)) {
            if (compare_float_arrays(output.logits, ref_data, 0.01f, true)) {
                printf("PASS: Output matches reference (tol=0.01)\n");
            } else {
                printf("WARN: Output differs from reference\n");
            }
        } else {
            printf("Note: Reference file not found, saving current output\n");
            save_ref_data(ref_path, output.logits);
        }
    }
    
    TimestampResult ts_result = convert_to_timestamps(output, hp.timestamp_segment_time_ms);
    printf("\n--- Converted timestamps ---\n");
    printf("n_words=%d\n", ts_result.n_words);
    
    printf("\n--- Cleanup ---\n");
    free(state);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}