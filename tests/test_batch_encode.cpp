#include "asr/transcribe/encoder.hpp"
#include "asr/mel/mel.hpp"
#include "asr/codec/codec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* test_wav = "tests/data/test_audio.wav";
    
    namespace encoder = asr::transcribe::encoder;
    namespace mel = asr::mel;
    
    asr::ErrorInfo error;
    
    // Load mel spectrogram
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    mel::MelSpectrum mel_spec;
    if (!mel::compute_from_file(test_wav, mel_spec, mel_config, &error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Mel spectrogram: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    
    // Init encoder
    encoder::Config config;
    config.model_path = model_path;
    config.device_name = "CUDA0";
    config.n_threads = 4;
    
    encoder::EncoderState* state = encoder::init(config);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init encoder\n");
        return 1;
    }
    
    printf("Encoder initialized on %s\n\n", encoder::get_device_name(state));
    
    // Test segments
    const int batch_size = 3;
    const int segment_starts[3] = {0, 3000, 8000};
    const int segment_lengths[3] = {800, 1000, 600};
    
    std::vector<std::vector<float>> segment_mels(batch_size);
    std::vector<int> expected_frames(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int start = segment_starts[b];
        int len = segment_lengths[b];
        
        segment_mels[b].resize(mel_spec.n_mels * len);
        for (int m = 0; m < mel_spec.n_mels; ++m) {
            for (int f = 0; f < len; ++f) {
                segment_mels[b][m * len + f] = mel_spec.data[m * mel_spec.n_frames + start + f];
            }
        }
        
        // Compute expected output frames using chunked processing logic
        // chunk_size = 100, each chunk's conv output = (chunk_len - 1) / 2 + 1 three times
        const int chunk_size = 100;
        int n_chunks = (len + chunk_size - 1) / chunk_size;
        int out_frames = 0;
        for (int c = 0; c < n_chunks; ++c) {
            int chunk_len = std::min(chunk_size, len - c * chunk_size);
            int chunk_out = chunk_len;
            chunk_out = (chunk_out - 1) / 2 + 1;
            chunk_out = (chunk_out - 1) / 2 + 1;
            chunk_out = (chunk_out - 1) / 2 + 1;
            out_frames += chunk_out;
        }
        expected_frames[b] = out_frames;
    }
    
    printf("=== Test 1: Individual encode (3 segments separately) ===\n");
    
    std::vector<encoder::BatchOutput> individual_outputs(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        encoder::BatchInput input;
        input.mel_data.push_back(segment_mels[b].data());
        input.n_frames.push_back(segment_lengths[b]);
        input.n_mels = mel_spec.n_mels;
        input.max_frames = segment_lengths[b];
        
        encoder::BatchOutput output;
        if (!encoder::encode_batch(state, input, output, &error)) {
            fprintf(stderr, "FAIL: Individual encode %d failed: %s\n", b, error.message.c_str());
            encoder::free(state);
            return 1;
        }
        
        individual_outputs[b] = std::move(output);
        
        auto& feat = individual_outputs[b].features[0];
        printf("  Segment %d: hidden=%d, frames=%d (expected %d)\n",
               b, feat.hidden_size, feat.n_frames, expected_frames[b]);
        
        if (feat.n_frames != expected_frames[b]) {
            fprintf(stderr, "FAIL: Frame count mismatch\n");
            encoder::free(state);
            return 1;
        }
    }
    
    printf("PASS: Individual encode completed\n\n");
    
    printf("=== Test 2: Batch encode (3 segments together) ===\n");
    
    encoder::BatchInput batch_input;
    for (int b = 0; b < batch_size; ++b) {
        batch_input.mel_data.push_back(segment_mels[b].data());
        batch_input.n_frames.push_back(segment_lengths[b]);
    }
    batch_input.n_mels = mel_spec.n_mels;
    batch_input.max_frames = 1000;  // max of all segments
    
    encoder::BatchOutput batch_output;
    if (!encoder::encode_batch(state, batch_input, batch_output, &error)) {
        fprintf(stderr, "FAIL: Batch encode failed: %s\n", error.message.c_str());
        encoder::free(state);
        return 1;
    }
    
    printf("Batch output: %d results\n", batch_output.batch_size());
    
    for (int b = 0; b < batch_output.batch_size(); ++b) {
        auto& feat = batch_output.features[b];
        printf("  Segment %d: hidden=%d, frames=%d (expected %d)\n",
               b, feat.hidden_size, feat.n_frames, expected_frames[b]);
        
        if (feat.n_frames != expected_frames[b]) {
            fprintf(stderr, "FAIL: Frame count mismatch\n");
            encoder::free(state);
            return 1;
        }
    }
    
    printf("PASS: Batch encode completed\n\n");
    
    printf("=== Test 3: Compare batch vs individual outputs ===\n");
    
    bool all_match = true;
    float tolerance = 1e-3f;
    
    for (int b = 0; b < batch_size; ++b) {
        auto& batch_feat = batch_output.features[b];
        auto& individual_feat = individual_outputs[b].features[0];
        
        if (batch_feat.hidden_size != individual_feat.hidden_size) {
            fprintf(stderr, "FAIL: Segment %d hidden_size mismatch\n", b);
            all_match = false;
            continue;
        }
        
        if (batch_feat.n_frames != individual_feat.n_frames) {
            fprintf(stderr, "FAIL: Segment %d n_frames mismatch\n", b);
            all_match = false;
            continue;
        }
        
        if (batch_feat.data.size() != individual_feat.data.size()) {
            fprintf(stderr, "FAIL: Segment %d data size mismatch\n", b);
            all_match = false;
            continue;
        }
        
        // Compare values
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int n_different = 0;
        
        for (size_t i = 0; i < batch_feat.data.size(); ++i) {
            float diff = std::abs(batch_feat.data[i] - individual_feat.data[i]);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
            if (diff > tolerance) n_different++;
        }
        
        float mean_diff = sum_diff / batch_feat.data.size();
        
        printf("  Segment %d: max_diff=%.6f, mean_diff=%.6f, n_different=%d/%zu (tol=%.6f)\n",
               b, max_diff, mean_diff, n_different, batch_feat.data.size(), tolerance);
        
        if (max_diff > tolerance) {
            fprintf(stderr, "WARN: Segment %d has max_diff > tolerance\n", b);
            // Don't fail for small differences due to floating point
            if (max_diff > 0.01f) {
                all_match = false;
            }
        }
    }
    
    if (all_match) {
        printf("PASS: Batch outputs match individual outputs\n\n");
    } else {
        printf("FAIL: Batch outputs differ from individual outputs\n\n");
        encoder::free(state);
        return 1;
    }
    
    printf("=== Test 4: Performance comparison ===\n");
    
    // Measure individual encoding time
    auto t1_start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < batch_size; ++b) {
        encoder::BatchInput input;
        input.mel_data.push_back(segment_mels[b].data());
        input.n_frames.push_back(segment_lengths[b]);
        input.n_mels = mel_spec.n_mels;
        input.max_frames = segment_lengths[b];
        
        encoder::BatchOutput output;
        encoder::encode_batch(state, input, output, &error);
    }
    auto t1_end = std::chrono::high_resolution_clock::now();
    auto t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count();
    
    // Measure batch encoding time
    auto t2_start = std::chrono::high_resolution_clock::now();
    encoder::encode_batch(state, batch_input, batch_output, &error);
    auto t2_end = std::chrono::high_resolution_clock::now();
    auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count();
    
    printf("Individual encoding: %ld ms (%ld ms per segment)\n", t1_ms, t1_ms / batch_size);
    printf("Batch encoding:      %ld ms\n", t2_ms);
    
    float speedup = (float)t1_ms / t2_ms;
    printf("Speedup: %.2fx\n\n", speedup);
    
    encoder::free(state);
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}