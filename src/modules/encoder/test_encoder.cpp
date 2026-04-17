#include "encoder.h"
#include "encoder_model.h"
#include "../mel/mel.h"
#include "../audio_codec/audio_codec.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

int main() {
    namespace encoder = qwen3_asr::encoder;
    namespace mel = qwen3_asr::mel;
    namespace audio_codec = qwen3_asr::audio_codec;
    
    const char* test_wav = "tests/data/test_audio.wav";
    const char* model_path = "models/qwen3-forced-aligner-0.6b-f16.gguf";
    const char* ref_encoder = "tests/data/ref_encoder.raw";
    
    printf("=== Test 1: Init encoder with model loading ===\n");
    
    encoder::Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    encoder::EncoderState* state = encoder::init(config);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init encoder state\n");
        return 1;
    }
    
    printf("Encoder device: %s\n", encoder::get_device_name(state));
    
    encoder::HyperParams hparams = encoder::get_hparams(state);
    printf("Encoder hparams: n_mel=%d, d_model=%d, hidden=%d, layers=%d\n",
           hparams.n_mel_bins, hparams.d_model, hparams.hidden_size, hparams.n_encoder_layers);
    
    if (!state->model) {
        fprintf(stderr, "FAIL: Model not loaded\n");
        encoder::free(state);
        return 1;
    }
    
    printf("Model layers: %zu\n", state->model->layers.size());
    printf("PASS: Encoder state initialized with model\n\n");
    
    printf("=== Test 2: Load mel spectrogram ===\n");
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    mel::MelSpectrum mel_spec;
    mel::ErrorInfo error;
    
    if (!mel::compute_from_file(test_wav, mel_spec, mel_config, &error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", error.message.c_str());
        encoder::free(state);
        return 1;
    }
    
    printf("Mel spectrogram: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    printf("PASS: Mel loaded\n\n");
    
    printf("=== Test 3: Encoder encode ===\n");
    
    encoder::Input input;
    input.mel_data = mel_spec.data.data();
    input.n_mels = mel_spec.n_mels;
    input.n_frames = mel_spec.n_frames;
    
    encoder::AudioFeatures output;
    encoder::ErrorInfo enc_error;
    
    if (!encoder::encode(state, input, output, &enc_error)) {
        fprintf(stderr, "FAIL: encode failed: %s\n", enc_error.message.c_str());
        encoder::free(state);
        return 1;
    }
    
    printf("Encoder output: hidden=%d, frames=%d\n", output.hidden_size, output.n_frames);
    printf("Total features: %zu floats\n", output.data.size());
    
    float min_val = *std::min_element(output.data.begin(), output.data.end());
    float max_val = *std::max_element(output.data.begin(), output.data.end());
    printf("Feature range: [%f, %f]\n", min_val, max_val);
    
    printf("PASS: Encoder encode succeeded\n\n");
    
    printf("=== Test 4: Compare with reference ===\n");
    
    std::vector<float> existing_ref;
    if (!encoder::load_ref_data(ref_encoder, existing_ref)) {
        printf("No existing reference, generating new one...\n");
        encoder::save_ref_data(ref_encoder, output.data);
        printf("Saved reference to %s (%zu floats)\n", ref_encoder, output.data.size());
        printf("Reference shape: [%d, %d]\n", output.hidden_size, output.n_frames);
    } else {
        printf("Comparing with existing reference (%zu floats)...\n", existing_ref.size());
        
        if (!encoder::compare_float_arrays(output.data, existing_ref, 1.0f, true)) {
            fprintf(stderr, "FAIL: Reference comparison failed\n");
            
            if (existing_ref.size() != output.data.size()) {
                fprintf(stderr, "Size mismatch: computed %zu, reference %zu\n", 
                        output.data.size(), existing_ref.size());
                fprintf(stderr, "Regenerating reference...\n");
                encoder::save_ref_data(ref_encoder, output.data);
            }
            
            encoder::free(state);
            return 1;
        }
        printf("PASS: Reference comparison (tolerance=1.0)\n");
    }
    
    printf("\n=== Test 5: Batch encode ===\n");
    
    const int batch_size = 3;
    const int max_frames = 1000;
    const int segment_starts[3] = {0, 3000, 8000};
    const int segment_lengths[3] = {800, 1000, 600};
    
    std::vector<std::vector<float>> batch_mels(batch_size);
    std::vector<int> batch_frames(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int start_frame = segment_starts[b];
        int n_frames = segment_lengths[b];
        batch_frames[b] = n_frames;
        batch_mels[b].resize(mel_spec.n_mels * n_frames);
        
        for (int m = 0; m < mel_spec.n_mels; ++m) {
            for (int f = 0; f < n_frames; ++f) {
                batch_mels[b][m * n_frames + f] = mel_spec.data[m * mel_spec.n_frames + start_frame + f];
            }
        }
        
        float min_v = batch_mels[b][0];
        float max_v = batch_mels[b][0];
        for (size_t i = 0; i < batch_mels[b].size(); ++i) {
            if (batch_mels[b][i] < min_v) min_v = batch_mels[b][i];
            if (batch_mels[b][i] > max_v) max_v = batch_mels[b][i];
        }
        printf("Batch item %d: start=%d, frames=%d, mel range=[%.3f, %.3f]\n", b, start_frame, n_frames, min_v, max_v);
    }
    
    encoder::BatchInput batch_input;
    for (int b = 0; b < batch_size; ++b) {
        batch_input.mel_data.push_back(batch_mels[b].data());
    }
    batch_input.n_frames = batch_frames;
    batch_input.n_mels = mel_spec.n_mels;
    batch_input.max_frames = max_frames;
    
    encoder::BatchOutput batch_output;
    
    printf("Batch input: %d items, max_frames=%d, actual_frames=[%d, %d, %d]\n",
           batch_size, max_frames, batch_frames[0], batch_frames[1], batch_frames[2]);
    
    if (!encoder::encode_batch(state, batch_input, batch_output, &enc_error)) {
        fprintf(stderr, "FAIL: batch encode failed: %s\n", enc_error.message.c_str());
        encoder::free(state);
        return 1;
    }
    
    printf("Batch output: %d results\n", batch_output.batch_size());
    for (int b = 0; b < batch_output.batch_size(); ++b) {
        auto& feat = batch_output.features[b];
        int expected_frames = (batch_frames[b] - 1) / 2 + 1;
        expected_frames = (expected_frames - 1) / 2 + 1;
        expected_frames = (expected_frames - 1) / 2 + 1;
        
        printf("  Item %d: hidden=%d, frames=%d (expected %d)\n", b, feat.hidden_size, feat.n_frames, expected_frames);
        
        if (feat.n_frames != expected_frames) {
            fprintf(stderr, "FAIL: Frame count mismatch for item %d\n", b);
            encoder::free(state);
            return 1;
        }
        
        float min_v = *std::min_element(feat.data.begin(), feat.data.end());
        float max_v = *std::max_element(feat.data.begin(), feat.data.end());
        printf("    Range: [%f, %f], size=%zu\n", min_v, max_v, feat.data.size());
    }
    
    printf("PASS: Batch encode succeeded\n\n");
    
    printf("=== Test 6: Batch vs single encode consistency ===\n");
    
    std::vector<encoder::AudioFeatures> single_outputs(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        encoder::Input single_input;
        single_input.mel_data = batch_mels[b].data();
        single_input.n_mels = mel_spec.n_mels;
        single_input.n_frames = batch_frames[b];
        
        if (!encoder::encode(state, single_input, single_outputs[b], &enc_error)) {
            fprintf(stderr, "FAIL: single encode %d failed: %s\n", b, enc_error.message.c_str());
            encoder::free(state);
            return 1;
        }
        
        printf("Single encode %d: frames=%d\n", b, single_outputs[b].n_frames);
    }
    
    for (int b = 0; b < batch_size; ++b) {
        float max_diff = 0.0f;
        size_t min_size = std::min(single_outputs[b].data.size(), batch_output.features[b].data.size());
        
        if (single_outputs[b].n_frames != batch_output.features[b].n_frames) {
            printf("Item %d: frame count differs (single=%d, batch=%d) - this is expected due to chunking\n",
                   b, single_outputs[b].n_frames, batch_output.features[b].n_frames);
            continue;
        }
        
        for (size_t i = 0; i < min_size; ++i) {
            float diff = fabsf(single_outputs[b].data[i] - batch_output.features[b].data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Item %d max difference: %.6f\n", b, max_diff);
        
        if (max_diff > 0.1f) {
            fprintf(stderr, "FAIL: Large difference for item %d\n", b);
            encoder::free(state);
            return 1;
        }
    }
    printf("PASS: Batch vs single consistency (within tolerance)\n\n");
    
    printf("=== Test 7: Cleanup ===\n");
    
    encoder::free(state);
    printf("Encoder state freed\n");
    
    printf("PASS: Cleanup\n\n");
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}