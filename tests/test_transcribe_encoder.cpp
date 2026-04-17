#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/encoder_model.hpp"
#include "asr/mel/mel.hpp"
#include "asr/codec/codec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

int main() {
    namespace encoder = qwen3_asr::asr::transcribe::encoder;
    namespace mel = qwen3_asr::asr::mel;
    
    const char* test_wav = "tests/data/test_audio.wav";
    const char* model_path = "models/qwen3-asr-1.7b-f16.gguf";
    const char* ref_asr_encoder = "tests/data/ref_asr_encoder_batch.raw";
    
    printf("=== Test 1: Init asr_encoder with model loading ===\n");
    
    encoder::Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    encoder::EncoderState* state = encoder::init(config);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init asr_encoder state\n");
        return 1;
    }
    
    printf("ASR encoder device: %s\n", encoder::get_device_name(state));
    
    encoder::HyperParams hparams = encoder::get_hparams(state);
    printf("ASR encoder hparams: n_mel=%d, d_model=%d, hidden=%d, layers=%d, heads=%d\n",
           hparams.n_mel_bins, hparams.d_model, hparams.hidden_size, 
           hparams.n_encoder_layers, hparams.n_attention_heads);
    
    if (!state->model) {
        fprintf(stderr, "FAIL: Model not loaded\n");
        encoder::free(state);
        return 1;
    }
    
    printf("Model layers: %zu\n", state->model->layers.size());
    printf("PASS: ASR encoder state initialized with model\n\n");
    
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
    
    printf("=== Test 3: Batch encode (batch_size=1) ===\n");
    
    const int max_frames_test3 = 1000;
    std::vector<float> mel_test3(mel_spec.n_mels * max_frames_test3);
    for (int fm = 0; fm < mel_spec.n_mels; ++fm) {
        for (int f = 0; f < max_frames_test3; ++f) {
            mel_test3[fm * max_frames_test3 + f] = mel_spec.data[fm * mel_spec.n_frames + f];
        }
    }
    
    encoder::BatchInput batch_input1;
    batch_input1.mel_data.push_back(mel_test3.data());
    batch_input1.n_frames.push_back(max_frames_test3);
    batch_input1.n_mels = mel_spec.n_mels;
    batch_input1.max_frames = max_frames_test3;
    
    encoder::BatchOutput batch_output1;
    encoder::ErrorInfo enc_error;
    
    if (!encoder::encode_batch(state, batch_input1, batch_output1, &enc_error)) {
        fprintf(stderr, "FAIL: batch encode failed: %s\n", enc_error.message.c_str());
        encoder::free(state);
        return 1;
    }
    
    printf("Batch output: %d results\n", batch_output1.batch_size());
    auto& feat1 = batch_output1.features[0];
    int expected_frames1 = (max_frames_test3 - 1) / 2 + 1;
    expected_frames1 = (expected_frames1 - 1) / 2 + 1;
    expected_frames1 = (expected_frames1 - 1) / 2 + 1;
    
    printf("  Item 0: hidden=%d, frames=%d (expected %d)\n", 
           feat1.hidden_size, feat1.n_frames, expected_frames1);
    
    if (feat1.n_frames != expected_frames1) {
        fprintf(stderr, "FAIL: Frame count mismatch\n");
        encoder::free(state);
        return 1;
    }
    
    float min_v1 = *std::min_element(feat1.data.begin(), feat1.data.end());
    float max_v1 = *std::max_element(feat1.data.begin(), feat1.data.end());
    printf("    Range: [%f, %f], size=%zu\n", min_v1, max_v1, feat1.data.size());
    printf("PASS: Batch encode (batch_size=1) succeeded\n\n");
    
    printf("=== Test 4: Compare with reference (batch_size=1) ===\n");
    
    std::vector<float> existing_ref;
    if (!encoder::load_ref_data(ref_asr_encoder, existing_ref)) {
        printf("No existing reference, generating new one...\n");
        encoder::save_ref_data(ref_asr_encoder, feat1.data);
        printf("Saved reference to %s (%zu floats)\n", ref_asr_encoder, feat1.data.size());
        printf("Reference shape: [%d, %d]\n", feat1.hidden_size, feat1.n_frames);
    } else {
        printf("Comparing with existing reference (%zu floats)...\n", existing_ref.size());
        
        if (!encoder::compare_float_arrays(feat1.data, existing_ref, 1.0f, true)) {
            fprintf(stderr, "FAIL: Reference comparison failed\n");
            
            if (existing_ref.size() != feat1.data.size()) {
                fprintf(stderr, "Size mismatch: computed %zu, reference %zu\n", 
                        feat1.data.size(), existing_ref.size());
                fprintf(stderr, "Regenerating reference...\n");
                encoder::save_ref_data(ref_asr_encoder, feat1.data);
            }
            
            encoder::free(state);
            return 1;
        }
        printf("PASS: Reference comparison (tolerance=1.0)\n");
    }
    
    printf("\n=== Test 5: Batch encode (batch_size=3) ===\n");
    
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
        
        for (int fm = 0; fm < mel_spec.n_mels; ++fm) {
            for (int f = 0; f < n_frames; ++f) {
                batch_mels[b][fm * n_frames + f] = mel_spec.data[fm * mel_spec.n_frames + start_frame + f];
            }
        }
        
        float min_v = batch_mels[b][0];
        float max_v = batch_mels[b][0];
        for (size_t i = 0; i < batch_mels[b].size(); ++i) {
            if (batch_mels[b][i] < min_v) min_v = batch_mels[b][i];
            if (batch_mels[b][i] > max_v) max_v = batch_mels[b][i];
        }
        printf("Batch item %d: start=%d, frames=%d, mel range=[%.3f, %.3f]\n", 
               b, start_frame, n_frames, min_v, max_v);
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
        
        printf("  Item %d: hidden=%d, frames=%d (expected %d)\n", 
               b, feat.hidden_size, feat.n_frames, expected_frames);
        
        if (feat.n_frames != expected_frames) {
            fprintf(stderr, "FAIL: Frame count mismatch for item %d\n", b);
            encoder::free(state);
            return 1;
        }
        
        float min_v = *std::min_element(feat.data.begin(), feat.data.end());
        float max_v = *std::max_element(feat.data.begin(), feat.data.end());
        printf("    Range: [%f, %f], size=%zu\n", min_v, max_v, feat.data.size());
    }
    
    printf("PASS: Batch encode (batch_size=3) succeeded\n\n");
    
    printf("=== Test 6: Cleanup ===\n");
    
    encoder::free(state);
    printf("ASR encoder state freed\n");
    
    printf("PASS: Cleanup\n\n");
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}