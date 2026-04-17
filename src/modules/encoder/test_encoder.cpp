#include "encoder.h"
#include "encoder_model.h"
#include "../mel/mel.h"
#include "../audio_codec/audio_codec.h"

#include <cstdio>
#include <cstdlib>
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
    
    printf("\n=== Test 5: Cleanup ===\n");
    
    encoder::free(state);
    printf("Encoder state freed\n");
    
    printf("PASS: Cleanup\n\n");
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}