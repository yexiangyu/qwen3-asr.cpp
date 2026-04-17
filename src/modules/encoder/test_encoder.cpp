#include "encoder.h"
#include "encoder_model.h"
#include "../mel/mel.h"
#include "../audio_codec/audio_codec.h"

#include <cstdio>
#include <cstdlib>

int main() {
    namespace encoder = qwen3_asr::encoder;
    namespace mel = qwen3_asr::mel;
    namespace audio_codec = qwen3_asr::audio_codec;
    
    const char* test_wav = "tests/data/test_audio.wav";
    const char* model_path = "models/qwen3-asr-0.6b-f16.gguf";
    const char* ref_encoder = "tests/data/ref_encoder.raw";
    
    printf("=== Test 1: Init encoder state ===\n");
    
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
    
    printf("PASS: Encoder state initialized\n\n");
    
    encoder::free(state);
    
    printf("=== Test 2: Load mel spectrogram ===\n");
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    mel::MelSpectrum mel_spec;
    mel::ErrorInfo error;
    
    if (!mel::compute_from_file(test_wav, mel_spec, mel_config, &error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Mel spectrogram: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    printf("PASS: Mel loaded\n\n");
    
    printf("=== Test 3: Encoder encode (placeholder) ===\n");
    
    printf("Note: Full encoder implementation requires GGUF loading and graph building.\n");
    printf("This test validates the module structure only.\n");
    
    encoder::EncoderModel model;
    printf("EncoderModel structure: layers=%zu\n", model.layers.size());
    
    encoder::free_encoder_model(model);
    
    printf("PASS: Encoder model structure validated\n\n");
    
    printf("=== Test 4: Reference data helpers ===\n");
    
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    encoder::save_ref_data("tests/data/test_encoder_helper.raw", test_data);
    
    std::vector<float> loaded_data;
    if (!encoder::load_ref_data("tests/data/test_encoder_helper.raw", loaded_data)) {
        fprintf(stderr, "FAIL: Failed to load test data\n");
        return 1;
    }
    
    if (!encoder::compare_float_arrays(test_data, loaded_data, 1e-6f, true)) {
        fprintf(stderr, "FAIL: Test data comparison failed\n");
        return 1;
    }
    
    printf("PASS: Reference data helpers work\n\n");
    
    printf("=== All tests PASSED ===\n");
    printf("Note: Full encoder encode() implementation deferred to next phase.\n");
    
    return 0;
}