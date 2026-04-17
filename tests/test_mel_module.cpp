#include "asr/mel/mel.hpp"
#include "asr/codec/codec.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

int main() {
    namespace mel = qwen3_asr::asr::mel;
    namespace codec = qwen3_asr::asr::codec;
    
    const char* test_wav = "tests/data/test_audio.wav";
    const char* ref_mel_path = "tests/data/ref_mel.raw";
    
    printf("=== Test 1: Load audio ===\n");
    
    std::vector<float> samples;
    int sample_rate;
    mel::ErrorInfo error;
    
    if (!codec::decode_file(test_wav, samples, sample_rate, &error)) {
        fprintf(stderr, "FAIL: Failed to load audio: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Loaded %d samples at %d Hz (%.2f seconds)\n", 
           (int)samples.size(), sample_rate, samples.size() / 16000.0f);
    
    printf("PASS: Audio loaded\n\n");
    
    printf("=== Test 2: Compute mel spectrogram ===\n");
    
    mel::Config config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.n_mels = 128;
    config.n_threads = 4;
    
    mel::Input input;
    input.samples = samples.data();
    input.n_samples = static_cast<int>(samples.size());
    
    mel::MelSpectrum mel_spec;
    if (!mel::compute(input, mel_spec, config, &error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Mel spectrogram: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    printf("Expected frames: %d (based on %d samples)\n", 
           static_cast<int>((samples.size() + 400) / 160), (int)samples.size());
    
    float min_val = *std::min_element(mel_spec.data.begin(), mel_spec.data.end());
    float max_val = *std::max_element(mel_spec.data.begin(), mel_spec.data.end());
    printf("Mel range: [%f, %f]\n", min_val, max_val);
    
    printf("PASS: Mel spectrogram computed\n\n");
    
    printf("=== Test 3: Compare with reference ===\n");
    
    std::vector<float> existing_ref;
    if (!mel::load_ref_data(ref_mel_path, existing_ref)) {
        printf("No existing reference, generating new one...\n");
        mel::save_ref_data(ref_mel_path, mel_spec.data);
        printf("Saved reference to %s (%zu floats)\n", ref_mel_path, mel_spec.data.size());
        printf("Reference shape: [%d, %d]\n", mel_spec.n_mels, mel_spec.n_frames);
    } else {
        printf("Comparing with existing reference (%zu floats)...\n", existing_ref.size());
        
        if (!mel::compare_float_arrays(mel_spec.data, existing_ref, 0.1f, true)) {
            fprintf(stderr, "FAIL: Reference comparison failed\n");
            
            if (existing_ref.size() != mel_spec.data.size()) {
                fprintf(stderr, "Size mismatch: computed %zu, reference %zu\n", 
                        mel_spec.data.size(), existing_ref.size());
                fprintf(stderr, "Regenerating reference...\n");
                mel::save_ref_data(ref_mel_path, mel_spec.data);
            }
            
            return 1;
        }
        printf("PASS: Reference comparison (tolerance=0.1)\n");
    }
    
    printf("\n=== Test 4: Filterbank ===\n");
    
    mel::FilterBank filters;
    if (!mel::create_filter_bank(filters, config)) {
        fprintf(stderr, "FAIL: Failed to create filter bank\n");
        return 1;
    }
    
    printf("Filter bank: %d mels, %d FFT bins\n", filters.n_mels, filters.n_fft_bins);
    
    float filter_sum = 0.0f;
    for (float v : filters.data) filter_sum += v;
    printf("Filter bank sum: %f\n", filter_sum);
    
    if (filters.n_mels != 128 || filters.n_fft_bins != 201) {
        fprintf(stderr, "FAIL: Filter bank dimensions wrong\n");
        return 1;
    }
    
    printf("PASS: Filter bank\n\n");
    
    printf("=== Test 5: compute_from_file ===\n");
    
    mel::MelSpectrum mel2;
    if (!mel::compute_from_file(test_wav, mel2, config, &error)) {
        fprintf(stderr, "FAIL: compute_from_file failed: %s\n", error.message.c_str());
        return 1;
    }
    
    if (!mel::compare_float_arrays(mel_spec.data, mel2.data, 1e-5f, true)) {
        fprintf(stderr, "FAIL: compute != compute_from_file\n");
        return 1;
    }
    
    printf("PASS: compute_from_file matches compute\n\n");
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}