#include "asr/codec/codec.h"
#include "asr/common/types.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace qwen3_asr {
namespace asr::codec {

bool load_ref_data(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return true;
}

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
    if (a.size() != b.size()) {
        fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return false;
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    if (max_diff > tolerance) {
        fprintf(stderr, "Max diff exceeds tolerance: %.6f > %.6f\n", max_diff, tolerance);
        return false;
    }
    
    return true;
}

void save_ref_data(const char* path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

} // namespace asr::codec
} // namespace qwen3_asr

int main(int argc, char** argv) {
    using namespace qwen3_asr::asr::codec;
    using qwen3_asr::asr::ErrorInfo;
    
    const char* test_wav = "tests/data/test_audio.wav";
    const char* ref_samples = "tests/data/ref_audio_samples.raw";
    
    printf("=== Test 1: WAV Loader ===\n");
    
    std::vector<float> samples;
    int sample_rate;
    ErrorInfo error;
    
    if (!load_wav(test_wav, samples, sample_rate, &error)) {
        fprintf(stderr, "FAIL: load_wav failed: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("Loaded %d samples at %d Hz\n", (int)samples.size(), sample_rate);
    
    if (sample_rate != 16000) {
        fprintf(stderr, "FAIL: Unexpected sample rate: %d (expected 16000)\n", sample_rate);
        return 1;
    }
    
    float min_val = *std::min_element(samples.begin(), samples.end());
    float max_val = *std::max_element(samples.begin(), samples.end());
    printf("Sample range: [%f, %f]\n", min_val, max_val);
    
    if (min_val < -1.0f || max_val > 1.0f) {
        fprintf(stderr, "FAIL: Samples out of range [-1, 1]\n");
        return 1;
    }
    
    printf("PASS: WAV loader basic test\n\n");
    
    printf("=== Test 2: Generate Reference ===\n");
    
    std::vector<float> existing_ref;
    if (!load_ref_data(ref_samples, existing_ref)) {
        printf("No existing reference found, generating new one...\n");
        save_ref_data(ref_samples, samples);
        printf("Saved reference to %s (%zu floats)\n", ref_samples, samples.size());
    } else {
        printf("Comparing with existing reference...\n");
        if (!compare_float_arrays(samples, existing_ref, 1e-6f)) {
            fprintf(stderr, "FAIL: Reference comparison failed\n");
            return 1;
        }
        printf("PASS: Reference comparison\n");
    }
    
    printf("\n=== Test 3: decode_file ===\n");
    
    std::vector<float> samples2;
    if (!decode_file(test_wav, samples2, sample_rate, &error)) {
        fprintf(stderr, "FAIL: decode_file failed: %s\n", error.message.c_str());
        return 1;
    }
    
    printf("decode_file loaded %d samples\n", (int)samples2.size());
    
    normalize_audio(samples2);
    
    std::vector<float> samples1_normalized = samples;
    normalize_audio(samples1_normalized);
    
    if (!compare_float_arrays(samples1_normalized, samples2, 1e-6f)) {
        fprintf(stderr, "FAIL: decode_file != load_wav after normalization\n");
        return 1;
    }
    
    printf("PASS: decode_file matches load_wav\n\n");
    
    printf("=== Test 4: pad_audio ===\n");
    
    std::vector<float> short_samples = {0.1f, 0.2f, 0.3f};
    pad_audio(short_samples, 0.5f, 16000);
    
    int expected_len = static_cast<int>(0.5f * 16000);
    if ((int)short_samples.size() != expected_len) {
        fprintf(stderr, "FAIL: pad_audio size mismatch: %zu (expected %d)\n", 
                short_samples.size(), expected_len);
        return 1;
    }
    
    printf("pad_audio: %zu samples (expected %d)\n", short_samples.size(), expected_len);
    printf("PASS: pad_audio\n\n");
    
    printf("=== All tests PASSED ===\n");
    
    return 0;
}