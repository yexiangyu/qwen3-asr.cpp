#pragma once

#include "asr/common/types.hpp"
#include <vector>
#include <string>

namespace qwen3_asr {
namespace asr::mel {

using asr::Result;
using asr::MelSpectrum;
using asr::ErrorInfo;

struct Config {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 128;
    float fmin = 0.0f;
    float fmax = 8000.0f;
    int n_threads = 4;
};

struct FilterBank {
    std::vector<float> data;
    int n_mels;
    int n_fft_bins;
};

struct Input {
    const float* samples;
    int n_samples;
};

struct Window {
    std::vector<double> data;
    int length;
};

bool compute(const Input& input, MelSpectrum& output, const Config& config = {}, ErrorInfo* error = nullptr);

bool compute_from_file(const char* wav_path, MelSpectrum& output, const Config& config = {}, ErrorInfo* error = nullptr);

bool create_filter_bank(FilterBank& filters, const Config& config = {});

bool create_hann_window(Window& window, int length, bool periodic = true);

bool load_ref_data(const char* path, std::vector<float>& data);

bool save_ref_data(const char* path, const std::vector<float>& data);

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace asr::mel
} // namespace qwen3_asr