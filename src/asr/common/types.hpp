#pragma once

#include <vector>
#include <string>

namespace asr {

constexpr int DEFAULT_SAMPLE_RATE = 16000;
constexpr int DEFAULT_N_MELS = 128;
constexpr int DEFAULT_N_FFT = 400;
constexpr int DEFAULT_HOP_LENGTH = 160;

template<typename T, typename E>
struct Result {
    T value;
    E error;
    bool success;
    
    Result(T v) : value(std::move(v)), success(true) {}
    Result(E e) : error(std::move(e)), success(false) {}
    
    bool ok() const { return success; }
    const T& get() const { return value; }
    const E& get_error() const { return error; }
};

struct AudioSamples {
    std::vector<float> data;
    int sample_rate;
    int n_samples;
    float duration_sec;
};

struct MelSpectrum {
    std::vector<float> data;
    int n_mels;
    int n_frames;
};

struct AudioFeatures {
    std::vector<float> data;
    int hidden_size;
    int n_frames;
};

struct ErrorInfo {
    std::string message;
    int code = 0;
};

} // namespace asr
