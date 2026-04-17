#pragma once

#include "../common/types.h"
#include <string>

namespace qwen3_asr {
namespace audio_codec {

using modules::Result;
using modules::AudioSamples;
using modules::ErrorInfo;

struct Config {
    int target_sample_rate = 16000;
    bool normalize = true;
    float min_duration_sec = 0.5f;
};

struct Input {
    std::string path;
    std::string format_hint;
};

bool decode_file(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error = nullptr);

bool load_wav(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error = nullptr);

bool ffmpeg_decode(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error = nullptr);

void normalize_audio(std::vector<float>& samples);

void pad_audio(std::vector<float>& samples, float min_duration_sec, int sample_rate);

} // namespace audio_codec
} // namespace qwen3_asr