#include "asr/codec/codec.h"
#include "asr/common/types.h"

#include <fstream>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace qwen3_asr {
namespace asr { namespace codec {

using asr::ErrorInfo;

bool load_wav(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        if (error) error->message = "Cannot open WAV file: " + std::string(path);
        return false;
    }

    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        if (error) error->message = "Not a valid WAV file (missing RIFF header)";
        return false;
    }

    uint32_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), 4);

    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        if (error) error->message = "Not a valid WAV file (missing WAVE header)";
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sr), 4);
            uint32_t byte_rate;
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align;
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            if (audio_format != 1) {
                if (error) error->message = "Only PCM format supported (got format " + std::to_string(audio_format) + ")";
                return false;
            }
            if (bits_per_sample != 16) {
                if (error) error->message = "Only 16-bit samples supported (got " + std::to_string(bits_per_sample) + " bits)";
                return false;
            }

            sample_rate = static_cast<int>(sr);
            int num_samples = chunk_size / (bits_per_sample / 8) / num_channels;
            samples.resize(num_samples);

            std::vector<int16_t> raw_samples(num_samples * num_channels);
            file.read(reinterpret_cast<char*>(raw_samples.data()), chunk_size);

            for (int i = 0; i < num_samples; i++) {
                if (num_channels == 1) {
                    samples[i] = raw_samples[i] / 32768.0f;
                } else {
                    float sum = 0;
                    for (int c = 0; c < num_channels; c++) {
                        sum += raw_samples[i * num_channels + c];
                    }
                    samples[i] = (sum / num_channels) / 32768.0f;
                }
            }
            return true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (error) error->message = "No data chunk found in WAV file";
    return false;
}

void normalize_audio(std::vector<float>& samples) {
    if (samples.empty()) return;
    
    float peak = 0.0f;
    for (float v : samples) {
        peak = std::max(peak, std::fabs(v));
    }
    
    if (peak == 0.0f) return;
    
    if (peak > 1.0f) {
        for (float& v : samples) {
            v = v / peak;
        }
    }
    
    for (float& v : samples) {
        v = std::max(-1.0f, std::min(1.0f, v));
    }
}

void pad_audio(std::vector<float>& samples, float min_duration_sec, int sample_rate) {
    int min_len = static_cast<int>(min_duration_sec * sample_rate);
    if (static_cast<int>(samples.size()) < min_len) {
        int pad = min_len - static_cast<int>(samples.size());
        samples.resize(samples.size() + pad, 0.0f);
    }
}

} // namespace codec
} // namespace asr
} // namespace qwen3_asr