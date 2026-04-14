#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace qwen3_asr {

constexpr int QWEN_SAMPLE_RATE = 16000;
constexpr float MAX_ASR_INPUT_SECONDS = 1200.0f;
constexpr float MAX_FORCE_ALIGN_INPUT_SECONDS = 180.0f;
constexpr float MIN_ASR_INPUT_SECONDS = 0.5f;

struct AudioChunk {
    int orig_index;
    int chunk_index;
    std::vector<float> wav;
    int sr;
    float offset_sec;
};

inline void float_range_normalize(std::vector<float> & audio) {
    if (audio.empty()) return;
    
    float peak = 0.0f;
    for (float v : audio) {
        peak = std::max(peak, std::fabs(v));
    }
    
    if (peak == 0.0f) return;
    
    if (peak > 1.0f) {
        for (float & v : audio) {
            v = v / peak;
        }
    }
    
    for (float & v : audio) {
        v = std::clamp(v, -1.0f, 1.0f);
    }
}

inline std::vector<float> normalize_audio_input(const float * samples, int n_samples, int input_sr) {
    std::vector<float> audio(samples, samples + n_samples);
    
    if (input_sr != QWEN_SAMPLE_RATE) {
        int target_len = static_cast<int>(static_cast<float>(n_samples) * QWEN_SAMPLE_RATE / input_sr);
        std::vector<float> resampled(target_len);
        
        float ratio = static_cast<float>(n_samples) / target_len;
        for (int i = 0; i < target_len; ++i) {
            float src_idx = i * ratio;
            int idx0 = static_cast<int>(src_idx);
            int idx1 = std::min(idx0 + 1, n_samples - 1);
            float frac = src_idx - idx0;
            resampled[i] = audio[idx0] * (1.0f - frac) + audio[idx1] * frac;
        }
        audio = std::move(resampled);
    }
    
    float_range_normalize(audio);
    
    return audio;
}

inline void pad_audio_to_min_length(std::vector<float> & audio, float min_seconds) {
    int min_len = static_cast<int>(min_seconds * QWEN_SAMPLE_RATE);
    if (static_cast<int>(audio.size()) < min_len) {
        int pad = min_len - static_cast<int>(audio.size());
        audio.resize(audio.size() + static_cast<size_t>(pad), 0.0f);
    }
}

std::vector<AudioChunk> split_audio_into_chunks(
    const std::vector<float> & wav,
    float max_chunk_sec,
    float search_expand_sec = 5.0f,
    float min_window_ms = 100.0f);

std::string detect_and_fix_repetitions(const std::string & text, int threshold = 20);

std::pair<std::string, std::string> parse_asr_output(
    const std::string & raw,
    const std::string & user_language = "");

std::string normalize_language_name(const std::string & language);

}