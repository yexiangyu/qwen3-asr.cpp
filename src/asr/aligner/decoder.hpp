#pragma once

#include "asr/common/types.hpp"
#include "asr/aligner/decoder_model.hpp"
#include <string>
#include <vector>

namespace asr::aligner::decoder {

using asr::ErrorInfo;

struct Config {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_ctx_length = 4096;
};

struct Input {
    const int32_t* tokens;
    int n_tokens;
    const float* audio_features;
    int n_audio_frames;
    int audio_start_pos;
};

struct Output {
    std::vector<int32_t> timestamp_indices;
    std::vector<float> logits;
    int n_classes;
};

struct TimestampResult {
    std::vector<float> timestamps;
    int n_words;
};

struct AlignedWord {
    std::string word;
    float start;
    float end;
};

struct AlignInput {
    const float* audio_features;
    int n_audio_frames;
    int audio_feature_dim;
    std::string text;
    std::string language;
};

struct AlignOutput {
    std::vector<AlignedWord> words;
    float audio_duration;
    bool success;
};

State* init(const Config& config);
void free(State* state);

bool decode(State* state, const Input& input, Output& output, ErrorInfo* error = nullptr);
TimestampResult convert_to_timestamps(const Output& output, int timestamp_segment_time_ms = 80);

bool align(State* state, const AlignInput& input, AlignOutput& output, ErrorInfo* error = nullptr);

std::vector<int32_t> tokenize(State* state, const std::string& text, std::vector<std::string>& words);
std::string decode_token(State* state, int32_t token_id);
std::vector<int32_t> build_token_sequence(State* state, int n_audio_frames, const std::vector<int32_t>& text_tokens);

void clear_kv_cache(State* state);
int get_kv_cache_used(State* state);
int get_kv_cache_capacity(State* state);

const char* get_device_name(State* state);
HyperParams get_hparams(State* state);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace asr::aligner::decoder
