#pragma once

#include "asr/common/types.hpp"
#include <string>
#include <vector>

namespace qwen3_asr {
namespace asr::transcribe::decoder {

using asr::ErrorInfo;

struct Config {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_ctx_length = 4096;
};

struct HyperParams {
    int vocab_size = 151936;
    int hidden_size = 1024;
    int n_layers = 28;
    int n_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 128;
    int intermediate_size = 3072;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    int audio_start_token = 151669;
    int audio_end_token = 151670;
    int audio_pad_token = 151676;
    int pad_token = 151643;
    int eos_token = 151645;
};

struct State;

struct PrefillInput {
    const int* tokens;
    int n_tokens;
    const float* audio_features;
    int n_audio_frames;
    int audio_feature_dim;
    int audio_start_pos;
};

struct DecodeInput {
    const int* tokens;
    int n_tokens;
    int n_past;
};

struct DecoderOutput {
    std::vector<float> logits;
    int vocab_size;
};

State* init(const Config& config);
void free(State* state);

bool prefill(State* state, const PrefillInput& input, DecoderOutput& output, ErrorInfo* error = nullptr);
bool decode(State* state, const DecodeInput& input, DecoderOutput& output, ErrorInfo* error = nullptr);

void clear_kv_cache(State* state);
int get_kv_cache_used(State* state);
int get_kv_cache_capacity(State* state);

const char* get_device_name(State* state);
HyperParams get_hparams(State* state);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace asr::transcribe::decoder
} // namespace qwen3_asr