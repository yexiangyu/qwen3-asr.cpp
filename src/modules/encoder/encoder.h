#pragma once

#include "../common/types.h"
#include "encoder_model.h"
#include <string>

namespace qwen3_asr {
namespace encoder {

using modules::AudioFeatures;
using modules::ErrorInfo;

struct Config {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_ctx_length = 4096;
};

struct Input {
    const float* mel_data;
    int n_mels;
    int n_frames;
};

struct BatchInput {
    std::vector<const float*> mel_data;
    std::vector<int> n_frames;
    int n_mels;
    int max_frames;
    
    int batch_size() const { return mel_data.size(); }
};

struct BatchOutput {
    std::vector<AudioFeatures> features;
    
    int batch_size() const { return features.size(); }
};

EncoderState* init(const Config& config);
void free(EncoderState* state);

bool encode(EncoderState* state, const Input& input, AudioFeatures& output, ErrorInfo* error = nullptr);
bool encode_batch(EncoderState* state, const BatchInput& input, BatchOutput& output, ErrorInfo* error = nullptr);

const char* get_device_name(EncoderState* state);
HyperParams get_hparams(EncoderState* state);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace encoder
} // namespace qwen3_asr