#pragma once

#include "asr/common/types.hpp"
#include "asr/aligner/encoder_model.hpp"
#include <string>

namespace qwen3_asr {
namespace asr::aligner::encoder {

using asr::AudioFeatures;
using asr::ErrorInfo;

struct Config {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_ctx_length = 4096;
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

bool encode_batch(EncoderState* state, const BatchInput& input, BatchOutput& output, ErrorInfo* error = nullptr);

const char* get_device_name(EncoderState* state);
HyperParams get_hparams(EncoderState* state);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace asr::aligner::encoder
} // namespace qwen3_asr