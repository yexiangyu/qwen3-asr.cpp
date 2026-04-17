#pragma once

#include "../common/types.h"
#include "align_encoder_model.h"
#include <string>

namespace qwen3_asr {
namespace align_encoder {

using modules::AudioFeatures;
using modules::ErrorInfo;

struct Config {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_ctx_length = 4096;
};

struct AlignBatchInput {
    std::vector<const float*> mel_data;
    std::vector<int> n_frames;
    int n_mels;
    int max_frames;
    
    int batch_size() const { return mel_data.size(); }
};

struct AlignBatchOutput {
    std::vector<AudioFeatures> features;
    
    int batch_size() const { return features.size(); }
};

AlignEncoderState* init(const Config& config);
void free(AlignEncoderState* state);

bool encode_batch(AlignEncoderState* state, const AlignBatchInput& input, AlignBatchOutput& output, ErrorInfo* error = nullptr);

const char* get_device_name(AlignEncoderState* state);
HyperParams get_hparams(AlignEncoderState* state);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

} // namespace align_encoder
} // namespace qwen3_asr