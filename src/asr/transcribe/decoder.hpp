#pragma once

#include "asr/common/types.hpp"
#include <string>
#include <vector>

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
    int32_t next_token = -1;
};

struct TranscribeInput {
    const float* audio_features;
    int n_audio_frames;
    int audio_feature_dim;
    int max_tokens = 1024;
    
    std::string language;
    std::string context;
    std::string hotwords;
    std::string prompt;
};

struct TranscribeOutput {
    std::string language;
    std::string text;
    std::vector<int> tokens;
    int n_tokens;
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

std::string decode_token(const State* state, int token_id);
std::string decode_tokens(const State* state, const std::vector<int>& tokens);
std::vector<int> tokenize(const State* state, const std::string& text);

std::vector<int> build_token_sequence(
    const State* state,
    int n_audio_frames,
    const std::string& language,
    const std::string& context,
    const std::string& hotwords,
    const std::string& prompt);

bool transcribe(State* state, 
                const TranscribeInput& input, 
                TranscribeOutput& output, 
                ErrorInfo* error = nullptr);

bool load_ref_data(const char* path, std::vector<float>& data);
bool save_ref_data(const char* path, const std::vector<float>& data);
bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose = false);

struct BatchDecodeState;

struct BatchConfig {
    std::string model_path;
    std::string device_name;
    int n_threads = 4;
    int max_batch_size = 4;
    int max_total_capacity = 16384;
};

struct BatchSequenceInput {
    const float* audio_features;
    int n_audio_frames;
    int audio_feature_dim;
    int max_tokens = 256;
    std::string language;
    std::string context;
    std::string hotwords;
    std::string prompt;
};

struct BatchSequenceOutput {
    int slot_id;
    std::string language;
    std::string text;
    std::vector<int> tokens;
    int n_tokens;
    bool is_eos;
};

BatchDecodeState* init_batch(const Config& config, int max_batch_size, int max_total_capacity);
void free_batch(BatchDecodeState* state);

int batch_add_sequence(BatchDecodeState* state,
                       const std::vector<int>& prefill_tokens,
                       const float* audio_features,
                       int n_audio_frames,
                       int audio_feature_dim,
                       int audio_start_pos,
                       int max_tokens,
                       const std::string& language,
                       ErrorInfo* error);

void batch_remove_sequence(BatchDecodeState* state, int slot_id);

bool batch_decode_step(BatchDecodeState* state, ErrorInfo* error);

std::vector<int> batch_get_active_slots(BatchDecodeState* state);
std::vector<int> batch_get_eos_slots(BatchDecodeState* state);
std::string batch_get_text(BatchDecodeState* state, int slot_id);
std::vector<int> batch_get_tokens(BatchDecodeState* state, int slot_id);
int batch_get_seq_id(BatchDecodeState* state, int slot_id);

int batch_get_n_active(BatchDecodeState* state);
int batch_get_capacity(BatchDecodeState* state);
int batch_get_slot_capacity(BatchDecodeState* state, int slot_id);
int batch_get_slot_used(BatchDecodeState* state, int slot_id);

void batch_clear(BatchDecodeState* state);

HyperParams batch_get_hparams(BatchDecodeState* state);
const char* batch_get_device_name(BatchDecodeState* state);

std::vector<int> batch_build_token_sequence(
    BatchDecodeState* state,
    int n_audio_frames,
    const std::string& language,
    const std::string& context,
    const std::string& hotwords,
    const std::string& prompt);

} // namespace asr::transcribe::decoder
