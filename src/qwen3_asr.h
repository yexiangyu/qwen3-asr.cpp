#pragma once

#include "mel_spectrogram.h"
#include "audio_encoder.h"
#include "text_decoder.h"
#include "audio_injection.h"

#include <string>
#include <vector>
#include <functional>

namespace qwen3_asr {

// Transcription parameters
struct transcribe_params {
    int32_t max_tokens = 1024;
    
    std::string language = "";
    
    std::string context = "";
    
    int32_t n_threads = 4;
    
    bool print_progress = false;
    
    bool print_timing = true;
    
    bool debug_input = false;
    
    bool debug_output = false;
};

// Transcription result
struct transcribe_result {
    std::string text;
    std::string text_prefix;
    std::string text_content;
    std::vector<int32_t> tokens;
    std::vector<float> token_confidences;
    std::vector<std::string> token_strings;
    bool success = false;
    std::string error_msg;
    
    int64_t t_load_ms = 0;
    int64_t t_mel_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;
};

struct batch_result {
    std::string text;
    std::string text_content;
    std::vector<int32_t> tokens;
    std::vector<float> token_confs;
    bool success = false;
    std::string error_msg;
};

// Progress callback type
using progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main ASR class that orchestrates the full pipeline
class Qwen3ASR {
public:
    Qwen3ASR();
    ~Qwen3ASR();
    
    // Load model from GGUF file
    // Returns true on success, false on failure (check get_error())
    bool load_model(const std::string & model_path, const std::string & device_name = "");
    
    // Transcribe audio file (WAV format, 16kHz mono)
    // Returns transcription result
    transcribe_result transcribe(const std::string & audio_path, 
                                  const transcribe_params & params = transcribe_params());
    
    // Transcribe raw audio samples
    // samples: audio samples normalized to [-1, 1]
    // n_samples: number of samples
    transcribe_result transcribe(const float * samples, int n_samples,
                                  const transcribe_params & params = transcribe_params());
    
    // Batch transcription for multiple audio inputs
    // audio_samples: vector of audio sample arrays (each normalized to [-1, 1])
    // n_samples: vector of sample counts for each audio
    // Returns vector of batch_result for each input
    std::vector<batch_result> transcribe_batch(
        const std::vector<const float *> & audio_samples,
        const std::vector<int> & n_samples,
        const transcribe_params & params = transcribe_params());
    
    // Set progress callback
    void set_progress_callback(progress_callback_t callback);
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if model is loaded
    bool is_loaded() const { return model_loaded_; }
    
    // Get model config
    const text_decoder_config & get_config() const { return decoder_.get_config(); }
    
private:
    // Internal transcription implementation
    transcribe_result transcribe_internal(const float * samples, int n_samples,
                                           const transcribe_params & params);
    
    // Build input token sequence for audio
    std::vector<int32_t> build_input_tokens(int32_t n_audio_frames,
                                             const std::string & context,
                                             const std::string & language);
    
    // Greedy decoding loop
    bool decode_greedy(const std::vector<int32_t> & input_tokens,
                       const std::vector<float> & audio_features,
                       int32_t n_audio_frames,
                       const transcribe_params & params,
                       std::vector<int32_t> & output_tokens,
                       std::vector<float> & output_confidences);
    
    // Sample next token with confidence (greedy: argmax + softmax)
    std::pair<int32_t, float> sample_greedy_with_conf(const float * logits, int32_t vocab_size);
    
    // Components
    AudioEncoder encoder_;
    TextDecoder decoder_;
    MelFilters mel_filters_;
    
    // State
    bool model_loaded_ = false;
    std::string error_msg_;
    progress_callback_t progress_callback_;
};

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate);

} // namespace qwen3_asr
