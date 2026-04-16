#pragma once

#include "gguf_loader.h"

#include <vector>

namespace qwen3_asr {

struct audio_encoder_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    struct ggml_tensor * embd_conv = nullptr;
    struct ggml_tensor * embd_enc = nullptr;
};

struct BatchMelInput {
    std::vector<const float*> mels;
    std::vector<int> mel_lengths;
    int batch_size;
    
    BatchMelInput() : batch_size(0) {}
};

struct BatchEncoderOutput {
    std::vector<std::vector<float>> features;
    std::vector<int> feature_lengths;
    std::vector<bool> success;
    std::vector<std::string> errors;
};

class AudioEncoder {
public:
    AudioEncoder();
    ~AudioEncoder();
    
    bool load_model(const std::string & model_path, const std::string & device_name = "");
    
    bool encode(const float * mel_data, int n_mel, int n_frames, 
                std::vector<float> & output);
    
    bool encode_conv_only(const float * mel_data, int n_mel, int n_frames,
                          std::vector<float> & output);
    
    bool encode_no_chunk(const float * mel_data, int n_mel, int n_frames,
                         std::vector<float> & output);
    
    bool encode_batch(const BatchMelInput& input, BatchEncoderOutput& output);
    
    const audio_encoder_hparams & get_hparams() const { return model_.hparams; }
    const text_decoder_hparams & get_text_hparams() const { return model_.text_hparams; }
    
    const std::string & get_error() const { return error_msg_; }
    
private:
    struct ggml_cgraph * build_graph_conv(int n_frames);
    struct ggml_cgraph * build_graph_encoder(int n_ctx);
    struct ggml_cgraph * build_graph_conv_batch(int max_frames, int batch_size);
    struct ggml_cgraph * build_graph_encoder_batch(int n_ctx, int batch_size);
    
    bool compute_graph(struct ggml_cgraph * graph);
    
    audio_encoder_model model_;
    audio_encoder_state state_;
    std::string error_msg_;
    
    int n_threads_ = 4;
};

}
