#pragma once

#include "../common/types.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <map>

namespace qwen3_asr {
namespace align_encoder {

struct HyperParams {
    int n_mel_bins = 128;
    int d_model = 1024;
    int hidden_size = 1024;
    int n_encoder_layers = 24;
    int n_attention_heads = 16;
    int head_dim = 64;
    int ff_dim = 4096;
    int conv_channels = 480;
};

struct AlignEncoderLayer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
};

struct AlignEncoderModel {
    HyperParams hparams;
    
    ggml_tensor* conv2d1_w = nullptr;
    ggml_tensor* conv2d1_b = nullptr;
    ggml_tensor* conv2d2_w = nullptr;
    ggml_tensor* conv2d2_b = nullptr;
    ggml_tensor* conv2d3_w = nullptr;
    ggml_tensor* conv2d3_b = nullptr;
    
    ggml_tensor* conv_out_w = nullptr;
    
    ggml_tensor* ln_post_w = nullptr;
    ggml_tensor* ln_post_b = nullptr;
    ggml_tensor* proj1_w = nullptr;
    ggml_tensor* proj1_b = nullptr;
    ggml_tensor* proj2_w = nullptr;
    ggml_tensor* proj2_b = nullptr;
    
    std::vector<AlignEncoderLayer> layers;
    
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    void* mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    std::map<std::string, ggml_tensor*> tensors;
};

struct AlignEncoderState {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    ggml_tensor* embd_conv = nullptr;
    ggml_tensor* embd_enc = nullptr;
    
    AlignEncoderModel* model = nullptr;
};

bool load_model(const char* path, AlignEncoderModel& model, modules::ErrorInfo* error);

void free_align_encoder_model(AlignEncoderModel& model);

} // namespace align_encoder
} // namespace qwen3_asr