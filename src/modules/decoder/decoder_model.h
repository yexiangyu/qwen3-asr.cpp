#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <map>

namespace qwen3_asr {
namespace decoder {

struct DecoderLayer {
    ggml_tensor* attn_norm = nullptr;
    
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_out = nullptr;
    
    ggml_tensor* attn_q_norm = nullptr;
    ggml_tensor* attn_k_norm = nullptr;
    
    ggml_tensor* ffn_norm = nullptr;
    
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct DecoderModel {
    HyperParams hparams;
    
    ggml_tensor* token_embd = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output = nullptr;
    
    std::vector<DecoderLayer> layers;
    
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    void* mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    std::map<std::string, ggml_tensor*> tensors;
};

struct KVCache {
    std::vector<ggml_tensor*> k_cache;
    std::vector<ggml_tensor*> v_cache;
    
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int n_ctx = 0;
    int n_used = 0;
    int head_dim = 0;
    int n_kv_heads = 0;
    int n_layers = 0;
};

struct DecoderState {
    DecoderModel* model = nullptr;
    KVCache kv_cache;
    
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    ggml_tensor* result_logits = nullptr;
};

} // namespace decoder
} // namespace qwen3_asr