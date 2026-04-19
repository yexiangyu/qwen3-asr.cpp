#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace asr::aligner::decoder {

struct HyperParams {
    int vocab_size = 152064;
    int hidden_size = 1024;
    int n_layers = 28;
    int n_heads = 16;
    int n_kv_heads = 8;
    int head_dim = 128;
    int intermediate_size = 3072;
    int classify_head_size = 5000;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    int timestamp_token_id = 151705;
    int audio_start_token_id = 151669;
    int audio_end_token_id = 151670;
    int audio_pad_token_id = 151676;
    int pad_token_id = 151643;
    int eos_token_id = 151645;
    int timestamp_segment_time_ms = 80;
};

struct Layer {
    ggml_tensor* attn_norm = nullptr;
    
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_output = nullptr;
    
    ggml_tensor* attn_q_norm = nullptr;
    ggml_tensor* attn_k_norm = nullptr;
    
    ggml_tensor* ffn_norm = nullptr;
    
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct Model {
    HyperParams hparams;
    
    ggml_tensor* token_embd = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* classify_head_w = nullptr;
    ggml_tensor* classify_head_b = nullptr;
    
    std::vector<Layer> layers;
    
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    void* mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    std::map<std::string, ggml_tensor*> tensors;
    
    std::vector<std::string> vocab;
    std::map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> decoded_to_id;
    std::unordered_map<std::string, int> bpe_ranks;
    std::unordered_set<std::string> ko_dict;
};

struct Cache {
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

struct State {
    Model* model = nullptr;
    Cache kv_cache;
    
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    ggml_tensor* result_logits = nullptr;
};

} // namespace asr::aligner::decoder
