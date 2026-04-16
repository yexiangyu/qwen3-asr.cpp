#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>

namespace qwen3_asr {

struct text_decoder_config {
    int32_t vocab_size = 151936;
    int32_t hidden_size = 1024;
    int32_t n_decoder_layers = 28;
    int32_t n_attention_heads = 32;
    int32_t n_key_value_heads = 16;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 64;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;
    int32_t audio_start_token_id = 151669;
    int32_t audio_end_token_id = 151670;
    int32_t audio_pad_token_id = 151676;
};

struct decoder_layer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// Text decoder model weights
struct text_decoder_model {
    text_decoder_config config;
    
    // Token embedding
    struct ggml_tensor * token_embd = nullptr;  // [vocab_size, hidden_size]
    
    // Transformer layers
    std::vector<decoder_layer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr; // [hidden_size]
    
    // LM head
    struct ggml_tensor * output = nullptr;      // [hidden_size, vocab_size]
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // mmap state — must outlive all tensors backed by this mapping
    void * mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

typedef int32_t kv_pos;
typedef int32_t kv_seq_id;

struct kv_cell {
    kv_pos pos = -1;
    std::set<kv_seq_id> seq_id;

    bool has_seq_id(const kv_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct kv_cache_batched {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int32_t n_ctx = 0;
    int32_t head = 0;
    int32_t n = 0;
    int32_t head_dim = 64;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;

    std::vector<kv_cell> cells;

    int32_t alloc_seq(kv_seq_id seq_id, int32_t n_tokens);
    void free_seq(kv_seq_id seq_id);
    void clear_all();
};

struct decode_batch {
    int32_t n_tokens;
    int32_t * token_ids;
    int32_t * positions;
    int32_t * seq_ids;
    int32_t n_seqs;
    int32_t * seq_n_tokens;
};

// Text decoder state
struct text_decoder_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    kv_cache_batched cache;
};

// Text decoder class
class TextDecoder {
public:
    TextDecoder();
    ~TextDecoder();
    
    bool load_model(const std::string & model_path, const std::string & device_name = "");
    
    bool init_kv_cache(int32_t n_ctx);
    
    void clear_kv_cache();
    
    bool forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                 std::vector<float> & output);
    
    bool forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                            const float * audio_embd, int32_t n_audio,
                            int32_t audio_start_pos, int32_t n_past,
                            std::vector<float> & output);
    
    bool forward_batch(const decode_batch & batch,
                       const float * audio_embd, int32_t n_audio,
                       int32_t audio_start_pos,
                       std::vector<float> & output);
    
    int32_t kv_alloc_seq(kv_seq_id seq_id, int32_t n_tokens);
    void kv_free_seq(kv_seq_id seq_id);
    void kv_clear_all();
    
    const text_decoder_config & get_config() const { return model_.config; }
    
    const std::string & get_error() const { return error_msg_; }
    
    std::string decode_token(int32_t token_id) const;
    
    std::string decode_tokens(const std::vector<int32_t> & tokens) const;
    
    std::vector<int32_t> tokenize(const std::string & text) const;
    
    bool forward_debug(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                       std::vector<float> & output,
                       std::map<std::string, std::vector<float>> & debug_tensors);
    
private:
    struct ggml_cgraph * build_graph(const int32_t * tokens, int32_t n_tokens,
                                     int32_t n_past,
                                     const float * audio_embd = nullptr,
                                     int32_t n_audio = 0,
                                     int32_t audio_start_pos = 0);
    
    struct ggml_cgraph * build_graph_batch(const decode_batch & batch,
                                           const float * audio_embd = nullptr,
                                           int32_t n_audio = 0,
                                           int32_t audio_start_pos = 0);
    
    bool init_state(const std::string & device_name = "");
    
    bool parse_config(struct gguf_context * ctx);
    
    bool assign_tensors(struct gguf_context * ctx_gguf);
    
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx_gguf);
    
    bool load_vocab(struct gguf_context * ctx);
    
    text_decoder_model model_;
    text_decoder_state state_;
    std::string error_msg_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;
};

void free_decoder_model(text_decoder_model & model);

void free_kv_cache(kv_cache_batched & cache);

decode_batch decode_batch_init(int32_t n_tokens, int32_t n_seqs);

void decode_batch_free(decode_batch & batch);

} // namespace qwen3_asr
