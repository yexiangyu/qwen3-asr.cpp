#include "text_decoder.h"
#include "timing.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <climits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define QWEN3_ASR_MAX_NODES 8192

namespace qwen3_asr {

TextDecoder::TextDecoder() = default;

TextDecoder::~TextDecoder() {
    free_kv_cache(state_.cache);
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend_gpu) {
        ggml_backend_free(state_.backend_gpu);
        state_.backend_gpu = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    free_decoder_model(model_);
}

bool TextDecoder::load_model(const std::string & model_path, const std::string & device_name) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model_.ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx_gguf) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx_gguf)) {
        gguf_free(ctx_gguf);
        if (model_.ctx) ggml_free(model_.ctx);
        model_.ctx = nullptr;
        return false;
    }
    
    if (!assign_tensors(ctx_gguf)) {
        gguf_free(ctx_gguf);
        if (model_.ctx) ggml_free(model_.ctx);
        model_.ctx = nullptr;
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx_gguf)) {
        free_decoder_model(model_);
        gguf_free(ctx_gguf);
        return false;
    }
    
    if (!load_vocab(ctx_gguf)) {
        free_decoder_model(model_);
        gguf_free(ctx_gguf);
        return false;
    }
    
    gguf_free(ctx_gguf);
    
    if (!init_state(device_name)) {
        return false;
    }
    
    if (!init_kv_cache(4096)) {
        return false;
    }
    
    return true;
}

bool TextDecoder::init_state(const std::string & device_name) {
    state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend_cpu) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }

    // Initialize GPU backend with device selection
    if (!device_name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name.c_str());
        if (dev) {
            state_.backend_gpu = ggml_backend_dev_init(dev, nullptr);
        }
    }
    
    if (!state_.backend_gpu) {
        state_.backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;

    if (state_.backend_gpu) {
        backends.push_back(state_.backend_gpu);
        backend_bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_gpu));
    }

    backends.push_back(state_.backend_cpu);
    backend_bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_cpu));

    state_.sched = ggml_backend_sched_new(backends.data(), backend_bufts.data(), backends.size(), QWEN3_ASR_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool TextDecoder::parse_config(struct gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & cfg = model_.config;
    cfg.vocab_size = get_u32("qwen3-asr.vocab_size", 151936);
    cfg.hidden_size = get_u32("qwen3-asr.embedding_length", 1024);
    cfg.n_decoder_layers = get_u32("qwen3-asr.block_count", 28);
    cfg.n_attention_heads = get_u32("qwen3-asr.attention.head_count", 16);
    cfg.n_key_value_heads = get_u32("qwen3-asr.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("qwen3-asr.feed_forward_length", 3072);
    cfg.head_dim = get_u32("qwen3-asr.attention.key_length", 128);
    cfg.rms_norm_eps = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("qwen3-asr.rope.freq_base", 1000000.0f);
    
    cfg.pad_token_id = 151643;
    cfg.eos_token_id = 151645;
    cfg.audio_start_token_id = get_u32("qwen3-asr.audio.start_token_id", 151669);
    cfg.audio_end_token_id = get_u32("qwen3-asr.audio.end_token_id", 151670);
    cfg.audio_pad_token_id = get_u32("qwen3-asr.audio.pad_token_id", 151676);
    
    return true;
}

bool TextDecoder::assign_tensors(struct gguf_context * ctx_gguf) {
    (void)ctx_gguf;
    const auto & cfg = model_.config;
    
    model_.layers.resize(cfg.n_decoder_layers);
    
    for (struct ggml_tensor * tensor = ggml_get_first_tensor(model_.ctx); 
         tensor; 
         tensor = ggml_get_next_tensor(model_.ctx, tensor)) {
        
        const char * name = ggml_get_name(tensor);
        
        if (strstr(name, "audio.encoder")) {
            continue;
        }
        
        model_.tensors[name] = tensor;
        
        if (strstr(name, "blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_decoder_layers) {
                auto & layer = model_.layers[layer_idx];
                
                if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        }
        else if (strstr(name, "token_embd.weight")) {
            model_.token_embd = tensor;
        } else if (strstr(name, "output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "output.weight")) {
            model_.output = tensor;
        }
    }
    
    if (!model_.output && model_.token_embd) {
        model_.output = model_.token_embd;
    }
    
    return true;
}

bool TextDecoder::load_tensor_data(const std::string & path, struct gguf_context * ctx_gguf) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg_ = "Failed to open file for mmap: " + path;
        return false;
    }
    
    struct stat st;
    if (fstat(fd, &st) != 0) {
        error_msg_ = "Failed to stat file: " + path;
        close(fd);
        return false;
    }
    
    void * mmap_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mmap_addr == MAP_FAILED) {
        error_msg_ = "Failed to mmap file: " + path;
        return false;
    }
    
    model_.mmap_addr = mmap_addr;
    model_.mmap_size = st.st_size;
    
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    uint8_t * data_base = (uint8_t *)mmap_addr + data_offset;
    
    // Following llama.cpp: use ggml_backend_alloc_ctx_tensors_from_buft
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_buffer_type_t buft = nullptr;
    
    if (gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (strstr(dev_name, "Metal") != nullptr) {
            buft = ggml_backend_dev_buffer_type(gpu_dev);
        }
#ifdef GGML_USE_CUDA
        else {
            buft = ggml_backend_cuda_buffer_type(0);
        }
#endif
    }
    
    if (!buft) {
        buft = ggml_backend_cpu_buffer_type();
    }
    
    model_.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model_.ctx, buft);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate context tensors with buffer type";
        munmap(mmap_addr, st.st_size);
        model_.mmap_addr = nullptr;
        model_.mmap_size = 0;
        return false;
    }
    
    fprintf(stderr, "info: text decoder allocated %zu bytes for model weights\n", 
            ggml_backend_buffer_get_size(model_.buffer));
    
    // Load tensor data - only for tensors we care about (skip audio.encoder)
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        if (strstr(name, "audio.encoder")) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);
        if (!tensor) {
            continue;
        }
        
        const size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
        const size_t sz = ggml_nbytes(tensor);
        
        ggml_backend_tensor_set(tensor, data_base + offset, 0, sz);
    }
    
    if (gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (strstr(dev_name, "Metal") == nullptr) {
            munmap(mmap_addr, st.st_size);
            model_.mmap_addr = nullptr;
            model_.mmap_size = 0;
        }
    }
    
    return true;
}

bool TextDecoder::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.head = 0;
    state_.cache.n = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_decoder_layers;
    
    state_.cache.cells.resize(n_ctx);
    for (int i = 0; i < n_ctx; ++i) {
        state_.cache.cells[i].pos = -1;
    }
    
    const size_t n_tensors = cfg.n_decoder_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(cfg.n_decoder_layers);
    state_.cache.v_cache.resize(cfg.n_decoder_layers);
    
    for (int il = 0; il < cfg.n_decoder_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    ggml_backend_t kv_backend = state_.backend_gpu ? state_.backend_gpu : state_.backend_cpu;
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, kv_backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

int32_t kv_cache_batched::alloc_seq(kv_seq_id seq_id, int32_t n_tokens) {
    int32_t start = head;
    for (int i = 0; i < n_tokens; ++i) {
        cells[head + i].pos = i;
        cells[head + i].seq_id.insert(seq_id);
    }
    head += n_tokens;
    n = head;
    return start;
}

void kv_cache_batched::free_seq(kv_seq_id seq_id) {
    for (int32_t i = 0; i < n_ctx; ++i) {
        cells[i].seq_id.erase(seq_id);
    }
}

void kv_cache_batched::clear_all() {
    head = 0;
    n = 0;
    for (int i = 0; i < n_ctx; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
    }
}

void TextDecoder::clear_kv_cache() {
    state_.cache.clear_all();
}

int32_t TextDecoder::kv_alloc_seq(kv_seq_id seq_id, int32_t n_tokens) {
    return state_.cache.alloc_seq(seq_id, n_tokens);
}

void TextDecoder::kv_free_seq(kv_seq_id seq_id) {
    state_.cache.free_seq(seq_id);
}

void TextDecoder::kv_clear_all() {
    state_.cache.clear_all();
}

struct ggml_cgraph * TextDecoder::build_graph(
    const int32_t * tokens, int32_t n_tokens, int32_t n_past,
    const float * audio_embd, int32_t n_audio, int32_t audio_start_pos) {
    (void)tokens;
    
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_decoder_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * inp_audio = nullptr;
    if (audio_embd && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);
    
    if (inp_audio && n_audio > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        struct ggml_tensor * embd_before = nullptr;
        struct ggml_tensor * embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio < n_tokens) {
            int after_start = audio_start_pos + n_audio;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            struct ggml_tensor * tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
            cur = ggml_concat(ctx0, tmp, embd_after, 1);
        } else if (embd_before) {
            cur = ggml_concat(ctx0, embd_before, inp_audio, 1);
        } else if (embd_after) {
            cur = ggml_concat(ctx0, inp_audio, embd_after, 1);
        } else {
            cur = inp_audio;
        }
        ggml_set_name(cur, "embd_with_audio");
        ggml_set_output(cur);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    int n_kv = n_past + n_tokens;
    struct ggml_tensor * fa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_name(fa_mask, "fa_mask");
    ggml_set_input(fa_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v || 
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate || 
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, K, V, fa_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        ggml_format_name(cur, "ffn_out_%d", il);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;

    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, hidden_size, 1, cur->nb[1], (n_tokens - 1) * cur->nb[1]);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TextDecoder::build_graph_batch(
    const decode_batch & batch,
    const float * audio_embd, int32_t n_audio, int32_t audio_start_pos) {
    
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_decoder_layers;
    
    const int n_tokens = batch.n_tokens;
    const int n_kv = state_.cache.n;
    const int kv_head = state_.cache.head;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * inp_seq_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_seq_ids, "inp_seq_ids");
    ggml_set_input(inp_seq_ids);
    
    struct ggml_tensor * inp_audio = nullptr;
    if (audio_embd && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);
    
    if (inp_audio && n_audio > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        struct ggml_tensor * embd_before = nullptr;
        struct ggml_tensor * embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio < n_tokens) {
            int after_start = audio_start_pos + n_audio;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            struct ggml_tensor * tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
            cur = ggml_concat(ctx0, tmp, embd_after, 1);
        } else if (embd_before) {
            cur = ggml_concat(ctx0, embd_before, inp_audio, 1);
        } else if (embd_after) {
            cur = ggml_concat(ctx0, inp_audio, embd_after, 1);
        } else {
            cur = inp_audio;
        }
        ggml_set_name(cur, "embd_with_audio");
        ggml_set_output(cur);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    struct ggml_tensor * fa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_name(fa_mask, "fa_mask");
    ggml_set_input(fa_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v || 
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate || 
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            kv_head * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            kv_head * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, K, V, fa_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        ggml_format_name(cur, "ffn_out_%d", il);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;

    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, hidden_size, 1, cur->nb[1], (n_tokens - 1) * cur->nb[1]);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool TextDecoder::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                          std::vector<float> & output) {
    return forward_with_audio(tokens, n_tokens, nullptr, 0, -1, n_past, output);
}

bool TextDecoder::forward_with_audio(
    const int32_t * tokens, int32_t n_tokens,
    const float * audio_embd, int32_t n_audio,
    int32_t audio_start_pos, int32_t n_past,
    std::vector<float> & output) {
    QWEN3_TIMER("decoder.forward");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(1024)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    decode_batch batch;
    std::vector<int32_t> tokens_vec(tokens, tokens + n_tokens);
    std::vector<int32_t> pos_vec(n_tokens);
    std::vector<int32_t> seq_vec(n_tokens, 0);
    
    for (int i = 0; i < n_tokens; ++i) {
        pos_vec[i] = n_past + i;
    }
    
    batch.n_tokens = n_tokens;
    batch.token_ids = tokens_vec.data();
    batch.positions = pos_vec.data();
    batch.seq_ids = seq_vec.data();
    batch.n_seqs = 1;
    batch.seq_n_tokens = nullptr;
    
    for (int i = 0; i < n_tokens; ++i) {
        state_.cache.cells[state_.cache.head + i].pos = pos_vec[i];
        state_.cache.cells[state_.cache.head + i].seq_id.insert(0);
    }
    state_.cache.n = state_.cache.head + n_tokens;
    
    struct ggml_cgraph * gf = build_graph_batch(batch,
                                          audio_embd, n_audio, audio_start_pos);
    
    if (!gf) {
        error_msg_ = "Failed to build graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        error_msg_ = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        ggml_backend_tensor_set(inp_pos, pos_vec.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_seq_ids = ggml_graph_get_tensor(gf, "inp_seq_ids");
    if (inp_seq_ids) {
        ggml_backend_tensor_set(inp_seq_ids, seq_vec.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = state_.cache.n;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            kv_seq_id seq_id_q = seq_vec[q];
            kv_pos pos_q = pos_vec[q];
            for (int k = 0; k < n_kv; ++k) {
                if (!state_.cache.cells[k].has_seq_id(seq_id_q) ||
                    state_.cache.cells[k].pos > pos_q) {
                    mask_data[k + q * n_kv] = neginf_f16;
                } else {
                    mask_data[k + q * n_kv] = zero_f16;
                }
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (audio_embd && n_audio > 0) {
        struct ggml_tensor * inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_embd, 0, 
                                    n_audio * model_.config.hidden_size * sizeof(float));
        }
    }
    
    {
        QWEN3_TIMER("decoder.compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    int64_t n_logit_rows = logits->ne[1];
    output.resize(n_logit_rows * vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.head += n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TextDecoder::forward_batch(const decode_batch & batch,
                                const float * audio_embd, int32_t n_audio,
                                int32_t audio_start_pos,
                                std::vector<float> & output) {
    QWEN3_TIMER("decoder.forward_batch");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(1024)) {
            return false;
        }
    }
    
    if (state_.cache.head + batch.n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    for (int i = 0; i < batch.n_tokens; ++i) {
        state_.cache.cells[state_.cache.head + i].pos = batch.positions[i];
        state_.cache.cells[state_.cache.head + i].seq_id.insert(batch.seq_ids[i]);
    }
    state_.cache.n = state_.cache.head + batch.n_tokens;
    
    struct ggml_cgraph * gf = build_graph_batch(batch,
                                          audio_embd, n_audio, audio_start_pos);
    
    if (!gf) {
        error_msg_ = "Failed to build graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens) {
        ggml_backend_tensor_set(inp_tokens, batch.token_ids, 0, batch.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        ggml_backend_tensor_set(inp_pos, batch.positions, 0, batch.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_seq_ids = ggml_graph_get_tensor(gf, "inp_seq_ids");
    if (inp_seq_ids) {
        ggml_backend_tensor_set(inp_seq_ids, batch.seq_ids, 0, batch.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = state_.cache.n;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * batch.n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < batch.n_tokens; ++q) {
            kv_seq_id seq_id_q = batch.seq_ids[q];
            kv_pos pos_q = batch.positions[q];
            for (int k = 0; k < n_kv; ++k) {
                if (!state_.cache.cells[k].has_seq_id(seq_id_q) ||
                    state_.cache.cells[k].pos > pos_q) {
                    mask_data[k + q * n_kv] = neginf_f16;
                } else {
                    mask_data[k + q * n_kv] = zero_f16;
                }
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (audio_embd && n_audio > 0) {
        struct ggml_tensor * inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_embd, 0,
                                    n_audio * model_.config.hidden_size * sizeof(float));
        }
    }
    
    {
        QWEN3_TIMER("decoder.compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    int64_t n_logit_rows = logits->ne[1];
    output.resize(n_logit_rows * vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.head += batch.n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TextDecoder::forward_debug(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                                std::vector<float> & output,
                                std::map<std::string, std::vector<float>> & debug_tensors) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(1024)) {
            return false;
        }
    }
    
    std::vector<int32_t> pos_vec(n_tokens);
    std::vector<int32_t> seq_vec(n_tokens, 0);
    for (int i = 0; i < n_tokens; ++i) {
        pos_vec[i] = n_past + i;
    }
    
    for (int i = 0; i < n_tokens; ++i) {
        state_.cache.cells[state_.cache.head + i].pos = pos_vec[i];
        state_.cache.cells[state_.cache.head + i].seq_id.insert(0);
    }
    state_.cache.n = state_.cache.head + n_tokens;
    
    decode_batch batch;
    batch.n_tokens = n_tokens;
    batch.token_ids = const_cast<int32_t*>(tokens);
    batch.positions = pos_vec.data();
    batch.seq_ids = seq_vec.data();
    batch.n_seqs = 1;
    batch.seq_n_tokens = nullptr;
    
    struct ggml_cgraph * gf = build_graph_batch(batch, nullptr, 0, -1);
    
    if (!gf) {
        error_msg_ = "Failed to build graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens_t = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens_t) {
        ggml_backend_tensor_set(inp_tokens_t, tokens, 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        ggml_backend_tensor_set(inp_pos, pos_vec.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_seq_ids = ggml_graph_get_tensor(gf, "inp_seq_ids");
    if (inp_seq_ids) {
        ggml_backend_tensor_set(inp_seq_ids, seq_vec.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = state_.cache.n;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            kv_seq_id seq_id_q = seq_vec[q];
            kv_pos pos_q = pos_vec[q];
            for (int k = 0; k < n_kv; ++k) {
                if (!state_.cache.cells[k].has_seq_id(seq_id_q) ||
                    state_.cache.cells[k].pos > pos_q) {
                    mask_data[k + q * n_kv] = neginf_f16;
                } else {
                    mask_data[k + q * n_kv] = zero_f16;
                }
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (logits) {
        int64_t vocab_size = logits->ne[0];
        output.resize(n_tokens * vocab_size);
        ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    }
    
    const char * debug_names[] = {"debug_norm0", "debug_q0_raw", "debug_q0_normed", "debug_q0_rope", 
                                   "debug_attn0_out", "debug_kq_scaled", "debug_kq_masked", "debug_kq_softmax"};
    for (const char * name : debug_names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) {
            size_t nbytes = ggml_nbytes(t);
            std::vector<float> data(nbytes / sizeof(float));
            ggml_backend_tensor_get(t, data.data(), 0, nbytes);
            debug_tensors[name] = std::move(data);
        }
    }
    
    state_.cache.head += n_tokens;
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_decoder_model(text_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.mmap_addr) {
        munmap(model.mmap_addr, model.mmap_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
    }
    model.tensors.clear();
    model.layers.clear();
}

void free_kv_cache(kv_cache_batched & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.cells.clear();
    cache.n_ctx = 0;
    cache.head = 0;
    cache.n = 0;
}

decode_batch decode_batch_init(int32_t n_tokens, int32_t n_seqs) {
    decode_batch batch;
    batch.n_tokens = n_tokens;
    batch.token_ids = (int32_t *) malloc(sizeof(int32_t) * n_tokens);
    batch.positions = (int32_t *) malloc(sizeof(int32_t) * n_tokens);
    batch.seq_ids = (int32_t *) malloc(sizeof(int32_t) * n_tokens);
    batch.n_seqs = n_seqs;
    batch.seq_n_tokens = (int32_t *) malloc(sizeof(int32_t) * n_seqs);
    return batch;
}

void decode_batch_free(decode_batch & batch) {
    if (batch.token_ids) free(batch.token_ids);
    if (batch.positions) free(batch.positions);
    if (batch.seq_ids) free(batch.seq_ids);
    if (batch.seq_n_tokens) free(batch.seq_n_tokens);
    batch.token_ids = nullptr;
    batch.positions = nullptr;
    batch.seq_ids = nullptr;
    batch.seq_n_tokens = nullptr;
    batch.n_tokens = 0;
    batch.n_seqs = 0;
}

bool TextDecoder::load_vocab(struct gguf_context * ctx) {
    int64_t tokens_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        error_msg_ = "Vocabulary not found in GGUF file";
        return false;
    }
    
    int64_t n_vocab = gguf_get_arr_n(ctx, tokens_idx);
    if (n_vocab <= 0) {
        error_msg_ = "Empty vocabulary in GGUF file";
        return false;
    }
    
    vocab_.resize(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        vocab_[i] = gguf_get_arr_str(ctx, tokens_idx, i);
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }
    
    int64_t merges_idx = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        int64_t n_merges = gguf_get_arr_n(ctx, merges_idx);
        for (int64_t i = 0; i < n_merges; ++i) {
            std::string merge = gguf_get_arr_str(ctx, merges_idx, i);
            bpe_ranks_[merge] = static_cast<int>(i);
        }
    }
    
    return true;
}

// GPT-2 byte-level BPE: reverse mapping from Unicode codepoints back to raw bytes.
// See HuggingFace tokenizers bytes_to_unicode() — printable bytes map to themselves,
// non-printable bytes map to codepoints 256+n.
static std::vector<int> build_unicode_to_byte_table() {
    std::vector<int> byte_to_cp(256, 0);
    std::vector<bool> assigned(256, false);

    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            byte_to_cp[b] = b;
            assigned[b] = true;
        }
    };
    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!assigned[b]) {
            byte_to_cp[b] = 256 + n;
            ++n;
        }
    }

    std::vector<int> cp_to_byte(512, -1);
    for (int b = 0; b < 256; ++b) {
        cp_to_byte[byte_to_cp[b]] = b;
    }
    return cp_to_byte;
}

std::string TextDecoder::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab_.size()) {
        return "";
    }

    std::string token = vocab_[token_id];

    // Skip special tokens like <|...|> and [PAD...]
    if (token.size() >= 3 && token[0] == '<' && token[1] == '|' &&
        token[token.size()-1] == '>' && token[token.size()-2] == '|') {
        return "";
    }
    if (token.size() >= 5 && token.substr(0, 4) == "[PAD") {
        return "";
    }

    // Byte-level BPE decode: each Unicode codepoint in the token string maps to a byte
    // via the GPT-2 bytes_to_unicode table.
    static const std::vector<int> cp_to_byte = build_unicode_to_byte_table();

    std::string bytes;
    bytes.reserve(token.size());

    size_t i = 0;
    while (i < token.size()) {
        // Decode one UTF-8 codepoint from the token string
        uint32_t cp = 0;
        unsigned char c = static_cast<unsigned char>(token[i]);
        size_t len = 0;

        if (c < 0x80) {
            cp = c; len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07; len = 4;
        } else {
            // Invalid UTF-8 start byte, pass through
            bytes += static_cast<char>(c);
            ++i;
            continue;
        }

        if (i + len > token.size()) {
            // Truncated sequence, pass through remaining bytes
            while (i < token.size()) {
                bytes += token[i++];
            }
            break;
        }

        for (size_t j = 1; j < len; ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(token[i + j]) & 0x3F);
        }
        i += len;

        // Look up the byte value for this codepoint
        if (cp < cp_to_byte.size() && cp_to_byte[cp] >= 0) {
            bytes += static_cast<char>(cp_to_byte[cp]);
        } else {
            // Codepoint not in BPE table — encode back as UTF-8 (shouldn't happen for valid vocab)
            if (cp < 0x80) {
                bytes += static_cast<char>(cp);
            } else if (cp < 0x800) {
                bytes += static_cast<char>(0xC0 | (cp >> 6));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                bytes += static_cast<char>(0xE0 | (cp >> 12));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                bytes += static_cast<char>(0xF0 | (cp >> 18));
                bytes += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
    }

    return bytes;
}

std::string TextDecoder::decode_tokens(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

namespace {

static const std::vector<std::string> & get_byte_to_unicode_table() {
    static std::vector<std::string> table;
    if (!table.empty()) return table;
    table.resize(256);

    std::vector<int> byte_to_cp(256, 0);
    std::vector<bool> assigned(256, false);

    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            byte_to_cp[b] = b;
            assigned[b] = true;
        }
    };
    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!assigned[b]) {
            byte_to_cp[b] = 256 + n;
            ++n;
        }
    }

    auto cp_to_utf8 = [](int cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    };

    for (int b = 0; b < 256; ++b) {
        table[b] = cp_to_utf8(byte_to_cp[b]);
    }
    return table;
}

static std::string bytes_to_bpe_string(const std::string & text) {
    const auto & table = get_byte_to_unicode_table();
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        result += table[c];
    }
    return result;
}

static std::vector<std::string> split_utf8_chars(const std::string & s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) len = 1;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

static std::vector<std::string> bpe_encode_word(
    const std::string & word_bpe,
    const std::unordered_map<std::string, int> & bpe_ranks) {

    std::vector<std::string> symbols = split_utf8_chars(word_bpe);
    if (symbols.size() <= 1) return symbols;

    while (true) {
        int best_rank = INT_MAX;
        size_t best_pos = 0;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = symbols[i] + " " + symbols[i + 1];
            auto it = bpe_ranks.find(key);
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_rank == INT_MAX) break;

        std::string merged = symbols[best_pos] + symbols[best_pos + 1];
        std::vector<std::string> new_symbols;
        new_symbols.reserve(symbols.size() - 1);
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i == best_pos) {
                new_symbols.push_back(merged);
                ++i;
            } else {
                new_symbols.push_back(symbols[i]);
            }
        }
        symbols = std::move(new_symbols);
        if (symbols.size() == 1) break;
    }

    return symbols;
}

}

std::vector<int32_t> TextDecoder::tokenize(const std::string & text) const {
    std::vector<int32_t> tokens;
    
    std::istringstream iss(text);
    std::string word;
    bool first = true;
    
    while (iss >> word) {
        std::string to_encode = first ? word : (" " + word);
        first = false;
        
        std::string bpe_str = bytes_to_bpe_string(to_encode);
        std::vector<std::string> subwords = bpe_encode_word(bpe_str, bpe_ranks_);
        
        for (const auto & sw : subwords) {
            auto it = token_to_id_.find(sw);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            }
        }
    }
    
    return tokens;
}

} // namespace qwen3_asr
