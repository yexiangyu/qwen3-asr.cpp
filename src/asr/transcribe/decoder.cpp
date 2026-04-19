#include "asr/transcribe/decoder.hpp"
#include "asr/transcribe/decoder_model.hpp"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

namespace asr { namespace transcribe { namespace decoder {

using asr::ErrorInfo;

constexpr int MAX_NODES = 8192;

static ggml_backend_buffer_type_t get_preferred_buft() {
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        const char* dev_name = ggml_backend_dev_name(gpu_dev);
        if (strstr(dev_name, "Metal") != nullptr) {
            return ggml_backend_dev_buffer_type(gpu_dev);
        }
#ifdef GGML_USE_CUDA
        return ggml_backend_cuda_buffer_type(0);
#endif
    }
    return ggml_backend_cpu_buffer_type();
}

static bool load_model_internal(const char* path, Model& model, ErrorInfo* error) {
    struct gguf_init_params params = { true, &model.ctx };
    struct gguf_context* ctx_gguf = gguf_init_from_file(path, params);
    if (!ctx_gguf) {
        if (error) error->message = "Failed to open GGUF: " + std::string(path);
        return false;
    }
    
    auto get_u32 = [&](const char* key, int32_t def) {
        int64_t idx = gguf_find_key(ctx_gguf, key);
        return idx < 0 ? def : (int32_t)gguf_get_val_u32(ctx_gguf, idx);
    };
    
    auto get_f32 = [&](const char* key, float def) {
        int64_t idx = gguf_find_key(ctx_gguf, key);
        return idx < 0 ? def : gguf_get_val_f32(ctx_gguf, idx);
    };
    
    model.hparams.vocab_size = get_u32("qwen3-asr.vocab_size", 151936);
    model.hparams.hidden_size = get_u32("qwen3-asr.embedding_length", 1024);
    model.hparams.n_layers = get_u32("qwen3-asr.block_count", 28);
    model.hparams.n_heads = get_u32("qwen3-asr.attention.head_count", 16);
    model.hparams.n_kv_heads = get_u32("qwen3-asr.attention.head_count_kv", 8);
    model.hparams.intermediate_size = get_u32("qwen3-asr.feed_forward_length", 3072);
    model.hparams.head_dim = get_u32("qwen3-asr.attention.key_length", 128);
    model.hparams.rms_norm_eps = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    model.hparams.rope_theta = get_f32("qwen3-asr.rope.freq_base", 1000000.0f);
    
    model.hparams.audio_start_token = get_u32("qwen3-asr.audio.start_token_id", 151669);
    model.hparams.audio_end_token = get_u32("qwen3-asr.audio.end_token_id", 151670);
    model.hparams.audio_pad_token = get_u32("qwen3-asr.audio.pad_token_id", 151676);
    model.hparams.pad_token = 151643;
    model.hparams.eos_token = 151645;
    
    model.layers.resize(model.hparams.n_layers);
    
    for (ggml_tensor* t = ggml_get_first_tensor(model.ctx); t; t = ggml_get_next_tensor(model.ctx, t)) {
        const char* name = ggml_get_name(t);
        
        if (strstr(name, "audio.encoder")) {
            continue;
        }
        
        model.tensors[name] = t;
        
        if (strstr(name, "blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < model.hparams.n_layers) {
                auto& layer = model.layers[layer_idx];
                
                if (strstr(name, "attn_output.weight")) layer.attn_out = t;
                else if (strstr(name, "attn_norm.weight")) layer.attn_norm = t;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = t;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = t;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = t;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = t;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = t;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = t;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = t;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = t;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = t;
            }
        }
        else if (strstr(name, "token_embd.weight")) {
            model.token_embd = t;
        }
        else if (strstr(name, "output_norm.weight")) {
            model.output_norm = t;
        }
        else if (strstr(name, "output.weight")) {
            model.output = t;
        }
    }
    
    if (!model.output && model.token_embd) {
        model.output = model.token_embd;
    }
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (error) error->message = "Cannot mmap file";
        gguf_free(ctx_gguf);
        return false;
    }
    
    struct stat st;
    fstat(fd, &st);
    void* mmap_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mmap_addr == MAP_FAILED) {
        if (error) error->message = "mmap failed";
        gguf_free(ctx_gguf);
        return false;
    }
    
    model.mmap_addr = mmap_addr;
    model.mmap_size = st.st_size;
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    uint8_t* data_base = (uint8_t*)mmap_addr + data_offset;
    
    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, get_preferred_buft());
    if (!model.buffer) {
        if (error) error->message = "Failed to allocate tensor buffer";
        munmap(mmap_addr, st.st_size);
        gguf_free(ctx_gguf);
        return false;
    }
    
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        
        if (strstr(name, "audio.encoder")) {
            continue;
        }
        
        ggml_tensor* t = ggml_get_tensor(model.ctx, name);
        if (t) {
            ggml_backend_tensor_set(t, data_base + gguf_get_tensor_offset(ctx_gguf, i), 0, ggml_nbytes(t));
        }
    }
    
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev && strstr(ggml_backend_dev_name(gpu_dev), "Metal") == nullptr) {
        munmap(mmap_addr, st.st_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
    }
    
    // Load vocabulary
    int64_t tokens_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (tokens_idx >= 0) {
        int64_t n_vocab = gguf_get_arr_n(ctx_gguf, tokens_idx);
        model.vocab.resize(n_vocab);
        for (int64_t i = 0; i < n_vocab; ++i) {
            model.vocab[i] = gguf_get_arr_str(ctx_gguf, tokens_idx, i);
            model.token_to_id[model.vocab[i]] = static_cast<int>(i);
        }
    }
    
    // Load BPE merges
    int64_t merges_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        int64_t n_merges = gguf_get_arr_n(ctx_gguf, merges_idx);
        for (int64_t i = 0; i < n_merges; ++i) {
            std::string merge = gguf_get_arr_str(ctx_gguf, merges_idx, i);
            model.bpe_ranks[merge] = static_cast<int>(i);
        }
    }
    
    gguf_free(ctx_gguf);
    fprintf(stderr, "Decoder model loaded: %zu bytes\n", ggml_backend_buffer_get_size(model.buffer));
    return true;
}

static void free_decoder_model(Model& model) {
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
    model.layers.clear();
    model.tensors.clear();
}

static bool init_kv_cache(State* state, int n_ctx) {
    const auto& hp = state->model->hparams;
    
    state->kv_cache.n_ctx = n_ctx;
    state->kv_cache.n_used = 0;
    state->kv_cache.head_dim = hp.head_dim;
    state->kv_cache.n_kv_heads = hp.n_kv_heads;
    state->kv_cache.n_layers = hp.n_layers;
    
    const size_t n_tensors = hp.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        ctx_size,
        nullptr,
        true
    };
    
    state->kv_cache.ctx = ggml_init(params);
    if (!state->kv_cache.ctx) {
        return false;
    }
    
    state->kv_cache.k_cache.resize(hp.n_layers);
    state->kv_cache.v_cache.resize(hp.n_layers);
    
    for (int il = 0; il < hp.n_layers; ++il) {
        state->kv_cache.k_cache[il] = ggml_new_tensor_3d(
            state->kv_cache.ctx, GGML_TYPE_F16,
            hp.head_dim, hp.n_kv_heads, n_ctx);
        ggml_format_name(state->kv_cache.k_cache[il], "k_cache_%d", il);
        
        state->kv_cache.v_cache[il] = ggml_new_tensor_3d(
            state->kv_cache.ctx, GGML_TYPE_F16,
            hp.head_dim, hp.n_kv_heads, n_ctx);
        ggml_format_name(state->kv_cache.v_cache[il], "v_cache_%d", il);
    }
    
    ggml_backend_t kv_backend = state->backend_gpu ? state->backend_gpu : state->backend_cpu;
    state->kv_cache.buffer = ggml_backend_alloc_ctx_tensors(state->kv_cache.ctx, kv_backend);
    if (!state->kv_cache.buffer) {
        return false;
    }
    
    return true;
}

static void free_kv_cache(Cache& cache) {
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
    cache.n_ctx = 0;
    cache.n_used = 0;
}

State* init(const Config& config) {
    State* state = new State();
    state->model = new Model();
    
    ErrorInfo err;
    if (!load_model_internal(config.model_path.c_str(), *state->model, &err)) {
        fprintf(stderr, "Failed to load decoder model: %s\n", err.message.c_str());
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state->backend_cpu) {
        free_decoder_model(*state->model);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    if (!config.device_name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(config.device_name.c_str());
        if (dev) {
            state->backend_gpu = ggml_backend_dev_init(dev, nullptr);
        }
    }
    if (!state->backend_gpu) {
        state->backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    
    std::vector<ggml_backend_t> backends;
    if (state->backend_gpu) {
        backends.push_back(state->backend_gpu);
    }
    backends.push_back(state->backend_cpu);
    
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto be : backends) {
        bufts.push_back(ggml_backend_get_default_buffer_type(be));
    }
    
    state->sched = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), MAX_NODES, false, true);
    if (!state->sched) {
        free_decoder_model(*state->model);
        if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
        ggml_backend_free(state->backend_cpu);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->compute_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    if (!init_kv_cache(state, config.max_ctx_length)) {
        ggml_backend_sched_free(state->sched);
        if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
        ggml_backend_free(state->backend_cpu);
        free_decoder_model(*state->model);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    return state;
}

void free(State* state) {
    if (!state) return;
    
    free_kv_cache(state->kv_cache);
    
    if (state->sched) {
        ggml_backend_sched_free(state->sched);
        state->sched = nullptr;
    }
    if (state->backend_gpu) {
        ggml_backend_free(state->backend_gpu);
        state->backend_gpu = nullptr;
    }
    if (state->backend_cpu) {
        ggml_backend_free(state->backend_cpu);
        state->backend_cpu = nullptr;
    }
    if (state->model) {
        free_decoder_model(*state->model);
        delete state->model;
        state->model = nullptr;
    }
    
    delete state;
}

void clear_kv_cache(State* state) {
    if (!state) return;
    state->kv_cache.n_used = 0;
    
    if (state->kv_cache.buffer) {
        for (int il = 0; il < state->kv_cache.n_layers; ++il) {
            if (state->kv_cache.k_cache[il]) {
                ggml_backend_buffer_clear(state->kv_cache.buffer, 0);
                break;
            }
        }
    }
}

int get_kv_cache_used(State* state) {
    if (!state) return 0;
    return state->kv_cache.n_used;
}

int get_kv_cache_capacity(State* state) {
    if (!state) return 0;
    return state->kv_cache.n_ctx;
}

const char* get_device_name(State* state) {
    if (!state) return "CPU";
    return state->backend_gpu ? ggml_backend_name(state->backend_gpu) : "CPU";
}

HyperParams get_hparams(State* state) {
    if (!state || !state->model) return HyperParams();
    return state->model->hparams;
}

bool load_ref_data(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    size_t sz = file.tellg();
    file.seekg(0);
    data.resize(sz / sizeof(float));
    file.read((char*)data.data(), sz);
    return true;
}

bool save_ref_data(const char* path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.write((const char*)data.data(), data.size() * sizeof(float));
    return true;
}

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose) {
    if (a.size() != b.size()) {
        if (verbose) fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return false;
    }
    float max_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, fabsf(a[i] - b[i]));
    }
    if (max_diff > tolerance) {
        if (verbose) fprintf(stderr, "Max diff: %.6f > %.6f\n", max_diff, tolerance);
        return false;
    }
    return true;
}

static ggml_cgraph* build_prefill_graph(
    State* state,
    const int* tokens, int n_tokens,
    const float* audio_features, int n_audio_frames, int audio_start_pos,
    int n_past) {
    
    const auto& hp = state->model->hparams;
    const int n_head = hp.n_heads;
    const int n_kv_head = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int hidden_size = hp.hidden_size;
    const float eps = hp.rms_norm_eps;
    const float rope_theta = hp.rope_theta;
    const int n_layer = hp.n_layers;
    
    struct ggml_init_params params = {
        state->compute_meta.size(),
        state->compute_meta.data(),
        true
    };
    
    struct ggml_context* ctx0 = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, MAX_NODES, false);
    
    struct ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor* inp_audio = nullptr;
    if (audio_features && n_audio_frames > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio_frames);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor* cur = ggml_get_rows(ctx0, state->model->token_embd, inp_tokens);
    
    if (inp_audio && n_audio_frames > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio_frames <= n_tokens) {
        struct ggml_tensor* embd_before = nullptr;
        struct ggml_tensor* embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio_frames < n_tokens) {
            int after_start = audio_start_pos + n_audio_frames;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            struct ggml_tensor* tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
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
    
    struct ggml_tensor* inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    int n_kv = n_past + n_tokens;
    struct ggml_tensor* fa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_name(fa_mask, "fa_mask");
    ggml_set_input(fa_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = state->model->layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v ||
            !layer.attn_out || !layer.ffn_norm || !layer.ffn_gate ||
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
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
        
        struct ggml_tensor* k_cache = state->kv_cache.k_cache[il];
        struct ggml_tensor* v_cache = state->kv_cache.v_cache[il];
        
        struct ggml_tensor* k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor* v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        struct ggml_tensor* K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor* V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor* Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, K, V, fa_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_out, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor* inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor* gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor* up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
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
    cur = ggml_mul(ctx0, cur, state->model->output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, state->model->output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

static std::vector<int32_t> find_audio_pad_positions(
    const int* tokens, int n_tokens, int audio_pad_token) {
    
    std::vector<int32_t> positions;
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens[i] == audio_pad_token) {
            positions.push_back(i);
        }
    }
    return positions;
}

bool prefill(State* state, const PrefillInput& input, DecoderOutput& output, ErrorInfo* error) {
    if (!state || !state->model) {
        if (error) error->message = "State or model not initialized";
        return false;
    }
    
    if (!input.tokens || input.n_tokens <= 0) {
        if (error) error->message = "Invalid input tokens";
        return false;
    }
    
    if (state->kv_cache.n_ctx == 0) {
        if (error) error->message = "KV cache not initialized";
        return false;
    }
    
    int n_past = 0;
    
    if (n_past + input.n_tokens > state->kv_cache.n_ctx) {
        if (error) error->message = "Context length exceeded";
        return false;
    }
    
    int audio_start_pos = input.audio_start_pos;
    if (audio_start_pos < 0 && input.audio_features && input.n_audio_frames > 0) {
        std::vector<int32_t> audio_positions = find_audio_pad_positions(
            input.tokens, input.n_tokens, state->model->hparams.audio_pad_token);
        
        if (!audio_positions.empty()) {
            audio_start_pos = audio_positions[0];
            
            if ((int)audio_positions.size() != input.n_audio_frames) {
                if (error) error->message = "Mismatch: " + 
                    std::to_string(audio_positions.size()) + " audio_pad tokens but " +
                    std::to_string(input.n_audio_frames) + " audio frames";
                return false;
            }
        }
    }
    
    struct ggml_cgraph* gf = build_prefill_graph(
        state, input.tokens, input.n_tokens,
        input.audio_features, input.n_audio_frames, audio_start_pos,
        n_past);
    
    if (!gf) {
        if (error) error->message = "Failed to build prefill graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) {
        if (error) error->message = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor* inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        if (error) error->message = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, input.tokens, 0, input.n_tokens * sizeof(int));
    
    struct ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(input.n_tokens);
        for (int i = 0; i < input.n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, input.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor* fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = n_past + input.n_tokens;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * input.n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < input.n_tokens; ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask_data[k + q * n_kv] = (k <= n_past + q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (input.audio_features && input.n_audio_frames > 0) {
        struct ggml_tensor* inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, input.audio_features, 0,
                                    input.n_audio_frames * state->model->hparams.hidden_size * sizeof(float));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state->sched, gf) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "Failed to compute graph";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    struct ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        if (error) error->message = "Failed to find logits tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    int64_t n_logit_rows = logits->ne[1];
    output.logits.resize(n_logit_rows * vocab_size);
    output.vocab_size = vocab_size;
    ggml_backend_tensor_get(logits, output.logits.data(), 0, output.logits.size() * sizeof(float));
    
    state->kv_cache.n_used = n_past + input.n_tokens;
    
    ggml_backend_sched_reset(state->sched);
    
    return true;
}

bool decode(State* state, const DecodeInput& input, DecoderOutput& output, ErrorInfo* error) {
    if (!state || !state->model) {
        if (error) error->message = "State or model not initialized";
        return false;
    }
    
    if (!input.tokens || input.n_tokens <= 0) {
        if (error) error->message = "Invalid input tokens";
        return false;
    }
    
    if (state->kv_cache.n_ctx == 0) {
        if (error) error->message = "KV cache not initialized";
        return false;
    }
    
    if (input.n_past + input.n_tokens > state->kv_cache.n_ctx) {
        if (error) error->message = "Context length exceeded";
        return false;
    }
    
    struct ggml_cgraph* gf = build_prefill_graph(
        state, input.tokens, input.n_tokens,
        nullptr, 0, -1,
        input.n_past);
    
    if (!gf) {
        if (error) error->message = "Failed to build decode graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) {
        if (error) error->message = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor* inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        if (error) error->message = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, input.tokens, 0, input.n_tokens * sizeof(int));
    
    struct ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(input.n_tokens);
        for (int i = 0; i < input.n_tokens; ++i) {
            positions[i] = input.n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, input.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor* fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = input.n_past + input.n_tokens;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * input.n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < input.n_tokens; ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask_data[k + q * n_kv] = (k <= input.n_past + q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (ggml_backend_sched_graph_compute(state->sched, gf) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "Failed to compute graph";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    struct ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        if (error) error->message = "Failed to find logits tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    int64_t n_logit_rows = logits->ne[1];
    output.logits.resize(n_logit_rows * vocab_size);
    output.vocab_size = vocab_size;
    ggml_backend_tensor_get(logits, output.logits.data(), 0, output.logits.size() * sizeof(float));
    
    state->kv_cache.n_used = input.n_past + input.n_tokens;
    
    ggml_backend_sched_reset(state->sched);
    
    return true;
}

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

static std::string decode_token_model(const Model* model, int token_id) {
    if (!model) return "";
    if (token_id < 0 || token_id >= (int)model->vocab.size()) return "";

    const std::string& token = model->vocab[token_id];

    if (token.size() >= 3 && token[0] == '<' && token[1] == '|' &&
        token[token.size()-1] == '>' && token[token.size()-2] == '|') {
        return "";
    }
    if (token.size() >= 5 && token.substr(0, 4) == "[PAD") {
        return "";
    }

    static const std::vector<int> cp_to_byte = build_unicode_to_byte_table();

    std::string bytes;
    bytes.reserve(token.size());

    size_t i = 0;
    while (i < token.size()) {
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
            bytes += static_cast<char>(c);
            ++i;
            continue;
        }

        if (i + len > token.size()) {
            while (i < token.size()) {
                bytes += token[i++];
            }
            break;
        }

        for (size_t j = 1; j < len; ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(token[i + j]) & 0x3F);
        }
        i += len;

        if (cp < cp_to_byte.size() && cp_to_byte[cp] >= 0) {
            bytes += static_cast<char>(cp_to_byte[cp]);
        } else {
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

std::string decode_token(const State* state, int token_id) {
    if (!state || !state->model) return "";
    return decode_token_model(state->model, token_id);
}

static std::string decode_tokens_model(const Model* model, const std::vector<int>& tokens) {
    std::string result;
    for (int token : tokens) {
        result += decode_token_model(model, token);
    }
    return result;
}

std::string decode_tokens(const State* state, const std::vector<int>& tokens) {
    if (!state || !state->model) return "";
    return decode_tokens_model(state->model, tokens);
}

static std::vector<std::string> get_byte_to_unicode_table() {
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

static std::string bytes_to_bpe_string(const std::string& text) {
    const auto& table = get_byte_to_unicode_table();
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        result += table[c];
    }
    return result;
}

static std::vector<std::string> bpe_encode_word(const std::string& word, const std::map<std::string, int>& bpe_ranks) {
    if (word.empty()) return {};

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > word.size()) len = 1;
        symbols.push_back(word.substr(i, len));
        i += len;
    }

    if (symbols.size() == 1) return symbols;

    std::map<std::string, int> ranks = bpe_ranks;

    while (symbols.size() > 1) {
        int min_rank = INT_MAX;
        size_t min_idx = 0;

        for (size_t j = 0; j < symbols.size() - 1; ++j) {
            std::string pair = symbols[j] + symbols[j + 1];
            auto it = ranks.find(pair);
            if (it != ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                min_idx = j;
            }
        }

        if (min_rank == INT_MAX) break;

        std::vector<std::string> new_symbols;
        for (size_t j = 0; j < symbols.size(); ++j) {
            if (j == min_idx) {
                new_symbols.push_back(symbols[j] + symbols[j + 1]);
                ++j;
            } else {
                new_symbols.push_back(symbols[j]);
            }
        }
        symbols = std::move(new_symbols);
        if (symbols.size() == 1) break;
    }

    return symbols;
}

static std::vector<int> tokenize_model(const Model* model, const std::string& text) {
    if (!model) return {};

    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    bool first = true;

    while (iss >> word) {
        std::string to_encode = first ? word : (" " + word);
        first = false;

        std::string bpe_str = bytes_to_bpe_string(to_encode);
        std::vector<std::string> subwords = bpe_encode_word(bpe_str, model->bpe_ranks);

        for (const auto& sw : subwords) {
            auto it = model->token_to_id.find(sw);
            if (it != model->token_to_id.end()) {
                tokens.push_back(it->second);
            }
        }
    }

    return tokens;
}

std::vector<int> tokenize(const State* state, const std::string& text) {
    if (!state || !state->model) return {};
    return tokenize_model(state->model, text);
}

static std::string build_hotwords_prompt(const std::string& hotwords) {
    if (hotwords.empty()) return "";
    
    std::vector<std::string> words;
    std::istringstream iss(hotwords);
    std::string word;
    while (std::getline(iss, word, ',')) {
        size_t start = word.find_first_not_of(" \t");
        size_t end = word.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos && start <= end) {
            words.push_back(word.substr(start, end - start + 1));
        }
    }
    
    if (words.empty()) return "";
    
    std::string prompt = "请注意识别以下词汇: ";
    for (size_t i = 0; i < words.size(); ++i) {
        prompt += words[i];
        if (i < words.size() - 1) prompt += "、";
    }
    
    return prompt;
}

static int find_audio_start_pos(const std::vector<int>& tokens, int audio_start_token) {
    for (int i = 0; i < (int)tokens.size(); ++i) {
        if (tokens[i] == audio_start_token) {
            return i + 1;
        }
    }
    return -1;
}

static int argmax(const std::vector<float>& logits) {
    if (logits.empty()) return -1;
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < (int)logits.size(); ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

static std::string extract_language_from_tokens_model(
    const Model* model,
    const std::vector<int>& tokens,
    const std::string& user_language) {
    
    if (!user_language.empty()) {
        std::string lang = user_language;
        for (auto& c : lang) c = std::tolower(static_cast<unsigned char>(c));
        return lang;
    }
    
    if (tokens.empty()) return "";
    
    std::string first_token = decode_token_model(model, tokens[0]);
    if (first_token != "language" && first_token.find("language") != 0) {
        return "";
    }
    
    std::string lang_part = first_token;
    int idx = 1;
    
    while (idx < (int)tokens.size()) {
        std::string tok = decode_token_model(model, tokens[idx]);
        if (tok.empty()) {
            idx++;
            continue;
        }
        
        bool is_alpha = true;
        for (char c : tok) {
            if (!std::isalpha(static_cast<unsigned char>(c)) && c != ' ' && c != '\t') {
                is_alpha = false;
                break;
            }
        }
        
        if (is_alpha) {
            lang_part += tok;
            idx++;
        } else {
            break;
        }
    }
    
    if (lang_part.find("language ") == 0) {
        std::string lang = lang_part.substr(9);
        size_t start = lang.find_first_not_of(" \t");
        size_t end = lang.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            lang = lang.substr(start, end - start + 1);
        }
        for (auto& c : lang) c = std::tolower(static_cast<unsigned char>(c));
        return lang;
    }
    
    return "";
}

static std::string extract_text_from_tokens_model(
    const Model* model,
    const std::vector<int>& tokens,
    bool user_specified_language) {
    
    if (user_specified_language) {
        return decode_tokens_model(model, tokens);
    }
    
    if (tokens.empty()) return "";
    
    std::string first_token = decode_token_model(model, tokens[0]);
    if (first_token != "language" && first_token.find("language") != 0) {
        return decode_tokens_model(model, tokens);
    }
    
    int tokens_to_skip = 1;
    int idx = 1;
    
    while (idx < (int)tokens.size()) {
        std::string tok = decode_token_model(model, tokens[idx]);
        if (tok.empty()) {
            idx++;
            tokens_to_skip++;
            continue;
        }
        
        bool is_alpha = true;
        for (char c : tok) {
            if (!std::isalpha(static_cast<unsigned char>(c)) && c != ' ' && c != '\t') {
                is_alpha = false;
                break;
            }
        }
        
        if (is_alpha) {
            tokens_to_skip++;
            idx++;
        } else {
            break;
        }
    }
    
    std::vector<int> text_tokens;
    for (int i = tokens_to_skip; i < (int)tokens.size(); ++i) {
        text_tokens.push_back(tokens[i]);
    }
    
    std::string text = decode_tokens_model(model, text_tokens);
    size_t start = text.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
        text = text.substr(start);
    }
    
    return text;
}

static std::string extract_language_from_tokens(
    const State* state,
    const std::vector<int>& tokens,
    const std::string& user_language) {
    
    if (!state || !state->model) return "";
    return extract_language_from_tokens_model(state->model, tokens, user_language);
}

static std::string extract_text_from_tokens(
    const State* state,
    const std::vector<int>& tokens,
    bool user_specified_language) {
    
    if (!state || !state->model) return "";
    return extract_text_from_tokens_model(state->model, tokens, user_specified_language);
}

std::vector<int> build_token_sequence(
    const State* state,
    int n_audio_frames,
    const std::string& language,
    const std::string& context,
    const std::string& hotwords,
    const std::string& prompt) {
    
    if (!state || !state->model) return {};
    
    const auto& hp = state->model->hparams;
    
    const int im_start = 151644;
    const int im_end = hp.eos_token;
    const int system_token = 8948;
    const int user_token = 872;
    const int assistant_token = 77091;
    const int newline = 198;
    
    std::vector<int> tokens;
    
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    
    if (!context.empty()) {
        std::vector<int> ctx_tokens = tokenize(state, context);
        for (int t : ctx_tokens) tokens.push_back(t);
    }
    
    std::string hw_prompt = build_hotwords_prompt(hotwords);
    if (!hw_prompt.empty()) {
        std::vector<int> hw_tokens = tokenize(state, hw_prompt);
        for (int t : hw_tokens) tokens.push_back(t);
    }
    
    if (!prompt.empty()) {
        std::vector<int> p_tokens = tokenize(state, prompt);
        for (int t : p_tokens) tokens.push_back(t);
    }
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    tokens.push_back(hp.audio_start_token);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(hp.audio_pad_token);
    }
    tokens.push_back(hp.audio_end_token);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    if (!language.empty()) {
        std::string lang_prompt = "language " + language + "\n";
        std::vector<int> lang_tokens = tokenize(state, lang_prompt);
        for (int t : lang_tokens) tokens.push_back(t);
    }
    
    return tokens;
}

bool transcribe(State* state, 
                const TranscribeInput& input, 
                TranscribeOutput& output, 
                ErrorInfo* error) {
    
    if (!state || !state->model) {
        if (error) error->message = "State not initialized";
        return false;
    }
    
    if (!input.audio_features || input.n_audio_frames <= 0) {
        if (error) error->message = "Invalid audio features";
        return false;
    }
    
    const auto& hp = state->model->hparams;
    const int im_end = hp.eos_token;
    
    clear_kv_cache(state);
    
    std::vector<int> tokens = build_token_sequence(
        state, input.n_audio_frames,
        input.language, input.context,
        input.hotwords, input.prompt);
    
    if (tokens.empty()) {
        if (error) error->message = "Failed to build token sequence";
        return false;
    }
    
    int audio_start_pos = find_audio_start_pos(tokens, hp.audio_start_token);
    if (audio_start_pos < 0) {
        if (error) error->message = "Failed to find audio_start_token in sequence";
        return false;
    }
    
    PrefillInput prefill_in;
    prefill_in.tokens = tokens.data();
    prefill_in.n_tokens = tokens.size();
    prefill_in.audio_features = input.audio_features;
    prefill_in.n_audio_frames = input.n_audio_frames;
    prefill_in.audio_feature_dim = input.audio_feature_dim;
    prefill_in.audio_start_pos = audio_start_pos;
    
    DecoderOutput prefill_out;
    if (!prefill(state, prefill_in, prefill_out, error)) {
        return false;
    }
    
    std::vector<int> generated_tokens;
    int n_past = tokens.size();
    int next_token = argmax(prefill_out.logits);
    
    if (next_token < 0) {
        if (error) error->message = "Invalid prefill output";
        return false;
    }
    
    if (next_token != im_end) {
        generated_tokens.push_back(next_token);
        
        for (int step = 1; step < input.max_tokens; ++step) {
            DecodeInput decode_in;
            decode_in.tokens = &next_token;
            decode_in.n_tokens = 1;
            decode_in.n_past = n_past;
            
            DecoderOutput decode_out;
            if (!decode(state, decode_in, decode_out, error)) {
                break;
            }
            
            next_token = argmax(decode_out.logits);
            if (next_token < 0 || next_token == im_end) {
                break;
            }
            
            generated_tokens.push_back(next_token);
            n_past++;
        }
    }
    
    output.tokens = generated_tokens;
    output.n_tokens = generated_tokens.size();
    
    bool user_specified_language = !input.language.empty();
    output.language = extract_language_from_tokens(state, generated_tokens, input.language);
    output.text = extract_text_from_tokens(state, generated_tokens, user_specified_language);
    
    return true;
}

static bool init_unified_kv_cache(BatchDecodeState* state, int total_capacity) {
    const auto& hp = state->model->hparams;
    
    state->unified_cache.total_capacity = total_capacity;
    state->unified_cache.head_dim = hp.head_dim;
    state->unified_cache.n_kv_heads = hp.n_kv_heads;
    state->unified_cache.n_layers = hp.n_layers;
    state->unified_cache.next_seq_id = 0;
    
    const size_t n_tensors = hp.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        ctx_size,
        nullptr,
        true
    };
    
    state->unified_cache.ctx = ggml_init(params);
    if (!state->unified_cache.ctx) {
        return false;
    }
    
    state->unified_cache.k_cache.resize(hp.n_layers);
    state->unified_cache.v_cache.resize(hp.n_layers);
    
    for (int il = 0; il < hp.n_layers; ++il) {
        state->unified_cache.k_cache[il] = ggml_new_tensor_3d(
            state->unified_cache.ctx, GGML_TYPE_F16,
            hp.head_dim, hp.n_kv_heads, total_capacity);
        ggml_format_name(state->unified_cache.k_cache[il], "batch_k_cache_%d", il);
        
        state->unified_cache.v_cache[il] = ggml_new_tensor_3d(
            state->unified_cache.ctx, GGML_TYPE_F16,
            hp.head_dim, hp.n_kv_heads, total_capacity);
        ggml_format_name(state->unified_cache.v_cache[il], "batch_v_cache_%d", il);
    }
    
    ggml_backend_t kv_backend = state->backend_gpu ? state->backend_gpu : state->backend_cpu;
    state->unified_cache.buffer = ggml_backend_alloc_ctx_tensors(state->unified_cache.ctx, kv_backend);
    if (!state->unified_cache.buffer) {
        return false;
    }
    
    return true;
}

static void free_unified_kv_cache(UnifiedCache& cache) {
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
    cache.slots.clear();
    cache.total_capacity = 0;
    cache.next_seq_id = 0;
}

static int find_free_slot_capacity(BatchDecodeState* state, int required_capacity) {
    int best_slot = -1;
    int best_capacity = -1;
    
    for (size_t i = 0; i < state->unified_cache.slots.size(); ++i) {
        auto& slot = state->unified_cache.slots[i];
        if (!slot.is_active && slot.capacity >= required_capacity) {
            if (best_slot < 0 || slot.capacity < best_capacity) {
                best_slot = static_cast<int>(i);
                best_capacity = slot.capacity;
            }
        }
    }
    
    return best_slot;
}

static int allocate_new_slot(BatchDecodeState* state, int required_capacity, ErrorInfo* error) {
    int total_used = 0;
    for (const auto& slot : state->unified_cache.slots) {
        total_used += slot.capacity;
    }
    
    if (total_used + required_capacity > state->unified_cache.total_capacity) {
        if (error) error->message = "Unified cache capacity exceeded";
        return -1;
    }
    
    int start_pos = total_used;
    
    SequenceSlot new_slot;
    new_slot.seq_id = state->unified_cache.next_seq_id++;
    new_slot.start_pos = start_pos;
    new_slot.n_used = 0;
    new_slot.capacity = required_capacity;
    new_slot.is_active = false;
    new_slot.next_token = -1;
    
    state->unified_cache.slots.push_back(new_slot);
    
    return static_cast<int>(state->unified_cache.slots.size() - 1);
}

BatchDecodeState* init_batch(const Config& config, int max_batch_size, int max_total_capacity) {
    BatchDecodeState* state = new BatchDecodeState();
    state->model = new Model();
    state->max_batch_size = max_batch_size;
    
    ErrorInfo err;
    if (!load_model_internal(config.model_path.c_str(), *state->model, &err)) {
        fprintf(stderr, "Failed to load decoder model: %s\n", err.message.c_str());
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state->backend_cpu) {
        free_decoder_model(*state->model);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    if (!config.device_name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(config.device_name.c_str());
        if (dev) {
            state->backend_gpu = ggml_backend_dev_init(dev, nullptr);
        }
    }
    if (!state->backend_gpu) {
        state->backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    
    std::vector<ggml_backend_t> backends;
    if (state->backend_gpu) {
        backends.push_back(state->backend_gpu);
    }
    backends.push_back(state->backend_cpu);
    
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto be : backends) {
        bufts.push_back(ggml_backend_get_default_buffer_type(be));
    }
    
    state->sched = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), MAX_NODES, false, true);
    if (!state->sched) {
        free_decoder_model(*state->model);
        if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
        ggml_backend_free(state->backend_cpu);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->compute_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    if (!init_unified_kv_cache(state, max_total_capacity)) {
        ggml_backend_sched_free(state->sched);
        if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
        ggml_backend_free(state->backend_cpu);
        free_decoder_model(*state->model);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    return state;
}

void free_batch(BatchDecodeState* state) {
    if (!state) return;
    
    free_unified_kv_cache(state->unified_cache);
    
    if (state->sched) {
        ggml_backend_sched_free(state->sched);
        state->sched = nullptr;
    }
    if (state->backend_gpu) {
        ggml_backend_free(state->backend_gpu);
        state->backend_gpu = nullptr;
    }
    if (state->backend_cpu) {
        ggml_backend_free(state->backend_cpu);
        state->backend_cpu = nullptr;
    }
    if (state->model) {
        free_decoder_model(*state->model);
        delete state->model;
        state->model = nullptr;
    }
    
    delete state;
}

static ggml_cgraph* build_prefill_to_slot_graph(
    BatchDecodeState* state,
    int slot_id,
    const int* tokens, int n_tokens,
    const float* audio_features, int n_audio_frames, int audio_start_pos) {
    
    const auto& hp = state->model->hparams;
    const int n_head = hp.n_heads;
    const int n_kv_head = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int hidden_size = hp.hidden_size;
    const float eps = hp.rms_norm_eps;
    const float rope_theta = hp.rope_theta;
    const int n_layer = hp.n_layers;
    
    auto& slot = state->unified_cache.slots[slot_id];
    int n_past = 0;
    
    struct ggml_init_params params = {
        state->compute_meta.size(),
        state->compute_meta.data(),
        true
    };
    
    struct ggml_context* ctx0 = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, MAX_NODES, false);
    
    struct ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor* inp_audio = nullptr;
    if (audio_features && n_audio_frames > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio_frames);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor* cur = ggml_get_rows(ctx0, state->model->token_embd, inp_tokens);
    
    if (inp_audio && n_audio_frames > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio_frames <= n_tokens) {
        struct ggml_tensor* embd_before = nullptr;
        struct ggml_tensor* embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio_frames < n_tokens) {
            int after_start = audio_start_pos + n_audio_frames;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            struct ggml_tensor* tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
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
    
    struct ggml_tensor* inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    int n_kv = n_past + n_tokens;
    struct ggml_tensor* fa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_name(fa_mask, "fa_mask");
    ggml_set_input(fa_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = state->model->layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v ||
            !layer.attn_out || !layer.ffn_norm || !layer.ffn_gate ||
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
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
        
        struct ggml_tensor* k_cache = state->unified_cache.k_cache[il];
        struct ggml_tensor* v_cache = state->unified_cache.v_cache[il];
        
        struct ggml_tensor* k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            slot.start_pos * k_cache->nb[2]);
        
        struct ggml_tensor* v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            slot.start_pos * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        struct ggml_tensor* K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2],
            slot.start_pos * k_cache->nb[2]);
        
        struct ggml_tensor* V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2],
            slot.start_pos * v_cache->nb[2]);
        
        struct ggml_tensor* Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, K, V, fa_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_out, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor* inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor* gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor* up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
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
    cur = ggml_mul(ctx0, cur, state->model->output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, state->model->output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

int batch_add_sequence(BatchDecodeState* state,
                       const std::vector<int>& prefill_tokens,
                       const float* audio_features,
                       int n_audio_frames,
                       int audio_feature_dim,
                       int audio_start_pos,
                       int max_tokens,
                       const std::string& language,
                       ErrorInfo* error) {
    
    if (!state || !state->model) {
        if (error) error->message = "State or model not initialized";
        return -1;
    }
    
    if (prefill_tokens.empty()) {
        if (error) error->message = "Empty prefill tokens";
        return -1;
    }
    
    int required_capacity = prefill_tokens.size() + max_tokens;
    
    int slot_id = find_free_slot_capacity(state, required_capacity);
    if (slot_id < 0) {
        slot_id = allocate_new_slot(state, required_capacity, error);
        if (slot_id < 0) {
            return -1;
        }
    }
    
    auto& slot = state->unified_cache.slots[slot_id];
    
    if (batch_get_n_active(state) >= state->max_batch_size) {
        if (error) error->message = "Max batch size reached";
        return -1;
    }
    
    struct ggml_cgraph* gf = build_prefill_to_slot_graph(
        state, slot_id,
        prefill_tokens.data(), prefill_tokens.size(),
        audio_features, n_audio_frames, audio_start_pos);
    
    if (!gf) {
        if (error) error->message = "Failed to build prefill graph";
        return -1;
    }
    
    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) {
        if (error) error->message = "Failed to allocate graph";
        return -1;
    }
    
    struct ggml_tensor* inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        if (error) error->message = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state->sched);
        return -1;
    }
    ggml_backend_tensor_set(inp_tokens, prefill_tokens.data(), 0, prefill_tokens.size() * sizeof(int));
    
    struct ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(prefill_tokens.size());
        for (size_t i = 0; i < prefill_tokens.size(); ++i) {
            positions[i] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    }
    
    struct ggml_tensor* fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = prefill_tokens.size();
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * prefill_tokens.size());
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (size_t q = 0; q < prefill_tokens.size(); ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask_data[k + q * n_kv] = (k <= (int)q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (audio_features && n_audio_frames > 0) {
        struct ggml_tensor* inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_features, 0,
                                    n_audio_frames * state->model->hparams.hidden_size * sizeof(float));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state->sched, gf) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "Failed to compute graph";
        ggml_backend_sched_reset(state->sched);
        return -1;
    }
    
    struct ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        if (error) error->message = "Failed to find logits tensor";
        ggml_backend_sched_reset(state->sched);
        return -1;
    }
    
    std::vector<float> logits_data(logits->ne[0]);
    ggml_backend_tensor_get(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    
    int next_token = argmax(logits_data);
    
    slot.n_used = static_cast<int>(prefill_tokens.size());
    slot.is_active = true;
    slot.next_token = next_token;
    slot.language = language;
    slot.generated_tokens.clear();
    
    if (next_token != state->model->hparams.eos_token) {
        slot.generated_tokens.push_back(next_token);
    }
    
    ggml_backend_sched_reset(state->sched);
    
    return slot_id;
}

void batch_remove_sequence(BatchDecodeState* state, int slot_id) {
    if (!state || slot_id < 0 || slot_id >= (int)state->unified_cache.slots.size()) return;
    
    state->unified_cache.slots[slot_id].is_active = false;
    state->unified_cache.slots[slot_id].n_used = 0;
    state->unified_cache.slots[slot_id].next_token = -1;
}

static ggml_cgraph* build_batch_decode_graph(
    BatchDecodeState* state,
    const std::vector<int>& active_slots) {
    
    const auto& hp = state->model->hparams;
    const int n_head = hp.n_heads;
    const int n_kv_head = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int hidden_size = hp.hidden_size;
    const float eps = hp.rms_norm_eps;
    const float rope_theta = hp.rope_theta;
    const int n_layer = hp.n_layers;
    
    int batch_size = active_slots.size();
    
    struct ggml_init_params params = {
        state->compute_meta.size(),
        state->compute_meta.data(),
        true
    };
    
    struct ggml_context* ctx0 = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, MAX_NODES, false);
    
    struct ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor* inp_n_past = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    ggml_set_name(inp_n_past, "inp_n_past");
    ggml_set_input(inp_n_past);
    
    struct ggml_tensor* cur = ggml_get_rows(ctx0, state->model->token_embd, inp_tokens);
    
    struct ggml_tensor* inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = state->model->layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v ||
            !layer.attn_out || !layer.ffn_norm || !layer.ffn_gate ||
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, batch_size);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, batch_size);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, batch_size);
        
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
        
        struct ggml_tensor* k_cache = state->unified_cache.k_cache[il];
        struct ggml_tensor* v_cache = state->unified_cache.v_cache[il];
        
        for (int bs = 0; bs < batch_size; ++bs) {
            auto& slot = state->unified_cache.slots[active_slots[bs]];
            
            struct ggml_tensor* k_cache_view = ggml_view_3d(ctx0, k_cache,
                head_dim, n_kv_head, 1,
                k_cache->nb[1], k_cache->nb[2],
                (slot.start_pos + slot.n_used) * k_cache->nb[2]);
            
            struct ggml_tensor* v_cache_view = ggml_view_3d(ctx0, v_cache,
                head_dim, n_kv_head, 1,
                v_cache->nb[1], v_cache->nb[2],
                (slot.start_pos + slot.n_used) * v_cache->nb[2]);
            
            struct ggml_tensor* Kcur_slice = ggml_view_3d(ctx0, Kcur,
                head_dim, n_kv_head, 1,
                Kcur->nb[1], Kcur->nb[2],
                bs * Kcur->nb[2]);
            
            struct ggml_tensor* Vcur_slice = ggml_view_3d(ctx0, Vcur,
                head_dim, n_kv_head, 1,
                Vcur->nb[1], Vcur->nb[2],
                bs * Vcur->nb[2]);
            
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur_slice, k_cache_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur_slice, v_cache_view));
        }
        
        struct ggml_tensor* attn_layer_out = nullptr;
        
        for (int bs = 0; bs < batch_size; ++bs) {
            auto& slot = state->unified_cache.slots[active_slots[bs]];
            int kv_len = slot.n_used + 1;
            
            struct ggml_tensor* Q_slot = ggml_view_3d(ctx0, Qcur,
                head_dim, n_head, 1,
                Qcur->nb[1], Qcur->nb[2],
                bs * Qcur->nb[2]);
            
            struct ggml_tensor* K_slot = ggml_view_3d(ctx0, k_cache,
                head_dim, n_kv_head, kv_len,
                k_cache->nb[1], k_cache->nb[2],
                slot.start_pos * k_cache->nb[2]);
            
            struct ggml_tensor* V_slot = ggml_view_3d(ctx0, v_cache,
                head_dim, n_kv_head, kv_len,
                v_cache->nb[1], v_cache->nb[2],
                slot.start_pos * v_cache->nb[2]);
            
            Q_slot = ggml_permute(ctx0, Q_slot, 0, 2, 1, 3);
            K_slot = ggml_permute(ctx0, K_slot, 0, 2, 1, 3);
            V_slot = ggml_permute(ctx0, V_slot, 0, 2, 1, 3);
            
            struct ggml_tensor* attn_result = ggml_flash_attn_ext(ctx0, Q_slot, K_slot, V_slot, nullptr, KQscale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_result, GGML_PREC_F32);
            attn_result = ggml_reshape_2d(ctx0, attn_result, n_head * head_dim, 1);
            
            attn_result = ggml_mul_mat(ctx0, layer.attn_out, attn_result);
            
            struct ggml_tensor* inpL_view = ggml_view_2d(ctx0, inpL,
                hidden_size, 1,
                inpL->nb[1],
                bs * inpL->nb[1]);
            
            attn_result = ggml_add(ctx0, attn_result, inpL_view);
            
            if (bs == 0) {
                attn_layer_out = ggml_cont(ctx0, attn_result);
            } else {
                attn_layer_out = ggml_concat(ctx0, attn_layer_out, attn_result, 1);
            }
        }
        
        cur = attn_layer_out;
        struct ggml_tensor* inpFF = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor* gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor* up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, state->model->output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, state->model->output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool batch_decode_step(BatchDecodeState* state, ErrorInfo* error) {
    if (!state || !state->model) {
        if (error) error->message = "State or model not initialized";
        return false;
    }
    
    std::vector<int> active_slots = batch_get_active_slots(state);
    if (active_slots.empty()) {
        return true;
    }
    
    std::vector<int32_t> tokens(active_slots.size());
    std::vector<int32_t> positions(active_slots.size());
    
    for (size_t i = 0; i < active_slots.size(); ++i) {
        auto& slot = state->unified_cache.slots[active_slots[i]];
        tokens[i] = slot.next_token;
        positions[i] = slot.n_used;
    }
    
    struct ggml_cgraph* gf = build_batch_decode_graph(state, active_slots);
    
    if (!gf) {
        if (error) error->message = "Failed to build batch decode graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) {
        if (error) error->message = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor* inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        if (error) error->message = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t));
    
    struct ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    }
    
    if (ggml_backend_sched_graph_compute(state->sched, gf) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "Failed to compute graph";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    struct ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        if (error) error->message = "Failed to find logits tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    int vocab_size = logits->ne[0];
    std::vector<float> all_logits(vocab_size * active_slots.size());
    ggml_backend_tensor_get(logits, all_logits.data(), 0, all_logits.size() * sizeof(float));
    
    const int eos_token = state->model->hparams.eos_token;
    
    for (size_t i = 0; i < active_slots.size(); ++i) {
        auto& slot = state->unified_cache.slots[active_slots[i]];
        
        std::vector<float> slot_logits(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            slot_logits[v] = all_logits[i * vocab_size + v];
        }
        
        int next_token = argmax(slot_logits);
        
        slot.n_used++;
        
        if (next_token == eos_token || slot.n_used >= slot.capacity) {
            slot.is_active = false;
            slot.next_token = -1;
        } else {
            slot.next_token = next_token;
            slot.generated_tokens.push_back(next_token);
        }
    }
    
    ggml_backend_sched_reset(state->sched);
    
    return true;
}

std::vector<int> batch_get_active_slots(BatchDecodeState* state) {
    if (!state) return {};
    
    std::vector<int> active;
    for (size_t i = 0; i < state->unified_cache.slots.size(); ++i) {
        if (state->unified_cache.slots[i].is_active) {
            active.push_back(static_cast<int>(i));
        }
    }
    return active;
}

std::vector<int> batch_get_eos_slots(BatchDecodeState* state) {
    if (!state) return {};
    
    std::vector<int> eos_slots;
    for (size_t i = 0; i < state->unified_cache.slots.size(); ++i) {
        auto& slot = state->unified_cache.slots[i];
        if (!slot.is_active && !slot.generated_tokens.empty()) {
            eos_slots.push_back(static_cast<int>(i));
        }
    }
    return eos_slots;
}

std::string batch_get_text(BatchDecodeState* state, int slot_id) {
    if (!state || !state->model || slot_id < 0 || 
        slot_id >= (int)state->unified_cache.slots.size()) return "";
    
    auto& slot = state->unified_cache.slots[slot_id];
    
    bool user_specified_language = !slot.language.empty();
    return extract_text_from_tokens_model(state->model, slot.generated_tokens, user_specified_language);
}

std::vector<int> batch_get_tokens(BatchDecodeState* state, int slot_id) {
    if (!state || slot_id < 0 || 
        slot_id >= (int)state->unified_cache.slots.size()) return {};
    
    return state->unified_cache.slots[slot_id].generated_tokens;
}

int batch_get_seq_id(BatchDecodeState* state, int slot_id) {
    if (!state || slot_id < 0 || 
        slot_id >= (int)state->unified_cache.slots.size()) return -1;
    
    return state->unified_cache.slots[slot_id].seq_id;
}

int batch_get_n_active(BatchDecodeState* state) {
    if (!state) return 0;
    
    int count = 0;
    for (const auto& slot : state->unified_cache.slots) {
        if (slot.is_active) count++;
    }
    return count;
}

int batch_get_capacity(BatchDecodeState* state) {
    if (!state) return 0;
    return state->unified_cache.total_capacity;
}

int batch_get_slot_capacity(BatchDecodeState* state, int slot_id) {
    if (!state || slot_id < 0 || 
        slot_id >= (int)state->unified_cache.slots.size()) return 0;
    
    return state->unified_cache.slots[slot_id].capacity;
}

int batch_get_slot_used(BatchDecodeState* state, int slot_id) {
    if (!state || slot_id < 0 || 
        slot_id >= (int)state->unified_cache.slots.size()) return 0;
    
    return state->unified_cache.slots[slot_id].n_used;
}

void batch_clear(BatchDecodeState* state) {
    if (!state) return;
    
    for (auto& slot : state->unified_cache.slots) {
        slot.is_active = false;
        slot.n_used = 0;
        slot.next_token = -1;
        slot.generated_tokens.clear();
    }
    
    if (state->unified_cache.buffer) {
        ggml_backend_buffer_clear(state->unified_cache.buffer, 0);
    }
}

HyperParams batch_get_hparams(BatchDecodeState* state) {
    if (!state || !state->model) return HyperParams();
    return state->model->hparams;
}

const char* batch_get_device_name(BatchDecodeState* state) {
    if (!state) return "CPU";
    return state->backend_gpu ? ggml_backend_name(state->backend_gpu) : "CPU";
}

std::vector<int> batch_build_token_sequence(
    BatchDecodeState* state,
    int n_audio_frames,
    const std::string& language,
    const std::string& context,
    const std::string& hotwords,
    const std::string& prompt) {
    
    if (!state || !state->model) return {};
    
    const auto& hp = state->model->hparams;
    
    const int im_start = 151644;
    const int im_end = hp.eos_token;
    const int system_token = 8948;
    const int user_token = 872;
    const int assistant_token = 77091;
    const int newline = 198;
    
    std::vector<int> tokens;
    
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    
    if (!context.empty()) {
        std::vector<int> ctx_tokens = tokenize_model(state->model, context);
        for (int t : ctx_tokens) tokens.push_back(t);
    }
    
    std::string hw_prompt = build_hotwords_prompt(hotwords);
    if (!hw_prompt.empty()) {
        std::vector<int> hw_tokens = tokenize_model(state->model, hw_prompt);
        for (int t : hw_tokens) tokens.push_back(t);
    }
    
    if (!prompt.empty()) {
        std::vector<int> p_tokens = tokenize_model(state->model, prompt);
        for (int t : p_tokens) tokens.push_back(t);
    }
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    tokens.push_back(hp.audio_start_token);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(hp.audio_pad_token);
    }
    tokens.push_back(hp.audio_end_token);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    if (!language.empty()) {
        std::string lang_prompt = "language " + language + "\n";
        std::vector<int> lang_tokens = tokenize_model(state->model, lang_prompt);
        for (int t : lang_tokens) tokens.push_back(t);
    }
    
    return tokens;
}

} // namespace decoder
} // namespace transcribe
} // namespace asr
