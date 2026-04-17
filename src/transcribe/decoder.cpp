#include "asr/transcribe/decoder.h"
#include "asr/transcribe/decoder_model.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

namespace qwen3_asr {
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

} // namespace decoder
} // namespace transcribe
} // namespace asr
} // namespace qwen3_asr