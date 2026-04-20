#include "asr/aligner/decoder.hpp"
#include "asr/aligner/decoder_model.hpp"
#include "asr/common/hf_tokenizer.hpp"
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
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace asr::aligner::decoder {

using asr::ErrorInfo;

constexpr int FA_MAX_NODES = 16384;

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
        if (idx < 0) return def;
        enum gguf_type vtype = gguf_get_kv_type(ctx_gguf, idx);
        if (vtype == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(ctx_gguf, idx);
        if (vtype == GGUF_TYPE_FLOAT64) return (float)gguf_get_val_f64(ctx_gguf, idx);
        return def;
    };
    
    model.hparams.vocab_size = get_u32("qwen3asr.llm.vocab_size", get_u32("qwen3-asr.vocab_size", 152064));
    model.hparams.hidden_size = get_u32("qwen3asr.llm.d_model", get_u32("qwen3-asr.embedding_length", 1024));
    model.hparams.n_layers = get_u32("qwen3asr.llm.n_layers", get_u32("qwen3-asr.block_count", 28));
    model.hparams.n_heads = get_u32("qwen3asr.llm.n_heads", get_u32("qwen3-asr.attention.head_count", 16));
    model.hparams.n_kv_heads = get_u32("qwen3asr.llm.n_kv_heads", get_u32("qwen3-asr.attention.head_count_kv", 8));
    model.hparams.intermediate_size = get_u32("qwen3asr.llm.ff_dim", get_u32("qwen3-asr.feed_forward_length", 3072));
    model.hparams.head_dim = get_u32("qwen3asr.llm.head_dim", get_u32("qwen3-asr.attention.key_length", 128));
    model.hparams.rms_norm_eps = get_f32("qwen3asr.llm.rms_norm_eps", get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f));
    model.hparams.rope_theta = get_f32("qwen3asr.llm.rope_theta", get_f32("qwen3-asr.rope.freq_base", 1000000.0f));
    model.hparams.classify_head_size = get_u32("qwen3asr.llm.classify_num", get_u32("qwen3-asr.classify_num", 5000));
    
    model.hparams.timestamp_token_id = get_u32("qwen3asr.timestamp_token_id", get_u32("qwen3-asr.timestamp_token_id", 151705));
    model.hparams.audio_start_token_id = get_u32("qwen3asr.audio_start_token_id", get_u32("qwen3_asr.audio_start_token_id", get_u32("qwen3-asr.audio.start_token_id", 151669)));
    model.hparams.audio_end_token_id = get_u32("qwen3asr.audio_end_token_id", get_u32("qwen3_asr.audio_end_token_id", get_u32("qwen3-asr.audio.end_token_id", 151670)));
    model.hparams.audio_pad_token_id = get_u32("qwen3asr.audio_pad_token_id", get_u32("qwen3_asr.audio_token_id", get_u32("qwen3-asr.audio.pad_token_id", 151676)));
    model.hparams.pad_token_id = get_u32("qwen3asr.pad_token_id", 151643);
    model.hparams.eos_token_id = get_u32("qwen3asr.eos_token_id", 151645);
    
    model.layers.resize(model.hparams.n_layers);
    
    for (ggml_tensor* t = ggml_get_first_tensor(model.ctx); t; t = ggml_get_next_tensor(model.ctx, t)) {
        const char* name = ggml_get_name(t);
        if (strstr(name, "audio.")) continue;
        model.tensors[name] = t;
        
        if (strstr(name, "blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && layer_idx >= 0 && layer_idx < model.hparams.n_layers) {
                auto& layer = model.layers[layer_idx];
                if (strstr(name, "attn_output.weight")) layer.attn_output = t;
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
        } else if (strstr(name, "token_embd.weight")) model.token_embd = t;
        else if (strstr(name, "output_norm.weight")) model.output_norm = t;
        else if (strstr(name, "output.weight")) model.classify_head_w = t;
        else if (strstr(name, "output.bias")) model.classify_head_b = t;
    }
    
    int64_t tokens_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (tokens_idx >= 0) {
        int64_t n_vocab = gguf_get_arr_n(ctx_gguf, tokens_idx);
        model.vocab.resize(n_vocab);
        for (int64_t i = 0; i < n_vocab; ++i) model.vocab[i] = gguf_get_arr_str(ctx_gguf, tokens_idx, i);
        for (int64_t i = 0; i < n_vocab; ++i) {
            if (!model.vocab[i].empty()) model.token_to_id[model.vocab[i]] = static_cast<int32_t>(i);
        }
        fprintf(stderr, "Loaded vocabulary: %lld tokens, token_to_id size: %zu\n", (long long)n_vocab, model.token_to_id.size());
    } else {
        fprintf(stderr, "Warning: tokenizer.ggml.tokens not found in model\n");
    }
    
    int64_t merges_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        int64_t n_merges = gguf_get_arr_n(ctx_gguf, merges_idx);
        for (int64_t i = 0; i < n_merges; ++i) {
            std::string merge = gguf_get_arr_str(ctx_gguf, merges_idx, i);
            model.bpe_ranks[merge] = static_cast<int>(i);
        }
        fprintf(stderr, "Loaded BPE merges: %lld\n", (long long)n_merges);
    } else {
        fprintf(stderr, "Warning: tokenizer.ggml.merges not found in model\n");
    }

    // Fallback: parse tokenizer from tokenizer.huggingface.json
    if (model.vocab.empty() || model.bpe_ranks.empty()) {
        int64_t hf_idx = gguf_find_key(ctx_gguf, "tokenizer.huggingface.json");
        if (hf_idx >= 0) {
            const char* hf_json = gguf_get_val_str(ctx_gguf, hf_idx);
            if (hf_json) {
                asr::HfTokenizerData hf_data;
                if (asr::load_tokenizer_from_hf_json(hf_json, model.hparams.vocab_size, hf_data)) {
                    model.vocab = std::move(hf_data.vocab);
                    model.token_to_id.insert(hf_data.token_to_id.begin(), hf_data.token_to_id.end());
                    model.bpe_ranks.insert(hf_data.bpe_ranks.begin(), hf_data.bpe_ranks.end());
                }
            }
        }
    }
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) { if (error) error->message = "Cannot mmap file"; gguf_free(ctx_gguf); return false; }
    
    struct stat st;
    fstat(fd, &st);
    void* mmap_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mmap_addr == MAP_FAILED) { if (error) error->message = "mmap failed"; gguf_free(ctx_gguf); return false; }
    
    model.mmap_addr = mmap_addr;
    model.mmap_size = st.st_size;
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    uint8_t* data_base = (uint8_t*)mmap_addr + data_offset;
    
    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, get_preferred_buft());
    if (!model.buffer) { if (error) error->message = "Failed to allocate tensor buffer"; munmap(mmap_addr, st.st_size); gguf_free(ctx_gguf); return false; }
    
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        if (strstr(name, "audio.")) continue;
        ggml_tensor* t = ggml_get_tensor(model.ctx, name);
        if (t) ggml_backend_tensor_set(t, data_base + gguf_get_tensor_offset(ctx_gguf, i), 0, ggml_nbytes(t));
    }
    
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev && strstr(ggml_backend_dev_name(gpu_dev), "Metal") == nullptr) {
        munmap(mmap_addr, st.st_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
    }
    
    gguf_free(ctx_gguf);
    fprintf(stderr, "Aligner decoder hparams: vocab=%d, hidden=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, ff=%d, rope_theta=%f, rms_eps=%f, classify=%d\n",
            model.hparams.vocab_size, model.hparams.hidden_size, model.hparams.n_layers,
            model.hparams.n_heads, model.hparams.n_kv_heads, model.hparams.head_dim,
            model.hparams.intermediate_size, model.hparams.rope_theta, model.hparams.rms_norm_eps,
            model.hparams.classify_head_size);
    fprintf(stderr, "Aligner decoder model loaded: %zu bytes, classify_head_size=%d\n", ggml_backend_buffer_get_size(model.buffer), model.hparams.classify_head_size);
    return true;
}

static void free_decoder_model(Model& model) {
    if (model.buffer) { ggml_backend_buffer_free(model.buffer); model.buffer = nullptr; }
    if (model.ctx) { ggml_free(model.ctx); model.ctx = nullptr; }
    if (model.mmap_addr) { munmap(model.mmap_addr, model.mmap_size); model.mmap_addr = nullptr; model.mmap_size = 0; }
    model.layers.clear();
    model.tensors.clear();
    model.vocab.clear();
    model.token_to_id.clear();
    model.bpe_ranks.clear();
    model.ko_dict.clear();
}

State* init(const Config& config) {
    State* state = new State();
    state->model = new Model();
    
    ErrorInfo err;
    if (!load_model_internal(config.model_path.c_str(), *state->model, &err)) {
        fprintf(stderr, "Failed to load aligner decoder model: %s\n", err.message.c_str());
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state->backend_cpu) { free_decoder_model(*state->model); delete state->model; delete state; return nullptr; }
    
    if (!config.device_name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(config.device_name.c_str());
        if (dev) state->backend_gpu = ggml_backend_dev_init(dev, nullptr);
    }
    if (!state->backend_gpu) state->backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    
    std::vector<ggml_backend_t> backends;
    if (state->backend_gpu) backends.push_back(state->backend_gpu);
    backends.push_back(state->backend_cpu);
    
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto be : backends) bufts.push_back(ggml_backend_get_default_buffer_type(be));
    
    state->sched = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), FA_MAX_NODES, false, true);
    if (!state->sched) {
        free_decoder_model(*state->model);
        if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
        ggml_backend_free(state->backend_cpu);
        delete state->model;
        delete state;
        return nullptr;
    }
    
    state->compute_meta.resize(ggml_tensor_overhead() * FA_MAX_NODES + ggml_graph_overhead());
    return state;
}

void free(State* state) {
    if (!state) return;
    if (state->sched) { ggml_backend_sched_free(state->sched); state->sched = nullptr; }
    if (state->backend_gpu) { ggml_backend_free(state->backend_gpu); state->backend_gpu = nullptr; }
    if (state->backend_cpu) { ggml_backend_free(state->backend_cpu); state->backend_cpu = nullptr; }
    if (state->model) { free_decoder_model(*state->model); delete state->model; state->model = nullptr; }
    delete state;
}

const char* get_device_name(State* state) {
    return state && state->backend_gpu ? ggml_backend_name(state->backend_gpu) : "CPU";
}

HyperParams get_hparams(State* state) {
    return state && state->model ? state->model->hparams : HyperParams();
}

// ==================== Decoder Graph ====================
// Ported directly from legacy ForcedAligner::build_decoder_graph

static ggml_cgraph* build_decoder_graph(
    State* state,
    const int32_t* tokens, int n_tokens,
    const float* audio_embd, int n_audio,
    int audio_start_pos) {
    
    (void)tokens;
    
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
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, FA_MAX_NODES, false);
    
    struct ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor* inp_audio = nullptr;
    if (audio_embd && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor* cur = ggml_get_rows(ctx0, state->model->token_embd, inp_tokens);
    
    if (inp_audio && n_audio > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        struct ggml_tensor* embd_before = nullptr;
        struct ggml_tensor* embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos, cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio < n_tokens) {
            int after_start = audio_start_pos + n_audio;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len, cur->nb[1], after_start * cur->nb[1]);
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
    }
    
    struct ggml_tensor* inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    struct ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = state->model->layers[il];
        
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
        
        struct ggml_tensor* Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor* Kfa = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor* Vfa = ggml_cast(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3), GGML_TYPE_F16);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, Kfa, Vfa, causal_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor* inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
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
    
    cur = ggml_mul_mat(ctx0, state->model->classify_head_w, cur);
    if (state->model->classify_head_b) cur = ggml_add(ctx0, cur, state->model->classify_head_b);
    
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    struct ggml_tensor* argmax_result = ggml_argmax(ctx0, cur);
    ggml_set_name(argmax_result, "argmax_result");
    ggml_set_output(argmax_result);
    
    ggml_build_forward_expand(gf, argmax_result);
    
    ggml_free(ctx0);
    
    return gf;
}

// Ported directly from legacy ForcedAligner::forward_decoder

bool decode(State* state, const Input& input, Output& output, ErrorInfo* error) {
    if (!state || !state->model) {
        if (error) error->message = "State or model not initialized";
        return false;
    }
    
    struct ggml_cgraph* gf = build_decoder_graph(
        state, input.tokens, input.n_tokens,
        input.audio_features, input.n_audio_frames, input.audio_start_pos);
    
    if (!gf) {
        if (error) error->message = "Failed to build decoder graph";
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
    ggml_backend_tensor_set(inp_tokens, input.tokens, 0, input.n_tokens * sizeof(int32_t));
    
    struct ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(input.n_tokens);
        for (int i = 0; i < input.n_tokens; ++i) positions[i] = i;
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, input.n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    if (mask_t) {
        std::vector<ggml_fp16_t> mask_data((size_t)input.n_tokens * input.n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < input.n_tokens; ++q) {
            for (int k = 0; k < input.n_tokens; ++k) {
                mask_data[k + q * input.n_tokens] = (k <= q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
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
    struct ggml_tensor* argmax_t = ggml_graph_get_tensor(gf, "argmax_result");
    if (!logits || !argmax_t) {
        if (error) error->message = "Failed to find logits/argmax tensor";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    int64_t n_classes = logits->ne[0];
    output.n_classes = n_classes;
    output.timestamp_indices.resize(input.n_tokens);
    ggml_backend_tensor_get(argmax_t, output.timestamp_indices.data(), 0, input.n_tokens * sizeof(int32_t));
    
    ggml_backend_sched_reset(state->sched);
    
    return true;
}

TimestampResult convert_to_timestamps(const Output& output, int timestamp_segment_time_ms) {
    TimestampResult result;
    result.n_words = output.timestamp_indices.size();
    result.timestamps.resize(output.timestamp_indices.size());
    for (size_t i = 0; i < output.timestamp_indices.size(); ++i) {
        result.timestamps[i] = output.timestamp_indices[i] * (timestamp_segment_time_ms / 1000.0f);
    }
    return result;
}

// ==================== BPE Tokenizer ====================
// Ported directly from legacy code

static const std::vector<std::string>& get_byte_to_unicode_table() {
    static std::vector<std::string> table;
    if (!table.empty()) return table;
    table.resize(256);
    
    std::vector<int> byte_to_cp(256, 0);
    std::vector<bool> assigned(256, false);
    
    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) { byte_to_cp[b] = b; assigned[b] = true; }
    };
    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);
    
    int n = 0;
    for (int b = 0; b < 256; ++b) { if (!assigned[b]) { byte_to_cp[b] = 256 + n; ++n; } }
    
    auto cp_to_utf8 = [](int cp) -> std::string {
        std::string s;
        if (cp < 0x80) { s += static_cast<char>(cp); }
        else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
        return s;
    };
    
    for (int b = 0; b < 256; ++b) table[b] = cp_to_utf8(byte_to_cp[b]);
    return table;
}

static std::string bytes_to_bpe_string(const std::string& text) {
    const auto& table = get_byte_to_unicode_table();
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) result += table[c];
    return result;
}

static std::vector<std::string> split_utf8_chars(const std::string& s) {
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

static std::vector<std::string> bpe_encode_word(const std::string& word_bpe, const std::unordered_map<std::string, int>& bpe_ranks) {
    std::vector<std::string> symbols = split_utf8_chars(word_bpe);
    if (symbols.size() <= 1) return symbols;
    
    while (true) {
        int best_rank = INT_MAX;
        size_t best_pos = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = symbols[i] + " " + symbols[i + 1];
            auto it = bpe_ranks.find(key);
            if (it != bpe_ranks.end() && it->second < best_rank) { best_rank = it->second; best_pos = i; }
        }
        if (best_rank == INT_MAX) break;
        
        std::string merged = symbols[best_pos] + symbols[best_pos + 1];
        std::vector<std::string> new_symbols;
        new_symbols.reserve(symbols.size() - 1);
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i == best_pos) { new_symbols.push_back(merged); ++i; }
            else new_symbols.push_back(symbols[i]);
        }
        symbols = std::move(new_symbols);
        if (symbols.size() == 1) break;
    }
    return symbols;
}

// ==================== Text Processing ====================
// Ported directly from legacy code

static uint32_t utf8_to_codepoint(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0;
    if ((c & 0x80) == 0) { cp = c; i += 1; }
    else if ((c & 0xE0) == 0xC0) {
        if (i + 1 < s.size()) { cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F); i += 2; }
        else { i += 1; }
    } else if ((c & 0xF0) == 0xE0) {
        if (i + 2 < s.size()) { cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) | (static_cast<unsigned char>(s[i + 2]) & 0x3F); i += 3; }
        else { i += 1; }
    } else if ((c & 0xF8) == 0xF0) {
        if (i + 3 < s.size()) { cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) | ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(s[i + 3]) & 0x3F); i += 4; }
        else { i += 1; }
    } else { i += 1; }
    return cp;
}

static bool is_cjk_char(uint32_t code) {
    return (0x4E00 <= code && code <= 0x9FFF) || (0x3400 <= code && code <= 0x4DBF) ||
           (0x20000 <= code && code <= 0x2A6DF) || (0x2A700 <= code && code <= 0x2B73F) ||
           (0x2B740 <= code && code <= 0x2B81F) || (0x2B820 <= code && code <= 0x2CEAF) ||
           (0xF900 <= code && code <= 0xFAFF);
}

static bool is_kept_char(uint32_t code) {
    if (code == '\'') return true;
    if (code < 0x80) {
        if (('A' <= code && code <= 'Z') || ('a' <= code && code <= 'z') || ('0' <= code && code <= '9')) return true;
        if (code == '.' || code == '!' || code == '?') return true;
        return false;
    }
    if (code == 0x3002) return true;
    if (code == 0xFF01) return true;
    if (code == 0xFF1F) return true;
    if (0x4E00 <= code && code <= 0x9FFF) return true;
    if (0x3400 <= code && code <= 0x4DBF) return true;
    if (0xAC00 <= code && code <= 0xD7AF) return true;
    if (0x3040 <= code && code <= 0x30FF) return true;
    return false;
}

static std::string clean_token(const std::string& token) {
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        size_t start = i;
        uint32_t cp = utf8_to_codepoint(token, i);
        if (is_kept_char(cp)) result += token.substr(start, i - start);
    }
    return result;
}

static bool is_end_punctuation(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F;
}

static std::vector<std::string> split_segment_with_cjk(const std::string& seg) {
    std::vector<std::string> tokens;
    std::string buf;
    size_t i = 0;
    while (i < seg.size()) {
        size_t start = i;
        uint32_t cp = utf8_to_codepoint(seg, i);
        if (is_cjk_char(cp)) {
            if (!buf.empty()) { std::string cleaned = clean_token(buf); if (!cleaned.empty()) tokens.push_back(cleaned); buf.clear(); }
            std::string cjk_char = seg.substr(start, i - start);
            std::string cleaned = clean_token(cjk_char);
            if (!cleaned.empty()) tokens.push_back(cleaned);
        } else if (is_end_punctuation(cp)) {
            if (!buf.empty()) { std::string cleaned = clean_token(buf); if (!cleaned.empty()) tokens.push_back(cleaned); buf.clear(); }
            tokens.push_back(seg.substr(start, i - start));
        } else {
            buf += seg.substr(start, i - start);
        }
    }
    if (!buf.empty()) { std::string cleaned = clean_token(buf); if (!cleaned.empty()) tokens.push_back(cleaned); }
    return tokens;
}

static std::vector<std::string> tokenize_space_lang(const std::string& text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
        if (i >= text.size()) break;
        size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r') ++i;
        std::string seg = text.substr(start, i - start);
        std::string cleaned_seg = clean_token(seg);
        if (!cleaned_seg.empty()) {
            auto sub_tokens = split_segment_with_cjk(cleaned_seg);
            for (const auto& t : sub_tokens) tokens.push_back(t);
        }
    }
    return tokens;
}

static size_t utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static size_t utf8_strlen(const std::string& s) {
    size_t count = 0;
    size_t i = 0;
    while (i < s.size()) { i += utf8_char_len(static_cast<unsigned char>(s[i])); ++count; }
    return count;
}

static std::string utf8_substr(const std::string& s, size_t char_start, size_t char_count) {
    size_t byte_start = 0;
    for (size_t c = 0; c < char_start && byte_start < s.size(); ++c) byte_start += utf8_char_len(static_cast<unsigned char>(s[byte_start]));
    size_t byte_end = byte_start;
    for (size_t c = 0; c < char_count && byte_end < s.size(); ++c) byte_end += utf8_char_len(static_cast<unsigned char>(s[byte_end]));
    return s.substr(byte_start, byte_end - byte_start);
}

static std::vector<std::string> tokenize_korean(const std::string& text, const std::unordered_set<std::string>& ko_dict) {
    std::vector<std::string> whitespace_words;
    {
        size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
            if (i >= text.size()) break;
            size_t start = i;
            while (i < text.size() && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r') ++i;
            whitespace_words.push_back(text.substr(start, i - start));
        }
    }
    
    std::vector<std::string> result;
    for (const auto& word : whitespace_words) {
        size_t length = utf8_strlen(word);
        if (length <= 2) { result.push_back(word); continue; }
        
        float best_score = -1e9f;
        size_t best_left_len = 0;
        std::string best_left;
        std::string best_right;
        
        for (size_t e = 2; e <= length; ++e) {
            std::string left = utf8_substr(word, 0, e);
            std::string right = utf8_substr(word, e, length - e);
            float score = ko_dict.count(left) ? 1.0f : 0.0f;
            if (score > best_score || (score == best_score && e > best_left_len)) { best_score = score; best_left_len = e; best_left = left; best_right = right; }
        }
        result.push_back(best_left);
        if (!best_right.empty()) result.push_back(best_right);
    }
    return result;
}

std::vector<int32_t> tokenize_with_timestamps(State* state, const std::string& text, std::vector<std::string>& words, const std::string& language) {
    if (!state || !state->model) return {};
    
    words.clear();
    std::vector<int32_t> tokens;
    const int32_t ts_token = state->model->hparams.timestamp_token_id;
    
    std::vector<std::string> raw_words;
    if (language == "korean" && !state->model->ko_dict.empty()) {
        raw_words = tokenize_korean(text, state->model->ko_dict);
    } else {
        raw_words = tokenize_space_lang(text);
    }
    
    for (size_t w = 0; w < raw_words.size(); ++w) {
        words.push_back(raw_words[w]);
        
        std::string bpe_str = bytes_to_bpe_string(raw_words[w]);
        std::vector<std::string> subwords = bpe_encode_word(bpe_str, state->model->bpe_ranks);
        
        for (const auto& sw : subwords) {
            auto it = state->model->token_to_id.find(sw);
            if (it != state->model->token_to_id.end()) tokens.push_back(it->second);
        }
        
        tokens.push_back(ts_token);
        tokens.push_back(ts_token);
    }
    
    return tokens;
}

bool load_korean_dict(State* state, const std::string& dict_path) {
    if (!state || !state->model) return false;
    std::ifstream f(dict_path);
    if (!f.is_open()) return false;
    state->model->ko_dict.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t pos = line.find(' ');
        std::string word = (pos != std::string::npos) ? line.substr(0, pos) : line;
        if (!word.empty()) state->model->ko_dict.insert(word);
    }
    return true;
}

// ==================== Timestamp Processing ====================
// Ported directly from legacy code

static int32_t get_feat_extract_output_lengths(int32_t input_lengths) {
    int32_t input_lengths_leave = input_lengths % 100;
    int32_t feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    int32_t output_lengths = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (input_lengths / 100) * 13;
    return output_lengths;
}

std::vector<int32_t> build_token_sequence(State* state, int n_audio_pads, const std::vector<int32_t>& text_tokens) {
    if (!state || !state->model) return {};
    const auto& hp = state->model->hparams;
    std::vector<int32_t> tokens;
    tokens.push_back(hp.audio_start_token_id);
    for (int i = 0; i < n_audio_pads; ++i) tokens.push_back(hp.audio_pad_token_id);
    tokens.push_back(hp.audio_end_token_id);
    for (auto tok : text_tokens) tokens.push_back(tok);
    return tokens;
}

std::vector<int32_t> extract_timestamp_classes(const Output& output, const std::vector<int32_t>& tokens, int timestamp_token_id) {
    std::vector<int32_t> classes;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == timestamp_token_id) {
            classes.push_back(output.timestamp_indices[i]);
        }
    }
    return classes;
}

std::vector<int32_t> fix_timestamp_classes(const std::vector<int32_t>& data) {
    const int n = static_cast<int>(data.size());
    if (n == 0) return {};
    
    // O(n log n) LIS using patience sorting
    std::vector<int> tails;
    std::vector<int> tails_idx;
    std::vector<int> parent(n, -1);
    
    for (int i = 0; i < n; ++i) {
        auto it = std::lower_bound(tails.begin(), tails.end(), data[i]);
        int pos = static_cast<int>(it - tails.begin());
        
        if (it != tails.end()) {
            *it = data[i];
            tails_idx[pos] = i;
        } else {
            tails.push_back(data[i]);
            tails_idx.push_back(i);
        }
        
        if (pos > 0) parent[i] = tails_idx[pos - 1];
    }
    
    std::vector<bool> is_normal(n, false);
    { int idx = tails_idx.empty() ? 0 : tails_idx.back(); while (idx != -1) { is_normal[idx] = true; idx = parent[idx]; } }
    
    std::vector<int32_t> result(data.begin(), data.end());
    int i = 0;
    
    while (i < n) {
        if (!is_normal[i]) {
            int j = i;
            while (j < n && !is_normal[j]) ++j;
            int anomaly_count = j - i;
            
            int32_t left_val = -1;
            for (int k = i - 1; k >= 0; --k) { if (is_normal[k]) { left_val = result[k]; break; } }
            
            int32_t right_val = -1;
            for (int k = j; k < n; ++k) { if (is_normal[k]) { right_val = result[k]; break; } }
            
            if (anomaly_count <= 2) {
                for (int k = i; k < j; ++k) {
                    if (left_val < 0) result[k] = right_val;
                    else if (right_val < 0) result[k] = left_val;
                    else result[k] = ((k - (i - 1)) <= (j - k)) ? left_val : right_val;
                }
            } else {
                if (left_val >= 0 && right_val >= 0) {
                    float step = static_cast<float>(right_val - left_val) / (anomaly_count + 1);
                    for (int k = i; k < j; ++k) result[k] = static_cast<int32_t>(left_val + step * (k - i + 1));
                } else if (left_val >= 0) {
                    for (int k = i; k < j; ++k) result[k] = left_val;
                } else if (right_val >= 0) {
                    for (int k = i; k < j; ++k) result[k] = right_val;
                }
            }
            
            i = j;
        } else { ++i; }
    }
    
    return result;
}

std::vector<float> classes_to_timestamps(const std::vector<int32_t>& classes, float segment_time_sec) {
    std::vector<float> timestamps;
    timestamps.reserve(classes.size());
    for (int32_t cls : classes) timestamps.push_back(cls * segment_time_sec);
    return timestamps;
}

// ==================== High-level align ====================
// Ported directly from legacy ForcedAligner::align

bool align(State* state, const AlignInput& input, AlignOutput& output, ErrorInfo* error) {
    if (!state || !state->model) { if (error) error->message = "State or model not initialized"; return false; }
    if (!input.audio_features || input.n_audio_frames <= 0) { if (error) error->message = "Invalid audio features"; return false; }
    if (input.text.empty()) { if (error) error->message = "Text is empty"; return false; }
    
    std::vector<std::string> words;
    std::vector<int32_t> text_tokens = tokenize_with_timestamps(state, input.text, words, input.language);
    if (text_tokens.empty() || words.empty()) { if (error) error->message = "Failed to tokenize text"; return false; }
    
    const auto& hp = state->model->hparams;
    
    int32_t n_audio_pads = input.n_mel_frames > 0 ? get_feat_extract_output_lengths(input.n_mel_frames) : input.n_audio_frames;
    std::vector<int32_t> tokens = build_token_sequence(state, n_audio_pads, text_tokens);
    
    int audio_start_pos = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == hp.audio_start_token_id) { audio_start_pos = i + 1; break; }
    }
    
    Input dec_input;
    dec_input.tokens = tokens.data();
    dec_input.n_tokens = tokens.size();
    dec_input.audio_features = input.audio_features;
    dec_input.n_audio_frames = input.n_audio_frames;
    dec_input.audio_start_pos = audio_start_pos;
    
    Output dec_output;
    if (!decode(state, dec_input, dec_output, error)) return false;
    
    float segment_time = hp.timestamp_segment_time_ms / 1000.0f;
    float audio_duration = (input.n_samples > 0 && input.sample_rate > 0)
        ? static_cast<float>(input.n_samples) / input.sample_rate
        : input.n_audio_frames * segment_time;
    
    std::vector<int32_t> timestamp_classes = extract_timestamp_classes(dec_output, tokens, hp.timestamp_token_id);
    std::vector<int32_t> fixed_classes = fix_timestamp_classes(timestamp_classes);
    std::vector<float> timestamps = classes_to_timestamps(fixed_classes, segment_time);
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (timestamps[i] > audio_duration) timestamps[i] = audio_duration;
    }
    
    output.words.clear();
    output.audio_duration = audio_duration;
    
    for (size_t w = 0; w < words.size(); ++w) {
        size_t start_idx = w * 2;
        size_t end_idx = w * 2 + 1;
        
        AlignedWord aw;
        aw.word = words[w];
        aw.start = (start_idx < timestamps.size()) ? timestamps[start_idx] : 0.0f;
        aw.end = (end_idx < timestamps.size()) ? timestamps[end_idx] : audio_duration;
        aw.conf_start = 0.0f;
        aw.conf_end = 0.0f;
        
        if (aw.end < aw.start) aw.end = aw.start;
        
        output.words.push_back(aw);
    }
    
    output.success = true;
    return true;
}

// ==================== Utility ====================

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
    if (a.size() != b.size()) { if (verbose) fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size()); return false; }
    float max_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) max_diff = std::max(max_diff, fabsf(a[i] - b[i]));
    if (max_diff > tolerance) { if (verbose) fprintf(stderr, "Max diff: %.6f > %.6f\n", max_diff, tolerance); return false; }
    return true;
}

} // namespace asr::aligner::decoder