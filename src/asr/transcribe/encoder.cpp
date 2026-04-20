#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/encoder_model.hpp"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

namespace asr { namespace transcribe { namespace encoder {

using asr::AudioFeatures;
using asr::ErrorInfo;

constexpr int MAX_NODES = 4096;

static void compute_sinusoidal_pe(float* pe, int n_ctx, int d_model) {
    const int half_dim = d_model / 2;
    for (int pos = 0; pos < n_ctx; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float div_term = expf(-logf(10000.0f) * i / (half_dim - 1));
            float angle = pos * div_term;
            pe[pos * d_model + i] = sinf(angle);
            pe[pos * d_model + half_dim + i] = cosf(angle);
        }
    }
}

static int compute_conv_output_len(int input_len) {
    int len = input_len;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    return len;
}

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

bool load_model(const char* path, EncoderModel& model, ErrorInfo* error) {
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
    
    model.hparams.n_encoder_layers = get_u32("qwen3_asr.audio.encoder_layers", get_u32("audio.encoder_layers", 18));
    model.hparams.d_model = get_u32("qwen3_asr.audio.d_model", get_u32("audio.d_model", 896));
    model.hparams.n_attention_heads = get_u32("qwen3_asr.audio.encoder_attention_heads", get_u32("audio.attention_heads", 14));
    model.hparams.n_mel_bins = get_u32("qwen3_asr.audio.num_mel_bins", get_u32("audio.num_mel_bins", 128));
    model.hparams.hidden_size = get_u32("qwen3_asr.audio.output_dim", get_u32("audio.output_dim", get_u32("text.hidden_size", 1024)));
    model.hparams.head_dim = model.hparams.d_model / model.hparams.n_attention_heads;
    model.hparams.ff_dim = get_u32("qwen3_asr.audio.encoder_ffn_dim", get_u32("audio.ffn_dim", 3584));
    model.hparams.conv_channels = get_u32("qwen3_asr.audio.conv_channels", get_u32("audio.conv_channels", 480));
    model.hparams.conv_out_dim = get_u32("qwen3_asr.audio.conv_out_dim", get_u32("audio.conv_out_dim", model.hparams.d_model));
    
    model.layers.resize(model.hparams.n_encoder_layers);
    
    for (ggml_tensor* t = ggml_get_first_tensor(model.ctx); t; t = ggml_get_next_tensor(model.ctx, t)) {
        const char* name = ggml_get_name(t);
        model.tensors[name] = t;
        
        if (strstr(name, "conv2d1.weight") || strstr(name, "conv1.weight")) model.conv1_w = t;
        else if (strstr(name, "conv2d1.bias") || strstr(name, "conv1.bias")) model.conv1_b = t;
        else if (strstr(name, "conv2d2.weight") || strstr(name, "conv2.weight")) model.conv2_w = t;
        else if (strstr(name, "conv2d2.bias") || strstr(name, "conv2.bias")) model.conv2_b = t;
        else if (strstr(name, "conv2d3.weight") || strstr(name, "conv3.weight")) model.conv3_w = t;
        else if (strstr(name, "conv2d3.bias") || strstr(name, "conv3.bias")) model.conv3_b = t;
        else if (strstr(name, "conv_out.weight")) model.conv_out_w = t;
        else if (strstr(name, "ln_post.weight")) model.ln_post_w = t;
        else if (strstr(name, "ln_post.bias")) model.ln_post_b = t;
        else if (strstr(name, "proj1.weight")) model.proj1_w = t;
        else if (strstr(name, "proj1.bias")) model.proj1_b = t;
        else if (strstr(name, "proj2.weight")) model.proj2_w = t;
        else if (strstr(name, "proj2.bias")) model.proj2_b = t;
        
        if (strstr(name, "layers.") || strstr(name, "encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "thinker.audio_tower.layers.%d.", &layer_idx) == 1 ||
                sscanf(name, "audio_tower.layers.%d.", &layer_idx) == 1 ||
                sscanf(name, "audio.encoder.blk.%d.", &layer_idx) == 1 ||
                sscanf(name, "encoder.blk.%d.", &layer_idx) == 1) {
                if (layer_idx >= 0 && layer_idx < model.hparams.n_encoder_layers) {
                    auto& layer = model.layers[layer_idx];
                    if (strstr(name, "q_proj.weight") || strstr(name, "attn_q.weight")) layer.attn_q_w = t;
                    else if (strstr(name, "q_proj.bias") || strstr(name, "attn_q.bias")) layer.attn_q_b = t;
                    else if (strstr(name, "k_proj.weight") || strstr(name, "attn_k.weight")) layer.attn_k_w = t;
                    else if (strstr(name, "k_proj.bias") || strstr(name, "attn_k.bias")) layer.attn_k_b = t;
                    else if (strstr(name, "v_proj.weight") || strstr(name, "attn_v.weight")) layer.attn_v_w = t;
                    else if (strstr(name, "v_proj.bias") || strstr(name, "attn_v.bias")) layer.attn_v_b = t;
                    else if (strstr(name, "out_proj.weight") || strstr(name, "attn_out.weight")) layer.attn_out_w = t;
                    else if (strstr(name, "out_proj.bias") || strstr(name, "attn_out.bias")) layer.attn_out_b = t;
                    else if (strstr(name, "self_attn_layer_norm.weight") || strstr(name, "attn_norm.weight")) layer.attn_norm_w = t;
                    else if (strstr(name, "self_attn_layer_norm.bias") || strstr(name, "attn_norm.bias")) layer.attn_norm_b = t;
                    else if (strstr(name, "fc1.weight") || strstr(name, "ffn_up.weight")) layer.ffn_up_w = t;
                    else if (strstr(name, "fc1.bias") || strstr(name, "ffn_up.bias")) layer.ffn_up_b = t;
                    else if (strstr(name, "fc2.weight") || strstr(name, "ffn_down.weight")) layer.ffn_down_w = t;
                    else if (strstr(name, "fc2.bias") || strstr(name, "ffn_down.bias")) layer.ffn_down_b = t;
                    else if (strstr(name, "final_layer_norm.weight") || strstr(name, "ffn_norm.weight")) layer.ffn_norm_w = t;
                    else if (strstr(name, "final_layer_norm.bias") || strstr(name, "ffn_norm.bias")) layer.ffn_norm_b = t;
                }
            }
        }
    }
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) { if (error) error->message = "Cannot mmap"; gguf_free(ctx_gguf); return false; }
    
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
    if (!model.buffer) { if (error) error->message = "alloc tensors failed"; munmap(mmap_addr, st.st_size); gguf_free(ctx_gguf); return false; }
    
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        ggml_tensor* t = ggml_get_tensor(model.ctx, name);
        if (t) ggml_backend_tensor_set(t, data_base + gguf_get_tensor_offset(ctx_gguf, i), 0, ggml_nbytes(t));
    }
    
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev && strstr(ggml_backend_dev_name(gpu_dev), "Metal") == nullptr) {
        munmap(mmap_addr, st.st_size);
        model.mmap_addr = nullptr; model.mmap_size = 0;
    }
    
    gguf_free(ctx_gguf);
    fprintf(stderr, "ASR encoder model loaded: %zu bytes\n", ggml_backend_buffer_get_size(model.buffer));
    return true;
}

void free_asr_encoder_model(EncoderModel& model) {
    if (model.buffer) { ggml_backend_buffer_free(model.buffer); model.buffer = nullptr; }
    if (model.ctx) { ggml_free(model.ctx); model.ctx = nullptr; }
    if (model.mmap_addr) { munmap(model.mmap_addr, model.mmap_size); model.mmap_addr = nullptr; }
    model.layers.clear(); model.tensors.clear();
}

EncoderState* init(const Config& config) {
    EncoderState* state = new EncoderState();
    state->model = new EncoderModel();
    
    ErrorInfo err;
    if (!load_model(config.model_path.c_str(), *state->model, &err)) {
        fprintf(stderr, "Failed to load model: %s\n", err.message.c_str());
        delete state->model; delete state; return nullptr;
    }
    
    state->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state->backend_cpu) { free_asr_encoder_model(*state->model); delete state->model; delete state; return nullptr; }
    
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
    
    state->sched = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), MAX_NODES, false, true);
    if (!state->sched) { free_asr_encoder_model(*state->model); delete state->model; delete state; return nullptr; }
    
    state->compute_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    int pe_out_w = compute_conv_output_len(100);
    int n_state = state->model->hparams.d_model;
    state->pe_cached.resize(pe_out_w * n_state);
    compute_sinusoidal_pe(state->pe_cached.data(), pe_out_w, n_state);
    state->pe_cached_out_w = pe_out_w;
    
    return state;
}

void free(EncoderState* state) {
    if (!state) return;
    if (state->sched) ggml_backend_sched_free(state->sched);
    if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
    if (state->backend_cpu) ggml_backend_free(state->backend_cpu);
    if (state->model) { free_asr_encoder_model(*state->model); delete state->model; }
    delete state;
}

const char* get_device_name(EncoderState* state) {
    return state && state->backend_gpu ? ggml_backend_name(state->backend_gpu) : "CPU";
}

HyperParams get_hparams(EncoderState* state) {
    return state && state->model ? state->model->hparams : HyperParams();
}

static ggml_tensor* conv_2d_via_im2col(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input, int s0, int s1, int p0, int p1, int d0, int d1) {
    ggml_tensor* im2col_out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F16);
    ggml_tensor* result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col_out, im2col_out->ne[0], im2col_out->ne[3] * im2col_out->ne[2] * im2col_out->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]));
    result = ggml_reshape_4d(ctx, result, im2col_out->ne[1], im2col_out->ne[2], im2col_out->ne[3], kernel->ne[3]);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
    return result;
}

static ggml_cgraph* build_graph_conv_batch(EncoderState* state, int n_frames, int batch_size) {
    EncoderModel& m = *state->model;
    int n_mel = m.hparams.n_mel_bins;
    int conv_ch = m.hparams.conv_channels;
    int d_model = m.hparams.d_model;
    
    ggml_init_params params = { state->compute_meta.size(), state->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    
    ggml_tensor* mel_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_frames, n_mel, 1, batch_size);
    ggml_set_name(mel_batch, "mel_batch"); ggml_set_input(mel_batch);
    
    ggml_tensor* cur = conv_2d_via_im2col(ctx, m.conv1_w, mel_batch, 2, 2, 1, 1, 1, 1);
    if (m.conv1_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv1_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = conv_2d_via_im2col(ctx, m.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv2_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = conv_2d_via_im2col(ctx, m.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv3_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv3_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    int64_t out_w = cur->ne[0];
    int64_t out_h = cur->ne[1];
    int64_t out_c = cur->ne[2];
    int64_t feat_dim = out_c * out_h;
    int64_t total_seq = out_w * batch_size;
    
    cur = ggml_reshape_4d(ctx, cur, out_w, out_h * out_c, batch_size, 1);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 0, 2, 3));
    cur = ggml_reshape_2d(ctx, cur, feat_dim, total_seq);
    
    if (m.conv_out_w) cur = ggml_mul_mat(ctx, m.conv_out_w, cur);
    
    // Add sinusoidal PE on GPU: PE tensor for max_out_w positions
    // Instead of ggml_repeat which may cause issues, we use a pre-computed
    // PE input tensor that matches the full output shape
    ggml_tensor* pe_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, out_w * batch_size);
    ggml_set_name(pe_tensor, "inp_pe"); ggml_set_input(pe_tensor);
    cur = ggml_add(ctx, cur, pe_tensor);
    
    ggml_set_name(cur, "embd_conv_batch"); ggml_set_output(cur);
    state->embd_conv = cur;
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

static ggml_cgraph* build_graph_encoder_batch(EncoderState* state, int n_ctx) {
    EncoderModel& m = *state->model;
    int n_state = m.hparams.d_model;
    int n_head = m.hparams.n_attention_heads;
    int n_layer = m.hparams.n_encoder_layers;
    int head_dim = n_state / n_head;
    float eps = 1e-5f, scale = 1.0f / sqrtf(head_dim);
    
    ggml_init_params params = { state->compute_meta.size(), state->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
    
    ggml_tensor* inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(inpL, "enc_input_batch"); ggml_set_input(inpL);
    
    ggml_tensor* cur = inpL;
    for (int L = 0; L < n_layer; ++L) {
        auto& l = m.layers[L];
        cur = ggml_norm(ctx, inpL, eps);
        if (l.attn_norm_w) cur = ggml_mul(ctx, cur, l.attn_norm_w);
        if (l.attn_norm_b) cur = ggml_add(ctx, cur, l.attn_norm_b);
        
        ggml_tensor* Qcur = ggml_mul_mat(ctx, l.attn_q_w, cur);
        if (l.attn_q_b) Qcur = ggml_add(ctx, Qcur, l.attn_q_b);
        ggml_tensor* Q = ggml_permute(ctx, ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_ctx), 0, 2, 1, 3);
        
        ggml_tensor* Kcur = ggml_mul_mat(ctx, l.attn_k_w, cur);
        if (l.attn_k_b) Kcur = ggml_add(ctx, Kcur, l.attn_k_b);
        ggml_tensor* K = ggml_permute(ctx, ggml_reshape_3d(ctx, Kcur, head_dim, n_head, n_ctx), 0, 2, 1, 3);
        
        ggml_tensor* Vcur = ggml_mul_mat(ctx, l.attn_v_w, cur);
        if (l.attn_v_b) Vcur = ggml_add(ctx, Vcur, l.attn_v_b);
        ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Vcur, head_dim, n_head, n_ctx), 1, 2, 0, 3));
        
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        ggml_tensor* attn = ggml_mul_mat(ctx, V, ggml_soft_max_ext(ctx, KQ, nullptr, scale, 0.0f));
        cur = ggml_cont_2d(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3), n_state, n_ctx);
        
        cur = ggml_mul_mat(ctx, l.attn_out_w, cur);
        if (l.attn_out_b) cur = ggml_add(ctx, cur, l.attn_out_b);
        cur = ggml_add(ctx, cur, inpL);
        
        ggml_tensor* ffn_in = cur;
        cur = ggml_norm(ctx, ffn_in, eps);
        if (l.ffn_norm_w) cur = ggml_mul(ctx, cur, l.ffn_norm_w);
        if (l.ffn_norm_b) cur = ggml_add(ctx, cur, l.ffn_norm_b);
        cur = ggml_mul_mat(ctx, l.ffn_up_w, cur);
        if (l.ffn_up_b) cur = ggml_add(ctx, cur, l.ffn_up_b);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, l.ffn_down_w, cur);
        if (l.ffn_down_b) cur = ggml_add(ctx, cur, l.ffn_down_b);
        inpL = ggml_add(ctx, cur, ffn_in);
    }
    
    cur = inpL;
    if (m.ln_post_w) { 
        cur = ggml_norm(ctx, cur, eps); 
        cur = ggml_mul(ctx, cur, m.ln_post_w); 
        if (m.ln_post_b) cur = ggml_add(ctx, cur, m.ln_post_b); 
    }
    
    if (m.proj1_w) {
        cur = ggml_mul_mat(ctx, m.proj1_w, cur);
        if (m.proj1_b) cur = ggml_add(ctx, cur, m.proj1_b);
        cur = ggml_gelu(ctx, cur);
    }
    
    if (m.proj2_w) {
        cur = ggml_mul_mat(ctx, m.proj2_w, cur);
        if (m.proj2_b) cur = ggml_add(ctx, cur, m.proj2_b);
    }
    
    ggml_set_name(cur, "embd_enc_batch"); ggml_set_output(cur);
    state->embd_enc = cur;
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

static int compute_chunk_output_len(int input_len) {
    int len = input_len;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    return len;
}

bool encode_batch(EncoderState* state, const BatchInput& input, BatchOutput& output, ErrorInfo* error) {
    if (!state || !state->model) { if (error) error->message = "State not initialized"; return false; }
    if (input.batch_size() == 0) { if (error) error->message = "Empty batch"; return false; }
    
    EncoderModel& m = *state->model;
    int n_state = m.hparams.d_model;
    int n_mel = m.hparams.n_mel_bins;
    int batch_size = input.batch_size();
    
    output.features.resize(batch_size);
    
    const int chunk_size = 100;
    int max_out_w = compute_chunk_output_len(chunk_size);
    
    std::vector<int> clip_n_chunks(batch_size);
    int total_chunks = 0;
    std::vector<std::vector<int>> clip_chunk_lengths(batch_size);
    std::vector<std::vector<int>> clip_chunk_out_lens(batch_size);
    std::vector<int> clip_total_out_frames(batch_size, 0);
    
    for (int b = 0; b < batch_size; ++b) {
        int n_frames = input.n_frames[b];
        int n_chunks = (n_frames + chunk_size - 1) / chunk_size;
        clip_n_chunks[b] = n_chunks;
        total_chunks += n_chunks;
        
        clip_chunk_lengths[b].resize(n_chunks);
        clip_chunk_out_lens[b].resize(n_chunks);
        for (int c = 0; c < n_chunks; ++c) {
            int start = c * chunk_size;
            clip_chunk_lengths[b][c] = std::min(chunk_size, n_frames - start);
            clip_chunk_out_lens[b][c] = compute_chunk_output_len(clip_chunk_lengths[b][c]);
            clip_total_out_frames[b] += clip_chunk_out_lens[b][c];
        }
    }
    
    // Batched conv: all chunks in single graph call
    ggml_cgraph* gf_conv = build_graph_conv_batch(state, chunk_size, total_chunks);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf_conv)) {
        if (error) error->message = "alloc conv batch failed";
        return false;
    }
    
    size_t mel_batch_sz = (size_t)chunk_size * n_mel * total_chunks;
    std::vector<float> mel_batch_data(mel_batch_sz, 0.0f);
    
    int chunk_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        const float* mel_data = input.mel_data[b];
        int n_frames = input.n_frames[b];
        int n_chunks = clip_n_chunks[b];
        
        for (int c = 0; c < n_chunks; ++c) {
            int clen = clip_chunk_lengths[b][c];
            int start_frame = c * chunk_size;
            for (int mi = 0; mi < n_mel; ++mi) {
                for (int f = 0; f < clen; ++f) {
                    size_t idx = (size_t)f + (size_t)mi * chunk_size + (size_t)chunk_idx * chunk_size * n_mel;
                    mel_batch_data[idx] = mel_data[mi * n_frames + start_frame + f];
                }
            }
            chunk_idx++;
        }
    }
    
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf_conv, "mel_batch");
    ggml_backend_tensor_set(mel_t, mel_batch_data.data(), 0, mel_batch_data.size() * sizeof(float));
    
    // Upload PE tensor to GPU (PE for each chunk, repeated for total_chunks)
    std::vector<float> pe_full(max_out_w * n_state * total_chunks);
    for (int c = 0; c < total_chunks; ++c) {
        memcpy(pe_full.data() + c * max_out_w * n_state, state->pe_cached.data(), max_out_w * n_state * sizeof(float));
    }
    ggml_tensor* pe_t = ggml_graph_get_tensor(gf_conv, "inp_pe");
    ggml_backend_tensor_set(pe_t, pe_full.data(), 0, pe_full.size() * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state->sched, gf_conv) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "compute conv batch failed";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    ggml_tensor* embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv_batch");
    int64_t feat_dim = embd_conv->ne[0];
    std::vector<float> conv_all(feat_dim * max_out_w * total_chunks);
    ggml_backend_tensor_get(embd_conv, conv_all.data(), 0, conv_all.size() * sizeof(float));
    
    ggml_backend_sched_reset(state->sched);
    
    for (int b = 0; b < batch_size; ++b) {
        int total_out_frames = clip_total_out_frames[b];
        int n_chunks = clip_n_chunks[b];
        
        // Rearrange conv outputs (PE already added on GPU)
        std::vector<float> all_conv_outputs(total_out_frames * n_state);
        
        chunk_idx = 0;
        for (int bb = 0; bb < b; ++bb) chunk_idx += clip_n_chunks[bb];
        
        int32_t dst_offset = 0;
        for (int c = 0; c < n_chunks; ++c) {
            int32_t valid = clip_chunk_out_lens[b][c];
            for (int32_t t = 0; t < valid; ++t) {
                memcpy(all_conv_outputs.data() + (dst_offset + t) * n_state,
                       conv_all.data() + t * n_state + chunk_idx * n_state * max_out_w,
                       n_state * sizeof(float));
            }
            dst_offset += valid;
            chunk_idx++;
        }
        
        // Run transformer encoder
        ggml_cgraph* gf_enc = build_graph_encoder_batch(state, total_out_frames);
        if (!ggml_backend_sched_alloc_graph(state->sched, gf_enc)) {
            if (error) error->message = "alloc enc batch failed";
            return false;
        }
        
        ggml_tensor* enc_in = ggml_graph_get_tensor(gf_enc, "enc_input_batch");
        ggml_backend_tensor_set(enc_in, all_conv_outputs.data(), 0, all_conv_outputs.size() * sizeof(float));
        
        if (ggml_backend_sched_graph_compute(state->sched, gf_enc) != GGML_STATUS_SUCCESS) {
            if (error) error->message = "compute enc batch failed";
            ggml_backend_sched_reset(state->sched);
            return false;
        }
        
        ggml_tensor* embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc_batch");
        
        int out_hidden = m.hparams.hidden_size;
        std::vector<float> enc_out(out_hidden * total_out_frames);
        ggml_backend_tensor_get(embd_enc, enc_out.data(), 0, enc_out.size() * sizeof(float));
        
        output.features[b].hidden_size = out_hidden;
        output.features[b].n_frames = total_out_frames;
        output.features[b].data = std::move(enc_out);
        
        ggml_backend_sched_reset(state->sched);
    }
    
    return true;
}

bool load_ref_data(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end); size_t sz = file.tellg(); file.seekg(0);
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

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tol, bool verbose) {
    if (a.size() != b.size()) { if (verbose) fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size()); return false; }
    float max_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) max_diff = std::max(max_diff, fabsf(a[i] - b[i]));
    if (max_diff > tol) { if (verbose) fprintf(stderr, "Max diff: %.6f > %.6f\n", max_diff, tol); return false; }
    return true;
}

} // namespace encoder
} // namespace transcribe
} // namespace asr
