#include "asr/aligner/encoder.hpp"
#include "asr/aligner/encoder_model.hpp"
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

namespace asr { namespace aligner { namespace encoder {

using asr::AudioFeatures;
using asr::ErrorInfo;

constexpr int MAX_NODES = 16384;

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

static int chunk_output_len(int chunk_frames) {
    int len = chunk_frames;
    for (int i = 0; i < 3; ++i) len = (len - 1) / 2 + 1;
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
    
    model.hparams.n_encoder_layers = get_u32("qwen3asr.audio.n_layers", 24);
    model.hparams.d_model = get_u32("qwen3asr.audio.d_model", 1024);
    model.hparams.n_attention_heads = get_u32("qwen3asr.audio.n_heads", 16);
    model.hparams.n_mel_bins = get_u32("qwen3asr.n_mels", 128);
    model.hparams.hidden_size = get_u32("qwen3asr.llm.d_model", 1024);
    model.hparams.head_dim = get_u32("qwen3asr.audio.head_dim", 64);
    model.hparams.ff_dim = get_u32("qwen3asr.audio.ff_dim", 4096);
    model.hparams.conv_channels = get_u32("qwen3asr.audio.conv_channels", 480);
    
    model.layers.resize(model.hparams.n_encoder_layers);
    
    for (ggml_tensor* t = ggml_get_first_tensor(model.ctx); t; t = ggml_get_next_tensor(model.ctx, t)) {
        const char* name = ggml_get_name(t);
        model.tensors[name] = t;
        
        if (strstr(name, "audio.conv.1.weight")) model.conv2d1_w = t;
        else if (strstr(name, "audio.conv.1.bias")) model.conv2d1_b = t;
        else if (strstr(name, "audio.conv.2.weight")) model.conv2d2_w = t;
        else if (strstr(name, "audio.conv.2.bias")) model.conv2d2_b = t;
        else if (strstr(name, "audio.conv.3.weight")) model.conv2d3_w = t;
        else if (strstr(name, "audio.conv.3.bias")) model.conv2d3_b = t;
        else if (strstr(name, "audio.conv_out.weight")) model.conv_out_w = t;
        else if (strstr(name, "audio.ln_post.weight")) model.ln_post_w = t;
        else if (strstr(name, "audio.ln_post.bias")) model.ln_post_b = t;
        else if (strstr(name, "audio.proj1.weight")) model.proj1_w = t;
        else if (strstr(name, "audio.proj1.bias")) model.proj1_b = t;
        else if (strstr(name, "audio.proj2.weight")) model.proj2_w = t;
        else if (strstr(name, "audio.proj2.bias")) model.proj2_b = t;
        
        if (strstr(name, "audio.blk.")) {
            for (int l = 0; l < model.hparams.n_encoder_layers; l++) {
                char pattern[64];
                snprintf(pattern, sizeof(pattern), ".blk.%d.", l);
                if (strstr(name, pattern)) {
                    auto& layer = model.layers[l];
                    if (strstr(name, "attn_q.weight")) layer.attn_q_w = t;
                    else if (strstr(name, "attn_q.bias")) layer.attn_q_b = t;
                    else if (strstr(name, "attn_k.weight")) layer.attn_k_w = t;
                    else if (strstr(name, "attn_k.bias")) layer.attn_k_b = t;
                    else if (strstr(name, "attn_v.weight")) layer.attn_v_w = t;
                    else if (strstr(name, "attn_v.bias")) layer.attn_v_b = t;
                    else if (strstr(name, "attn_out.weight")) layer.attn_out_w = t;
                    else if (strstr(name, "attn_out.bias")) layer.attn_out_b = t;
                    else if (strstr(name, "attn_norm.weight")) layer.attn_norm_w = t;
                    else if (strstr(name, "attn_norm.bias")) layer.attn_norm_b = t;
                    else if (strstr(name, "ffn_up.weight")) layer.ffn_up_w = t;
                    else if (strstr(name, "ffn_up.bias")) layer.ffn_up_b = t;
                    else if (strstr(name, "ffn_down.weight")) layer.ffn_down_w = t;
                    else if (strstr(name, "ffn_down.bias")) layer.ffn_down_b = t;
                    else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm_w = t;
                    else if (strstr(name, "ffn_norm.bias")) layer.ffn_norm_b = t;
                    break;
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
    fprintf(stderr, "Encoder model loaded: %zu bytes\n", ggml_backend_buffer_get_size(model.buffer));
    return true;
}

void free_align_encoder_model(EncoderModel& model) {
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
    if (!state->backend_cpu) { free_align_encoder_model(*state->model); delete state->model; delete state; return nullptr; }
    
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
    if (!state->sched) { free_align_encoder_model(*state->model); delete state->model; delete state; return nullptr; }
    
    state->compute_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    return state;
}

void free(EncoderState* state) {
    if (!state) return;
    if (state->sched) ggml_backend_sched_free(state->sched);
    if (state->backend_gpu) ggml_backend_free(state->backend_gpu);
    if (state->backend_cpu) ggml_backend_free(state->backend_cpu);
    if (state->model) { free_align_encoder_model(*state->model); delete state->model; }
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

static ggml_cgraph* build_conv_graph(EncoderState* state, int max_chunk_len, int n_mel, int n_chunks) {
    EncoderModel& m = *state->model;
    int conv_ch = m.hparams.conv_channels;
    
    ggml_init_params params = { state->compute_meta.size(), state->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
    
    ggml_tensor* mel_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, max_chunk_len, n_mel, 1, n_chunks);
    ggml_set_name(mel_batch, "mel_batch");
    ggml_set_input(mel_batch);
    
    ggml_tensor* cur = conv_2d_via_im2col(ctx, m.conv2d1_w, mel_batch, 2, 2, 1, 1, 1, 1);
    if (m.conv2d1_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d1_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = conv_2d_via_im2col(ctx, m.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv2d2_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d2_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = conv_2d_via_im2col(ctx, m.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv2d3_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d3_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    int64_t conv_out_w = cur->ne[0];
    int64_t conv_out_h = cur->ne[1];
    int64_t conv_out_c = cur->ne[2];
    int64_t feat_dim = conv_out_c * conv_out_h;
    
    cur = ggml_reshape_3d(ctx, cur, conv_out_w, feat_dim, n_chunks);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 0, 2, 3));
    cur = ggml_reshape_2d(ctx, cur, feat_dim, conv_out_w * n_chunks);
    if (m.conv_out_w) cur = ggml_mul_mat(ctx, m.conv_out_w, cur);
    
    int n_state = m.hparams.d_model;
    cur = ggml_reshape_3d(ctx, cur, n_state, conv_out_w, n_chunks);
    
    ggml_set_name(cur, "conv_out");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

static ggml_cgraph* build_transformer_graph(EncoderState* state, int n_ctx) {
    EncoderModel& m = *state->model;
    int n_state = m.hparams.d_model;
    int n_head = m.hparams.n_attention_heads;
    int n_layer = m.hparams.n_encoder_layers;
    int head_dim = n_state / n_head;
    float eps = 1e-5f;
    float scale = 1.0f / sqrtf(float(head_dim));
    
    ggml_init_params params = { state->compute_meta.size(), state->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAX_NODES, false);
    
    ggml_tensor* inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(inpL, "inp_hidden");
    ggml_set_input(inpL);
    
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ctx, n_ctx);
    ggml_set_name(mask, "attn_mask");
    ggml_set_input(mask);
    
    ggml_tensor* cur = inpL;
    for (int il = 0; il < n_layer; ++il) {
        auto& l = m.layers[il];
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
        
        ggml_tensor* attn = ggml_mul_mat(ctx, V, ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), mask, scale, 0.0f));
        cur = ggml_cont_2d(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3), n_state, n_ctx);
        
        cur = ggml_mul_mat(ctx, l.attn_out_w, cur);
        if (l.attn_out_b) cur = ggml_add(ctx, cur, l.attn_out_b);
        cur = ggml_add(ctx, cur, inpL);
        
        ggml_tensor* inpFF = cur;
        cur = ggml_norm(ctx, inpFF, eps);
        if (l.ffn_norm_w) cur = ggml_mul(ctx, cur, l.ffn_norm_w);
        if (l.ffn_norm_b) cur = ggml_add(ctx, cur, l.ffn_norm_b);
        cur = ggml_mul_mat(ctx, l.ffn_up_w, cur);
        if (l.ffn_up_b) cur = ggml_add(ctx, cur, l.ffn_up_b);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, l.ffn_down_w, cur);
        if (l.ffn_down_b) cur = ggml_add(ctx, cur, l.ffn_down_b);
        inpL = ggml_add(ctx, cur, inpFF);
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
    
    ggml_set_name(cur, "audio_enc_out");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

bool encode_batch(EncoderState* state, const BatchInput& input, BatchOutput& output, ErrorInfo* error) {
    if (!state || !state->model) { if (error) error->message = "State not initialized"; return false; }
    if (input.batch_size() == 0) { if (error) error->message = "Empty batch"; return false; }
    
    EncoderModel& m = *state->model;
    int n_state = m.hparams.d_model;
    int n_mel = m.hparams.n_mel_bins;
    int batch_size = input.batch_size();
    
    constexpr int32_t chunk_mel_size = 100;
    constexpr int32_t n_window_infer = 800;
    int32_t max_chunk_len = chunk_mel_size;
    int32_t max_out_w = chunk_output_len(max_chunk_len);
    int32_t window_aftercnn = max_out_w * (n_window_infer / chunk_mel_size);
    
    std::vector<std::vector<int32_t>> clip_chunk_lengths(batch_size);
    std::vector<std::vector<int32_t>> clip_chunk_out_lens(batch_size);
    std::vector<int> clip_total_out_frames(batch_size, 0);
    std::vector<int> clip_n_chunks(batch_size);
    int total_chunks = 0;
    
    for (int b = 0; b < batch_size; ++b) {
        int n_frames = input.n_frames[b];
        int n_chunks = (n_frames + chunk_mel_size - 1) / chunk_mel_size;
        clip_n_chunks[b] = n_chunks;
        total_chunks += n_chunks;
        
        clip_chunk_lengths[b].resize(n_chunks);
        clip_chunk_out_lens[b].resize(n_chunks);
        for (int32_t c = 0; c < n_chunks; ++c) {
            if (c < n_chunks - 1) clip_chunk_lengths[b][c] = chunk_mel_size;
            else {
                clip_chunk_lengths[b][c] = n_frames - c * chunk_mel_size;
                if (clip_chunk_lengths[b][c] == 0) clip_chunk_lengths[b][c] = chunk_mel_size;
            }
            clip_chunk_out_lens[b][c] = chunk_output_len(clip_chunk_lengths[b][c]);
            clip_total_out_frames[b] += clip_chunk_out_lens[b][c];
        }
    }
    
    ggml_cgraph* gf_conv = build_conv_graph(state, max_chunk_len, n_mel, total_chunks);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf_conv)) {
        if (error) error->message = "alloc conv graph failed";
        return false;
    }
    
    size_t mel_batch_size = (size_t)max_chunk_len * n_mel * 1 * total_chunks;
    std::vector<float> mel_batch_data(mel_batch_size, 0.0f);
    
    int chunk_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        const float* mel_data = input.mel_data[b];
        int n_frames = input.n_frames[b];
        int n_chunks = clip_n_chunks[b];
        
        for (int32_t c = 0; c < n_chunks; ++c) {
            int32_t clen = clip_chunk_lengths[b][c];
            int32_t start_frame = c * chunk_mel_size;
            for (int mi = 0; mi < n_mel; ++mi) {
                for (int32_t f = 0; f < clen; ++f) {
                    size_t idx = (size_t)f + (size_t)mi * max_chunk_len + (size_t)chunk_idx * max_chunk_len * n_mel;
                    mel_batch_data[idx] = mel_data[mi * n_frames + start_frame + f];
                }
            }
            chunk_idx++;
        }
    }
    
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf_conv, "mel_batch");
    ggml_backend_tensor_set(mel_t, mel_batch_data.data(), 0, mel_batch_data.size() * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state->sched, gf_conv) != GGML_STATUS_SUCCESS) {
        if (error) error->message = "compute conv failed";
        ggml_backend_sched_reset(state->sched);
        return false;
    }
    
    ggml_tensor* conv_out_t = ggml_graph_get_tensor(gf_conv, "conv_out");
    std::vector<float> conv_all(n_state * max_out_w * total_chunks);
    ggml_backend_tensor_get(conv_out_t, conv_all.data(), 0, conv_all.size() * sizeof(float));
    
    ggml_backend_sched_reset(state->sched);
    
    std::vector<float> pos_emb_data(max_out_w * n_state);
    compute_sinusoidal_pe(pos_emb_data.data(), max_out_w, n_state);
    
    std::vector<std::vector<float>> clip_hidden(batch_size);
    std::vector<int> clip_n_ctx(batch_size);
    
    chunk_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        int total_out = clip_total_out_frames[b];
        clip_hidden[b].resize(total_out * n_state);
        clip_n_ctx[b] = total_out;
        
        int32_t dst_offset = 0;
        for (int32_t c = 0; c < clip_n_chunks[b]; ++c) {
            int32_t valid = clip_chunk_out_lens[b][c];
            for (int32_t t = 0; t < valid; ++t) {
                for (int32_t d = 0; d < n_state; ++d) {
                    float val = conv_all[d + t * n_state + chunk_idx * n_state * max_out_w];
                    float pe = pos_emb_data[t * n_state + d];
                    clip_hidden[b][(dst_offset + t) * n_state + d] = val + pe;
                }
            }
            dst_offset += valid;
            chunk_idx++;
        }
    }
    
    output.features.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int n_ctx = clip_n_ctx[b];
        
        std::vector<int32_t> cu_seqlens;
        cu_seqlens.push_back(0);
        {
            int32_t remaining = n_ctx;
            while (remaining > 0) {
                if (remaining >= window_aftercnn) {
                    cu_seqlens.push_back(cu_seqlens.back() + window_aftercnn);
                    remaining -= window_aftercnn;
                } else {
                    cu_seqlens.push_back(cu_seqlens.back() + remaining);
                    remaining = 0;
                }
            }
        }
        
        std::vector<float> attn_mask(n_ctx * n_ctx, -INFINITY);
        for (size_t seg = 1; seg < cu_seqlens.size(); ++seg) {
            int32_t seg_start = cu_seqlens[seg - 1];
            int32_t seg_end = cu_seqlens[seg];
            for (int32_t r = seg_start; r < seg_end; ++r) {
                for (int32_t c = seg_start; c < seg_end; ++c) {
                    attn_mask[r * n_ctx + c] = 0.0f;
                }
            }
        }
        
        ggml_cgraph* gf_enc = build_transformer_graph(state, n_ctx);
        if (!ggml_backend_sched_alloc_graph(state->sched, gf_enc)) {
            if (error) error->message = "alloc transformer graph failed";
            return false;
        }
        
        ggml_tensor* hidden_t = ggml_graph_get_tensor(gf_enc, "inp_hidden");
        ggml_backend_tensor_set(hidden_t, clip_hidden[b].data(), 0, clip_hidden[b].size() * sizeof(float));
        
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf_enc, "attn_mask");
        ggml_backend_tensor_set(mask_t, attn_mask.data(), 0, attn_mask.size() * sizeof(float));
        
        if (ggml_backend_sched_graph_compute(state->sched, gf_enc) != GGML_STATUS_SUCCESS) {
            if (error) error->message = "compute transformer failed";
            ggml_backend_sched_reset(state->sched);
            return false;
        }
        
        ggml_tensor* enc_out = ggml_graph_get_tensor(gf_enc, "audio_enc_out");
        std::vector<float> enc_all(n_state * n_ctx);
        ggml_backend_tensor_get(enc_out, enc_all.data(), 0, enc_all.size() * sizeof(float));
        
        output.features[b].hidden_size = n_state;
        output.features[b].n_frames = n_ctx;
        output.features[b].data = enc_all;
        
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
} // namespace aligner
} // namespace asr
