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

static ggml_cgraph* build_graph_conv_batch(EncoderState* state, int n_frames, int batch_size) {
    EncoderModel& m = *state->model;
    int n_mel = m.hparams.n_mel_bins;
    int conv_ch = m.hparams.conv_channels;
    
    ggml_init_params params = { state->compute_meta.size(), state->compute_meta.data(), true };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    
    ggml_tensor* mel_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_frames, n_mel, 1, batch_size);
    ggml_set_name(mel_batch, "mel_batch"); ggml_set_input(mel_batch);
    
    ggml_tensor* cur = ggml_conv_2d(ctx, m.conv2d1_w, mel_batch, 2, 2, 1, 1, 1, 1);
    if (m.conv2d1_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d1_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = ggml_conv_2d(ctx, m.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv2d2_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d2_b, 1, 1, conv_ch, 1));
    cur = ggml_gelu(ctx, cur);
    
    cur = ggml_conv_2d(ctx, m.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (m.conv2d3_b) cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, m.conv2d3_b, 1, 1, conv_ch, 1));
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
    
    ggml_set_name(cur, "embd_conv_batch"); ggml_set_output(cur);
    state->embd_conv = cur;
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

static ggml_cgraph* build_graph_encoder_batch(EncoderState* state, int n_ctx, int /*batch_size*/, int /*seq_len_per_item*/) {
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
    
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ctx, n_ctx);
    ggml_set_name(mask, "attn_mask"); ggml_set_input(mask);
    
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
        
        ggml_tensor* attn = ggml_mul_mat(ctx, V, ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), mask, scale, 0.0f));
        cur = ggml_cont_2d(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3), n_state, n_ctx);
        
        cur = ggml_mul_mat(ctx, l.attn_out_w, cur);
        if (l.attn_out_b) cur = ggml_add(ctx, cur, l.attn_out_b);
        cur = ggml_add(ctx, cur, inpL);
        
        ggml_tensor* ffn_in = cur;
        cur = ggml_norm(ctx, ffn_in, eps);
        if (l.ffn_norm_w) cur = ggml_mul(ctx, cur, l.ffn_norm_w);
        if (l.ffn_norm_b) cur = ggml_add(ctx, cur, l.ffn_norm_b);
        cur = ggml_gelu(ctx, ggml_mul_mat(ctx, l.ffn_up_w, cur));
        if (l.ffn_up_b) cur = ggml_add(ctx, cur, l.ffn_up_b);
        cur = ggml_mul_mat(ctx, l.ffn_down_w, cur);
        if (l.ffn_down_b) cur = ggml_add(ctx, cur, l.ffn_down_b);
        inpL = ggml_add(ctx, cur, ffn_in);
    }
    
    cur = inpL;
    if (m.ln_post_w) { cur = ggml_norm(ctx, cur, eps); cur = ggml_mul(ctx, cur, m.ln_post_w); if (m.ln_post_b) cur = ggml_add(ctx, cur, m.ln_post_b); }
    if (m.proj1_w) { cur = ggml_gelu(ctx, ggml_mul_mat(ctx, m.proj1_w, cur)); if (m.proj1_b) cur = ggml_add(ctx, cur, m.proj1_b); }
    if (m.proj2_w) { cur = ggml_mul_mat(ctx, m.proj2_w, cur); if (m.proj2_b) cur = ggml_add(ctx, cur, m.proj2_b); }
    
    ggml_set_name(cur, "embd_enc_batch"); ggml_set_output(cur);
    state->embd_enc = cur;
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
    int max_frames = input.max_frames;
    int seq_len_per_item = compute_conv_output_len(max_frames);
    
    std::vector<float> mel_padded(batch_size * n_mel * max_frames, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        int n_frames = input.n_frames[b];
        for (int m = 0; m < n_mel; ++m) {
            for (int f = 0; f < n_frames; ++f) {
                mel_padded[b * n_mel * max_frames + m * max_frames + f] = input.mel_data[b][m * n_frames + f];
            }
        }
    }
    
    ggml_cgraph* gf_conv = build_graph_conv_batch(state, max_frames, batch_size);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf_conv)) { if (error) error->message = "alloc conv batch failed"; return false; }
    
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf_conv, "mel_batch");
    ggml_backend_tensor_set(mel_t, mel_padded.data(), 0, mel_padded.size() * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state->sched, gf_conv) != GGML_STATUS_SUCCESS) { if (error) error->message = "compute conv batch failed"; ggml_backend_sched_reset(state->sched); return false; }
    
    ggml_tensor* embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv_batch");
    int64_t feat_dim = embd_conv->ne[0];
    int64_t total_seq = embd_conv->ne[1];
    
    std::vector<float> conv_out(feat_dim * total_seq);
    ggml_backend_tensor_get(embd_conv, conv_out.data(), 0, conv_out.size() * sizeof(float));
    
    std::vector<int> out_frames(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        out_frames[b] = compute_conv_output_len(input.n_frames[b]);
    }
    
    std::vector<float> pe(seq_len_per_item * feat_dim);
    compute_sinusoidal_pe(pe.data(), seq_len_per_item, feat_dim);
    
    for (int b = 0; b < batch_size; ++b) {
        int frame_offset = b * seq_len_per_item;
        int valid_len = out_frames[b];
        for (int f = 0; f < valid_len; ++f) {
            for (int h = 0; h < feat_dim; ++h) {
                conv_out[(frame_offset + f) * feat_dim + h] += pe[f * feat_dim + h];
            }
        }
    }
    
std::vector<float> attn_mask(total_seq * total_seq, -INFINITY);
    for (int b = 0; b < batch_size; ++b) {
        int frame_start = b * seq_len_per_item;
        int valid_len = out_frames[b];
        
        for (int r = frame_start; r < frame_start + valid_len; ++r) {
            for (int c = frame_start; c < frame_start + valid_len; ++c) {
                attn_mask[r * total_seq + c] = 0.0f;
            }
        }
        
        for (int r = frame_start + valid_len; r < frame_start + seq_len_per_item; ++r) {
            for (int c = frame_start; c < frame_start + valid_len; ++c) {
                attn_mask[r * total_seq + c] = 0.0f;
            }
        }
    }
    
    ggml_backend_sched_reset(state->sched);
    
    ggml_cgraph* gf_enc = build_graph_encoder_batch(state, total_seq, batch_size, seq_len_per_item);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf_enc)) { if (error) error->message = "alloc enc batch failed"; return false; }
    
    ggml_tensor* enc_in = ggml_graph_get_tensor(gf_enc, "enc_input_batch");
    ggml_backend_tensor_set(enc_in, conv_out.data(), 0, conv_out.size() * sizeof(float));
    
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf_enc, "attn_mask");
    ggml_backend_tensor_set(mask_t, attn_mask.data(), 0, attn_mask.size() * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state->sched, gf_enc) != GGML_STATUS_SUCCESS) { if (error) error->message = "compute enc batch failed"; ggml_backend_sched_reset(state->sched); return false; }
    
    ggml_tensor* embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc_batch");
    std::vector<float> all_out(n_state * total_seq);
    ggml_backend_tensor_get(embd_enc, all_out.data(), 0, all_out.size() * sizeof(float));
    
    output.features.resize(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        int actual_out = out_frames[b];
        int offset = b * seq_len_per_item;
        output.features[b].hidden_size = n_state;
        output.features[b].n_frames = actual_out;
        output.features[b].data.resize(actual_out * n_state);
        for (int f = 0; f < actual_out; ++f) {
            for (int h = 0; h < n_state; ++h) {
                output.features[b].data[f * n_state + h] = all_out[(offset + f) * n_state + h];
            }
        }
    }
    
    ggml_backend_sched_reset(state->sched);
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
