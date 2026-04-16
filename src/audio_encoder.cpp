#include "audio_encoder.h"
#include "timing.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#define QWEN3_ASR_MAX_NODES 4096

namespace qwen3_asr {

static void compute_sinusoidal_pe(float * pe, int n_ctx, int d_model) {
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

AudioEncoder::AudioEncoder() = default;

AudioEncoder::~AudioEncoder() {
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
    free_model(model_);
}

bool AudioEncoder::load_model(const std::string & model_path, const std::string & device_name) {
    GGUFLoader loader;
    if (!loader.load(model_path, model_)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
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

struct ggml_cgraph * AudioEncoder::build_graph_conv(int n_frames) {
    const auto & hp = model_.hparams;
    const int n_mel = hp.n_mel_bins;
    const int conv_ch = hp.conv_channels;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    
    struct ggml_tensor * mel_4d = ggml_reshape_4d(ctx0, mel, n_frames, n_mel, 1, 1);
    
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model_.conv2d1_w, mel_4d, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d1_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d1_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    ggml_set_name(cur, "after_conv1_pre_gelu");
    ggml_set_output(cur);
    cur = ggml_gelu(ctx0, cur);
    ggml_set_name(cur, "after_conv1");
    ggml_set_output(cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d2_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d2_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d3_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d3_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    ggml_set_name(cur, "after_conv3");
    ggml_set_output(cur);
    
    int64_t out_w = cur->ne[0];
    int64_t out_h = cur->ne[1];
    int64_t out_c = cur->ne[2];
    int64_t seq_len = out_w;
    int64_t feat_dim = out_c * out_h;
    
    cur = ggml_reshape_3d(ctx0, cur, out_w, out_h * out_c, 1);
    cur = ggml_transpose(ctx0, cur);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, feat_dim, seq_len);
    
    ggml_set_name(cur, "before_conv_out");
    ggml_set_output(cur);
    
    if (model_.conv_out_w) {
        cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);
    }
    
    ggml_set_name(cur, "embd_conv");
    ggml_set_output(cur);
    state_.embd_conv = cur;
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * AudioEncoder::build_graph_encoder(int n_ctx) {
    const auto & hp = model_.hparams;
    const int n_state = hp.d_model;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    struct ggml_tensor * cur = ggml_view_tensor(ctx0, state_.embd_conv);
    
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = cur;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(ctx0, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(ctx0, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(ctx0, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Vcur, n_state_head, n_head, n_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
        }
        
        {
            cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(ctx0, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(ctx0, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(ctx0, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(ctx0, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(ctx0, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(ctx0, cur);
            
            cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    state_.embd_enc = cur;
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioEncoder::compute_graph(struct ggml_cgraph * graph) {
    if (!ggml_backend_sched_alloc_graph(state_.sched, graph)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, graph) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    ggml_backend_sched_reset(state_.sched);
    return true;
}

static int compute_chunk_output_length(int chunk_len) {
    int len = chunk_len;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    len = (len - 1) / 2 + 1;
    return len;
}

bool AudioEncoder::encode(const float * mel_data, int n_mel, int n_frames, 
                          std::vector<float> & output) {
    QWEN3_TIMER("audio_encoding.total");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch: expected " + std::to_string(model_.hparams.n_mel_bins) + 
                     ", got " + std::to_string(n_mel);
        return false;
    }
    
    const int n_window = 50;
    const int chunk_size = n_window * 2;
    const int n_state = model_.hparams.d_model;
    
    int n_chunks = (n_frames + chunk_size - 1) / chunk_size;
    
    std::vector<int> chunk_lengths(n_chunks);
    std::vector<int> chunk_output_lengths(n_chunks);
    int total_output_frames = 0;
    
    for (int i = 0; i < n_chunks; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_frames);
        chunk_lengths[i] = end - start;
        chunk_output_lengths[i] = compute_chunk_output_length(chunk_lengths[i]);
        total_output_frames += chunk_output_lengths[i];
    }
    
    std::vector<float> all_conv_outputs;
    all_conv_outputs.reserve(total_output_frames * n_state);
    
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        QWEN3_TIMER("audio_encoding.conv_chunk");
        int chunk_start = chunk_idx * chunk_size;
        int chunk_len = chunk_lengths[chunk_idx];
        int chunk_out_len = chunk_output_lengths[chunk_idx];
        
        struct ggml_cgraph * gf_conv = build_graph_conv(chunk_len);
        
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
            error_msg_ = "Failed to allocate conv graph for chunk " + std::to_string(chunk_idx);
            return false;
        }
        
        struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
        if (!mel_tensor) {
            error_msg_ = "Failed to find mel tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        std::vector<float> chunk_mel(n_mel * chunk_len);
        for (int m = 0; m < n_mel; ++m) {
            for (int f = 0; f < chunk_len; ++f) {
                chunk_mel[f + m * chunk_len] = mel_data[m * n_frames + chunk_start + f];
            }
        }
        
        ggml_backend_tensor_set(mel_tensor, chunk_mel.data(), 0, n_mel * chunk_len * sizeof(float));
        
        if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute conv graph for chunk " + std::to_string(chunk_idx);
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
        if (!embd_conv) {
            error_msg_ = "Failed to find embd_conv tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        int64_t out_ctx = embd_conv->ne[1];
        int64_t out_state = embd_conv->ne[0];
        
        if (out_ctx != chunk_out_len) {
            fprintf(stderr, "WARNING: Expected %d output frames, got %lld\n", chunk_out_len, (long long)out_ctx);
        }
        
        std::vector<float> chunk_output(out_ctx * out_state);
        ggml_backend_tensor_get(embd_conv, chunk_output.data(), 0, out_ctx * out_state * sizeof(float));
        
        std::vector<float> chunk_pe(out_ctx * out_state);
        compute_sinusoidal_pe(chunk_pe.data(), out_ctx, out_state);
        for (int64_t i = 0; i < out_ctx * out_state; ++i) {
            chunk_output[i] += chunk_pe[i];
        }
        
        all_conv_outputs.insert(all_conv_outputs.end(), chunk_output.begin(), chunk_output.end());
        
        ggml_backend_sched_reset(state_.sched);
    }
    
    int64_t n_ctx = total_output_frames;
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    struct ggml_init_params enc_params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * enc_ctx = ggml_init(enc_params);
    struct ggml_cgraph * gf_enc = ggml_new_graph_custom(enc_ctx, QWEN3_ASR_MAX_NODES, false);
    
    const auto & hp = model_.hparams;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = ggml_new_tensor_2d(enc_ctx, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(inpL, "enc_input");
    ggml_set_input(inpL);
    
    struct ggml_tensor * cur = inpL;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(enc_ctx, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(enc_ctx, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(enc_ctx, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(enc_ctx, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(enc_ctx, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(enc_ctx, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(enc_ctx, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(enc_ctx, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Qcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Kcur, n_state_head, n_head, n_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(enc_ctx, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(enc_ctx, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(enc_ctx, ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Vcur, n_state_head, n_head, n_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(enc_ctx, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(enc_ctx, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(enc_ctx, KQV_merged, n_state, n_ctx);
        }
        
        {
            cur = ggml_mul_mat(enc_ctx, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(enc_ctx, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(enc_ctx, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(enc_ctx, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(enc_ctx, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(enc_ctx, cur);
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(enc_ctx, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(enc_ctx, cur, eps);
        cur = ggml_mul(enc_ctx, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(enc_ctx, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj1_w, cur);
        if (model_.proj1_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj1_b);
        }
        cur = ggml_gelu(enc_ctx, cur);
    }
    
    if (model_.proj2_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj2_w, cur);
        if (model_.proj2_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj2_b);
        }
    }
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf_enc, cur);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_enc)) {
        error_msg_ = "Failed to allocate encoder graph";
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * enc_input = ggml_graph_get_tensor(gf_enc, "enc_input");
    if (!enc_input) {
        error_msg_ = "Failed to find enc_input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    ggml_backend_tensor_set(enc_input, all_conv_outputs.data(), 0, n_ctx * n_state * sizeof(float));
    
    {
        QWEN3_TIMER("audio_encoding.transformer");
        if (ggml_backend_sched_graph_compute(state_.sched, gf_enc) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute encoder graph";
            ggml_backend_sched_reset(state_.sched);
            ggml_free(enc_ctx);
            return false;
        }
    }
    
    struct ggml_tensor * embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc");
    if (!embd_enc) {
        error_msg_ = "Failed to find embd_enc tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    int64_t out_n_ctx = embd_enc->ne[1];
    int64_t out_n_state = embd_enc->ne[0];
    
    output.resize(out_n_ctx * out_n_state);
    ggml_backend_tensor_get(embd_enc, output.data(), 0, out_n_ctx * out_n_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    ggml_free(enc_ctx);
    
    return true;
}

bool AudioEncoder::encode_no_chunk(const float * mel_data, int n_mel, int n_frames,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch";
        return false;
    }
    
    const int n_state = model_.hparams.d_model;
    
    struct ggml_cgraph * gf_conv = build_graph_conv(n_frames);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
        error_msg_ = "Failed to allocate conv graph";
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    std::vector<float> transposed_mel(n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            transposed_mel[f + m * n_frames] = mel_data[m * n_frames + f];
        }
    }
    
    ggml_backend_tensor_set(mel_tensor, transposed_mel.data(), 0, n_mel * n_frames * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute conv graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
    if (!embd_conv) {
        error_msg_ = "Failed to find embd_conv tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t out_ctx = embd_conv->ne[1];
    int64_t out_state = embd_conv->ne[0];
    
    std::vector<float> conv_output(out_ctx * out_state);
    ggml_backend_tensor_get(embd_conv, conv_output.data(), 0, out_ctx * out_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    std::vector<float> pos_emb(out_ctx * n_state);
    compute_sinusoidal_pe(pos_emb.data(), out_ctx, n_state);
    for (int64_t i = 0; i < out_ctx * n_state; ++i) {
        conv_output[i] += pos_emb[i];
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    struct ggml_init_params enc_params = {
        state_.compute_meta.size(),
        state_.compute_meta.data(),
        true,
    };
    
    struct ggml_context * enc_ctx = ggml_init(enc_params);
    struct ggml_cgraph * gf_enc = ggml_new_graph_custom(enc_ctx, QWEN3_ASR_MAX_NODES, false);
    
    const auto & hp = model_.hparams;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_tensor * inpL = ggml_new_tensor_2d(enc_ctx, GGML_TYPE_F32, n_state, out_ctx);
    ggml_set_name(inpL, "enc_input");
    ggml_set_input(inpL);
    
    struct ggml_tensor * cur = inpL;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(enc_ctx, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(enc_ctx, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(enc_ctx, layer.attn_q_w, cur);
            if (layer.attn_q_b) {
                Qcur = ggml_add(enc_ctx, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(enc_ctx, layer.attn_k_w, cur);
            if (layer.attn_k_b) {
                Kcur = ggml_add(enc_ctx, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(enc_ctx, layer.attn_v_w, cur);
            if (layer.attn_v_b) {
                Vcur = ggml_add(enc_ctx, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Qcur, n_state_head, n_head, out_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Kcur, n_state_head, n_head, out_ctx),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(enc_ctx, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(enc_ctx, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(enc_ctx, ggml_permute(enc_ctx,
                ggml_reshape_3d(enc_ctx, Vcur, n_state_head, n_head, out_ctx),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(enc_ctx, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(enc_ctx, KQV, 0, 2, 1, 3);
            
            cur = ggml_cont_2d(enc_ctx, KQV_merged, n_state, out_ctx);
        }
        
        {
            cur = ggml_mul_mat(enc_ctx, layer.attn_out_w, cur);
            if (layer.attn_out_b) {
                cur = ggml_add(enc_ctx, cur, layer.attn_out_b);
            }
        }
        
        cur = ggml_add(enc_ctx, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(enc_ctx, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(enc_ctx, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(enc_ctx, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_up_w, cur);
            if (layer.ffn_up_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(enc_ctx, cur);
            
            cur = ggml_mul_mat(enc_ctx, layer.ffn_down_w, cur);
            if (layer.ffn_down_b) {
                cur = ggml_add(enc_ctx, cur, layer.ffn_down_b);
            }
        }
        
        inpL = ggml_add(enc_ctx, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(enc_ctx, cur, eps);
        cur = ggml_mul(enc_ctx, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(enc_ctx, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj1_w, cur);
        if (model_.proj1_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj1_b);
        }
        cur = ggml_gelu(enc_ctx, cur);
    }
    
    if (model_.proj2_w) {
        cur = ggml_mul_mat(enc_ctx, model_.proj2_w, cur);
        if (model_.proj2_b) {
            cur = ggml_add(enc_ctx, cur, model_.proj2_b);
        }
    }
    
    ggml_set_name(cur, "embd_enc");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf_enc, cur);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_enc)) {
        error_msg_ = "Failed to allocate encoder graph";
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * enc_input = ggml_graph_get_tensor(gf_enc, "enc_input");
    if (!enc_input) {
        error_msg_ = "Failed to find enc_input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    ggml_backend_tensor_set(enc_input, conv_output.data(), 0, out_ctx * n_state * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_enc) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute encoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    struct ggml_tensor * embd_enc = ggml_graph_get_tensor(gf_enc, "embd_enc");
    if (!embd_enc) {
        error_msg_ = "Failed to find embd_enc tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(enc_ctx);
        return false;
    }
    
    int64_t final_ctx = embd_enc->ne[1];
    int64_t final_state = embd_enc->ne[0];
    
    output.resize(final_ctx * final_state);
    ggml_backend_tensor_get(embd_enc, output.data(), 0, final_ctx * final_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    ggml_free(enc_ctx);
    
    return true;
}

bool AudioEncoder::encode_conv_only(const float * mel_data, int n_mel, int n_frames,
                                     std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (n_mel != model_.hparams.n_mel_bins) {
        error_msg_ = "Mel bins mismatch";
        return false;
    }
    
    const int n_state = model_.hparams.d_model;
    
    struct ggml_cgraph * gf_conv = build_graph_conv(n_frames);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
        error_msg_ = "Failed to allocate conv graph";
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    std::vector<float> transposed_mel(n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            transposed_mel[f + m * n_frames] = mel_data[m * n_frames + f];
        }
    }
    
    ggml_backend_tensor_set(mel_tensor, transposed_mel.data(), 0, n_mel * n_frames * sizeof(float));
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute conv graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * embd_conv = ggml_graph_get_tensor(gf_conv, "embd_conv");
    if (!embd_conv) {
        error_msg_ = "Failed to find embd_conv tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t out_ctx = embd_conv->ne[1];
    int64_t out_state = embd_conv->ne[0];
    
    output.resize(out_ctx * out_state);
    ggml_backend_tensor_get(embd_conv, output.data(), 0, out_ctx * out_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

struct ggml_cgraph * AudioEncoder::build_graph_conv_batch(int max_frames, int batch_size) {
    const auto & hp = model_.hparams;
    const int n_mel = hp.n_mel_bins;
    const int conv_ch = hp.conv_channels;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    
    struct ggml_tensor * mel_batch = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, max_frames, n_mel, 1, batch_size);
    ggml_set_name(mel_batch, "mel_batch");
    ggml_set_input(mel_batch);
    
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model_.conv2d1_w, mel_batch, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d1_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d1_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d2_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d2_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d3_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d3_b, 1, 1, conv_ch, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    int64_t out_w = cur->ne[0];
    int64_t out_h = cur->ne[1];
    int64_t out_c = cur->ne[2];
    int64_t feat_dim = out_c * out_h;
    
    cur = ggml_reshape_3d(ctx0, cur, out_w, feat_dim, batch_size);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_2d(ctx0, cur, feat_dim, out_w * batch_size);
    
    if (model_.conv_out_w) {
        cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);
    }
    
    cur = ggml_reshape_3d(ctx0, cur, hp.d_model, out_w, batch_size);
    
    ggml_set_name(cur, "conv_batch_out");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * AudioEncoder::build_graph_encoder_batch(int n_ctx, int batch_size) {
    const auto & hp = model_.hparams;
    const int n_state = hp.d_model;
    const int n_head = hp.n_attention_heads;
    const int n_layer = hp.n_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.layer_norm_eps;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    struct ggml_tensor * inpL = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state, n_ctx, batch_size);
    ggml_set_name(inpL, "enc_batch_input");
    ggml_set_input(inpL);
    
    struct ggml_tensor * cur = inpL;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        {
            cur = ggml_norm(ctx0, inpL, eps);
            if (layer.attn_norm_w) {
                cur = ggml_mul(ctx0, cur, layer.attn_norm_w);
            }
            if (layer.attn_norm_b) {
                cur = ggml_add(ctx0, cur, layer.attn_norm_b);
            }
        }
        
        {
            struct ggml_tensor * cur_c = ggml_cont(ctx0, cur);
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur_c);
            if (layer.attn_q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
            }
            
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur_c);
            if (layer.attn_k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.attn_k_b);
            }
            
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur_c);
            if (layer.attn_v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
            }
            
            struct ggml_tensor * Q = ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, Qcur, n_state_head, n_head, n_ctx, batch_size),
                0, 2, 1, 3);
            
            struct ggml_tensor * K = ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, Kcur, n_state_head, n_head, n_ctx, batch_size),
                0, 2, 1, 3);
            
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
            
            struct ggml_tensor * V = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, Vcur, n_state_head, n_head, n_ctx, batch_size),
                1, 2, 0, 3));
            
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            
            cur = ggml_reshape_3d(ctx0, KQV_merged, n_state, n_ctx, batch_size);
        }
        
        {
            cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx * batch_size);
            struct ggml_tensor * cur_c = ggml_cont(ctx0, cur);
            cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur_c);
            if (layer.attn_out_b) {
                cur = ggml_add(ctx0, cur, layer.attn_out_b);
            }
            cur = ggml_reshape_3d(ctx0, cur, n_state, n_ctx, batch_size);
        }
        
        cur = ggml_add(ctx0, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        {
            {
                cur = ggml_norm(ctx0, inpFF, eps);
                if (layer.ffn_norm_w) {
                    cur = ggml_mul(ctx0, cur, layer.ffn_norm_w);
                }
                if (layer.ffn_norm_b) {
                    cur = ggml_add(ctx0, cur, layer.ffn_norm_b);
                }
            }
            
            cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx * batch_size);
            struct ggml_tensor * cur_c = ggml_cont(ctx0, cur);
            cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur_c);
            if (layer.ffn_up_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_up_b);
            }
            
            cur = ggml_gelu(ctx0, cur);
            
            cur_c = ggml_cont(ctx0, cur);
            cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur_c);
            if (layer.ffn_down_b) {
                cur = ggml_add(ctx0, cur, layer.ffn_down_b);
            }
            cur = ggml_reshape_3d(ctx0, cur, n_state, n_ctx, batch_size);
        }
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(ctx0, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx * batch_size);
        struct ggml_tensor * cur_c = ggml_cont(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model_.proj1_w, cur_c);
        if (model_.proj1_b) {
            cur = ggml_add(ctx0, cur, model_.proj1_b);
        }
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_reshape_3d(ctx0, cur, model_.proj1_w->ne[0], n_ctx, batch_size);
    }
    
    if (model_.proj2_w) {
        cur = ggml_reshape_2d(ctx0, cur, model_.proj1_w ? model_.proj1_w->ne[0] : n_state, n_ctx * batch_size);
        struct ggml_tensor * cur_c = ggml_cont(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model_.proj2_w, cur_c);
        if (model_.proj2_b) {
            cur = ggml_add(ctx0, cur, model_.proj2_b);
        }
        cur = ggml_reshape_3d(ctx0, cur, model_.proj2_w->ne[0], n_ctx, batch_size);
    }
    
    ggml_set_name(cur, "enc_batch_out");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioEncoder::encode_batch(const BatchMelInput& input, BatchEncoderOutput& output) {
    QWEN3_TIMER("audio_encoding.batch_total");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const int batch_size = input.batch_size;
    if (batch_size <= 0 || input.mels.size() != (size_t)batch_size || 
        input.mel_lengths.size() != (size_t)batch_size) {
        error_msg_ = "Invalid batch input";
        return false;
    }
    
    const auto & hp = model_.hparams;
    const int n_mel = hp.n_mel_bins;
    const int n_state = hp.d_model;
    
    for (int i = 0; i < batch_size; ++i) {
        if (!input.mels[i]) {
            error_msg_ = "Null mel pointer at index " + std::to_string(i);
            return false;
        }
    }
    
    int max_frames = 0;
    for (int i = 0; i < batch_size; ++i) {
        max_frames = std::max(max_frames, input.mel_lengths[i]);
    }
    
    if (max_frames <= 0) {
        error_msg_ = "All mel inputs are empty";
        return false;
    }
    
    std::vector<int> output_lengths(batch_size);
    int max_out_frames = compute_chunk_output_length(max_frames);
    
    for (int i = 0; i < batch_size; ++i) {
        output_lengths[i] = compute_chunk_output_length(input.mel_lengths[i]);
    }
    
    struct ggml_cgraph * gf_conv = build_graph_conv_batch(max_frames, batch_size);
    if (!gf_conv) {
        error_msg_ = "Failed to build conv batch graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_conv)) {
        error_msg_ = "Failed to allocate conv batch graph";
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf_conv, "mel_batch");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel_batch tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    {
        QWEN3_TIMER("audio_encoding.batch_conv_input");
        size_t batch_data_size = (size_t)max_frames * n_mel * batch_size;
        std::vector<float> mel_batch_data(batch_data_size, 0.0f);
        
        for (int b = 0; b < batch_size; ++b) {
            const float* mel_src = input.mels[b];
            int n_frames = input.mel_lengths[b];
            
            for (int m = 0; m < n_mel; ++m) {
                for (int f = 0; f < n_frames; ++f) {
                    size_t idx = (size_t)f + (size_t)m * max_frames + (size_t)b * max_frames * n_mel;
                    mel_batch_data[idx] = mel_src[m * n_frames + f];
                }
            }
        }
        
        ggml_backend_tensor_set(mel_tensor, mel_batch_data.data(), 0, batch_data_size * sizeof(float));
    }
    
    {
        QWEN3_TIMER("audio_encoding.batch_conv_compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf_conv) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute conv batch graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    struct ggml_tensor * conv_out = ggml_graph_get_tensor(gf_conv, "conv_batch_out");
    if (!conv_out) {
        error_msg_ = "Failed to find conv_batch_out tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t conv_out_frames = conv_out->ne[1];
    
    std::vector<float> conv_all_data((size_t)n_state * conv_out_frames * batch_size);
    ggml_backend_tensor_get(conv_out, conv_all_data.data(), 0, conv_all_data.size() * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    std::vector<float> pos_emb(max_out_frames * n_state);
    compute_sinusoidal_pe(pos_emb.data(), max_out_frames, n_state);
    
    std::vector<std::vector<float>> enc_inputs(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        int valid_frames = output_lengths[b];
        enc_inputs[b].resize(valid_frames * n_state);
        
        for (int t = 0; t < valid_frames; ++t) {
            for (int d = 0; d < n_state; ++d) {
                float conv_val = conv_all_data[d + t * n_state + b * n_state * conv_out_frames];
                float pe_val = pos_emb[t * n_state + d];
                enc_inputs[b][t * n_state + d] = conv_val + pe_val;
            }
        }
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    struct ggml_cgraph * gf_enc = build_graph_encoder_batch(max_out_frames, batch_size);
    if (!gf_enc) {
        error_msg_ = "Failed to build encoder batch graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf_enc)) {
        error_msg_ = "Failed to allocate encoder batch graph";
        return false;
    }
    
    struct ggml_tensor * enc_input = ggml_graph_get_tensor(gf_enc, "enc_batch_input");
    if (!enc_input) {
        error_msg_ = "Failed to find enc_batch_input tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    {
        QWEN3_TIMER("audio_encoding.batch_enc_input");
        size_t enc_input_size = (size_t)n_state * max_out_frames * batch_size;
        std::vector<float> enc_input_data(enc_input_size, 0.0f);
        
        for (int b = 0; b < batch_size; ++b) {
            int valid_frames = output_lengths[b];
            for (int t = 0; t < valid_frames; ++t) {
                for (int d = 0; d < n_state; ++d) {
                    size_t idx = d + t * n_state + b * n_state * max_out_frames;
                    enc_input_data[idx] = enc_inputs[b][t * n_state + d];
                }
            }
        }
        
        ggml_backend_tensor_set(enc_input, enc_input_data.data(), 0, enc_input_size * sizeof(float));
    }
    
    {
        QWEN3_TIMER("audio_encoding.batch_enc_compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf_enc) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute encoder batch graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    struct ggml_tensor * enc_out = ggml_graph_get_tensor(gf_enc, "enc_batch_out");
    if (!enc_out) {
        error_msg_ = "Failed to find enc_batch_out tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t final_dim = enc_out->ne[0];
    int64_t final_frames = enc_out->ne[1];
    
    std::vector<float> enc_all_data((size_t)final_dim * final_frames * batch_size);
    ggml_backend_tensor_get(enc_out, enc_all_data.data(), 0, enc_all_data.size() * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    output.features.resize(batch_size);
    output.feature_lengths.resize(batch_size);
    output.success.resize(batch_size);
    output.errors.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int valid_frames = output_lengths[b];
        output.feature_lengths[b] = valid_frames;
        output.features[b].resize(valid_frames * final_dim);
        
        for (int t = 0; t < valid_frames; ++t) {
            for (int d = 0; d < final_dim; ++d) {
                size_t src_idx = d + t * final_dim + b * final_dim * final_frames;
                output.features[b][t * final_dim + d] = enc_all_data[src_idx];
            }
        }
        
        output.success[b] = true;
        output.errors[b] = "";
    }
    
    return true;
}

}
