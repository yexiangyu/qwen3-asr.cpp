#include "gguf_loader.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace qwen3_asr {

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() = default;

bool GGUFLoader::load(const std::string & path, audio_encoder_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    if (!parse_hparams(ctx_gguf, model)) {
        gguf_free(ctx_gguf);
        if (model.ctx) ggml_free(model.ctx);
        model.ctx = nullptr;
        return false;
    }
    
    if (!assign_tensors(ctx_gguf, model)) {
        gguf_free(ctx_gguf);
        if (model.ctx) ggml_free(model.ctx);
        model.ctx = nullptr;
        return false;
    }
    
    if (!load_tensor_data(path, ctx_gguf, model)) {
        free_model(model);
        gguf_free(ctx_gguf);
        return false;
    }
    
    gguf_free(ctx_gguf);
    
    return true;
}

bool GGUFLoader::parse_hparams(struct gguf_context * ctx, audio_encoder_model & model) {
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
    
    auto & hp = model.hparams;
    hp.n_encoder_layers = get_u32("audio.encoder_layers", 18);
    hp.d_model = get_u32("audio.d_model", 896);
    hp.n_attention_heads = get_u32("audio.attention_heads", 14);
    hp.ffn_dim = get_u32("audio.ffn_dim", 3584);
    hp.conv_channels = get_u32("audio.conv_channels", 480);
    hp.conv_out_dim = get_u32("audio.conv_out_dim", 896);
    hp.n_mel_bins = get_u32("audio.num_mel_bins", 128);
    hp.n_window_infer = get_u32("audio.n_window_infer", 800);
    hp.layer_norm_eps = get_f32("audio.layer_norm_eps", 1e-5f);
    
    auto & thp = model.text_hparams;
    thp.hidden_size = get_u32("text.hidden_size", 1024);
    thp.n_decoder_layers = get_u32("text.decoder_layers", 28);
    thp.n_attention_heads = get_u32("text.attention_heads", 16);
    thp.n_key_value_heads = get_u32("text.num_key_value_heads", 8);
    thp.intermediate_size = get_u32("text.intermediate_size", 3072);
    thp.rms_norm_eps = get_f32("text.rms_norm_eps", 1e-6f);
    
    return true;
}

static ggml_backend_buffer_type_t get_preferred_buft() {
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (strstr(dev_name, "Metal") != nullptr) {
            return ggml_backend_dev_buffer_type(gpu_dev);
        }
#ifdef GGML_USE_CUDA
        else {
            return ggml_backend_cuda_buffer_type(0);
        }
#endif
    }
    return ggml_backend_cpu_buffer_type();
}

bool GGUFLoader::assign_tensors(struct gguf_context * ctx_gguf, audio_encoder_model & model) {
    (void)ctx_gguf;
    model.layers.resize(model.hparams.n_encoder_layers);
    
    for (struct ggml_tensor * tensor = ggml_get_first_tensor(model.ctx); 
         tensor; 
         tensor = ggml_get_next_tensor(model.ctx, tensor)) {
        
        const char * name = ggml_get_name(tensor);
        model.tensors[name] = tensor;
        
        bool is_audio_encoder = strstr(name, "audio.encoder.") != nullptr;
        
        if (strstr(name, "encoder.conv1.weight")) {
            model.conv2d1_w = tensor;
        } else if (strstr(name, "encoder.conv1.bias")) {
            model.conv2d1_b = tensor;
        } else if (strstr(name, "encoder.conv2.weight")) {
            model.conv2d2_w = tensor;
        } else if (strstr(name, "encoder.conv2.bias")) {
            model.conv2d2_b = tensor;
        } else if (strstr(name, "encoder.conv3.weight")) {
            model.conv2d3_w = tensor;
        } else if (strstr(name, "encoder.conv3.bias")) {
            model.conv2d3_b = tensor;
        } else if (strstr(name, "encoder.conv_out.weight")) {
            model.conv_out_w = tensor;
        } else if (strstr(name, "encoder.ln_post.weight")) {
            model.ln_post_w = tensor;
        } else if (strstr(name, "encoder.ln_post.bias")) {
            model.ln_post_b = tensor;
        } else if (strstr(name, "encoder.proj1.weight")) {
            model.proj1_w = tensor;
        } else if (strstr(name, "encoder.proj1.bias")) {
            model.proj1_b = tensor;
        } else if (strstr(name, "encoder.proj2.weight")) {
            model.proj2_w = tensor;
        } else if (strstr(name, "encoder.proj2.bias")) {
            model.proj2_b = tensor;
        }
        
        if (is_audio_encoder) {
            for (int32_t l = 0; l < model.hparams.n_encoder_layers; l++) {
                char pattern[64];
                snprintf(pattern, sizeof(pattern), ".blk.%d.", l);
                
                if (strstr(name, pattern)) {
                    auto & layer = model.layers[l];
                    
                    if (strstr(name, "attn_q.weight")) layer.attn_q_w = tensor;
                    else if (strstr(name, "attn_q.bias")) layer.attn_q_b = tensor;
                    else if (strstr(name, "attn_k.weight")) layer.attn_k_w = tensor;
                    else if (strstr(name, "attn_k.bias")) layer.attn_k_b = tensor;
                    else if (strstr(name, "attn_v.weight")) layer.attn_v_w = tensor;
                    else if (strstr(name, "attn_v.bias")) layer.attn_v_b = tensor;
                    else if (strstr(name, "attn_out.weight")) layer.attn_out_w = tensor;
                    else if (strstr(name, "attn_out.bias")) layer.attn_out_b = tensor;
                    else if (strstr(name, "attn_norm.weight")) layer.attn_norm_w = tensor;
                    else if (strstr(name, "attn_norm.bias")) layer.attn_norm_b = tensor;
                    else if (strstr(name, "ffn_up.weight")) layer.ffn_up_w = tensor;
                    else if (strstr(name, "ffn_up.bias")) layer.ffn_up_b = tensor;
                    else if (strstr(name, "ffn_down.weight")) layer.ffn_down_w = tensor;
                    else if (strstr(name, "ffn_down.bias")) layer.ffn_down_b = tensor;
                    else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm_w = tensor;
                    else if (strstr(name, "ffn_norm.bias")) layer.ffn_norm_b = tensor;
                    break;
                }
            }
        }
    }
    
    return true;
}

bool GGUFLoader::load_tensor_data(const std::string & path, struct gguf_context * ctx_gguf, 
                                   audio_encoder_model & model) {
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
    
    model.mmap_addr = mmap_addr;
    model.mmap_size = st.st_size;
    
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    uint8_t * data_base = (uint8_t *)mmap_addr + data_offset;
    
    // Following llama.cpp pattern: use ggml_backend_alloc_ctx_tensors_from_buft
    // to allocate ALL tensors in the context to a single buffer type
    ggml_backend_buffer_type_t buft = get_preferred_buft();
    
    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, buft);
    if (!model.buffer) {
        error_msg_ = "Failed to allocate context tensors with buffer type";
        munmap(mmap_addr, st.st_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
        return false;
    }
    
    fprintf(stderr, "info: allocated %zu bytes for model weights\n", 
            ggml_backend_buffer_get_size(model.buffer));
    
    // Load tensor data from mmap to allocated buffer
    // For CUDA: this triggers H2D transfer
    // For Metal/CPU: this is a simple memcpy
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx, name);
        if (!tensor) {
            continue;
        }
        
        const size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
        const size_t sz = ggml_nbytes(tensor);
        
        ggml_backend_tensor_set(tensor, data_base + offset, 0, sz);
    }
    
    // For non-unified memory (CUDA), unmap the file after loading
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (strstr(dev_name, "Metal") == nullptr) {
            munmap(mmap_addr, st.st_size);
            model.mmap_addr = nullptr;
            model.mmap_size = 0;
        }
    }
    
    return true;
}

void free_model(audio_encoder_model & model) {
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

} // namespace qwen3_asr
