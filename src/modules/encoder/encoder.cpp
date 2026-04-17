#include "encoder.h"
#include "encoder_model.h"

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace qwen3_asr {
namespace encoder {

constexpr int MAX_NODES = 8192;

EncoderState* init(const Config& config) {
    EncoderState* state = new EncoderState();
    
    state->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state->backend_cpu) {
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
    for (auto backend : backends) {
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    
    state->sched = ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), MAX_NODES, false, true);
    if (!state->sched) {
        delete state;
        return nullptr;
    }
    
    state->compute_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    return state;
}

void free(EncoderState* state) {
    if (!state) return;
    
    if (state->sched) {
        ggml_backend_sched_free(state->sched);
    }
    if (state->backend_gpu) {
        ggml_backend_free(state->backend_gpu);
    }
    if (state->backend_cpu) {
        ggml_backend_free(state->backend_cpu);
    }
    
    delete state;
}

const char* get_device_name(EncoderState* state) {
    if (state && state->backend_gpu) {
        return ggml_backend_name(state->backend_gpu);
    }
    return "CPU";
}

HyperParams get_hparams(EncoderState* state) {
    return HyperParams();
}

bool load_ref_data(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return true;
}

bool save_ref_data(const char* path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose) {
    if (a.size() != b.size()) {
        if (verbose) fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return false;
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    if (max_diff > tolerance) {
        if (verbose) fprintf(stderr, "Max diff exceeds tolerance: %.6f > %.6f\n", max_diff, tolerance);
        return false;
    }
    
    return true;
}

void free_encoder_model(EncoderModel& model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.layers.clear();
    model.tensors.clear();
}

} // namespace encoder
} // namespace qwen3_asr