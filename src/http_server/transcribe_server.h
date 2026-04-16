#pragma once

#include "qwen3asr_c_api.h"
#include <string>
#include <mutex>
#include <vector>
#include <cstdint>

namespace qwen3_asr {

struct ASRServerConfig {
    std::string asr_model_path;
    int port = 8081;
    int n_threads = 4;
    int max_tokens = 1024;
    std::string default_language = "";
    std::string device = "";
};

class ASRServer {
private:
    ASRServerConfig config_;
    qwen3asr_handle handle_;
    std::mutex asr_mutex_;
    bool model_loaded_;
    
public:
    ASRServer(const ASRServerConfig& config);
    ~ASRServer();
    
    bool init();
    void run();
    
    std::string handle_transcribe(
        const std::vector<int16_t>& pcm_data,
        const std::string& language,
        const std::string& context,
        int max_tokens
    );
    
    bool is_model_loaded() const { return model_loaded_; }
    const ASRServerConfig& get_config() const { return config_; }
};

}