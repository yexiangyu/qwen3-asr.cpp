#pragma once

#include "qwen3asr_c_api.h"
#include <string>
#include <mutex>
#include <vector>
#include <cstdint>

namespace qwen3_asr {

struct ServerConfig {
    std::string aligner_model_path;
    std::string korean_dict_path;
    int port = 8080;
    int n_threads = 4;
    std::string default_language = "";
    std::string device = "";
};

class AlignServer {
private:
    ServerConfig config_;
    qwen3aligner_handle handle_;
    std::mutex align_mutex_;
    bool model_loaded_;
    
public:
    AlignServer(const ServerConfig& config);
    ~AlignServer();
    
    bool init();
    void run();
    
    std::string handle_align(
        const std::string& text,
        const std::vector<int16_t>& pcm_data,
        const std::string& language
    );
    
    bool is_model_loaded() const { return model_loaded_; }
    const ServerConfig& get_config() const { return config_; }
};

}