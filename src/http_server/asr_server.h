#pragma once

#include "qwen3asr_c_api.h"
#include <string>
#include <mutex>
#include <vector>
#include <cstdint>

namespace qwen3_asr {

struct CombinedServerConfig {
    std::string asr_model_path;
    std::string aligner_model_path;
    std::string korean_dict_path;
    int port = 8082;
    int n_threads = 4;
    int max_tokens = 1024;
    std::string default_language = "";
    std::string asr_device = "";
    std::string aligner_device = "";
};

class CombinedASRServer {
private:
    CombinedServerConfig config_;
    qwen3asr_handle asr_handle_;
    qwen3aligner_handle aligner_handle_;
    std::mutex server_mutex_;
    bool models_loaded_;
    
public:
    CombinedASRServer(const CombinedServerConfig& config);
    ~CombinedASRServer();
    
    bool init();
    
    std::string handle_transcribe(
        const std::vector<int16_t>& pcm_data,
        const std::string& language,
        const std::string& context,
        int max_tokens
    );
    
    std::string handle_align(
        const std::string& text,
        const std::vector<int16_t>& pcm_data,
        const std::string& language
    );
    
    std::string handle_transcribe_align(
        const std::vector<int16_t>& pcm_data,
        const std::string& language,
        const std::string& context,
        int max_tokens
    );
    
    bool is_models_loaded() const { return models_loaded_; }
    const CombinedServerConfig& get_config() const { return config_; }
};

}