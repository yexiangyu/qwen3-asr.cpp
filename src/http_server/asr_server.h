#pragma once

#include "batch_scheduler.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace qwen3_asr {

// Forward declarations
class Qwen3ASR;
class ForcedAligner;

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
    int max_batch_size = 2;
    int batch_timeout_ms = 100;
};

class CombinedASRServer {
private:
    CombinedServerConfig config_;
    std::unique_ptr<Qwen3ASR> asr_;
    std::unique_ptr<ForcedAligner> aligner_;
    std::unique_ptr<BatchScheduler> batch_scheduler_;
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