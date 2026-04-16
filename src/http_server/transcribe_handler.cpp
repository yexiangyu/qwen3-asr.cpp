#include "transcribe_server.h"
#include "logger.h"

#include <sstream>
#include <iomanip>
#include <mutex>
#include <cstring>

namespace qwen3_asr {

static std::string escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

static std::string build_success_response(const qwen3asr_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"text\": \"" << escape_json_string(result.text) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string(result.text_content) << "\",\n";
    json << "  \"processing_time_ms\": " << result.t_total_ms << ",\n";
    json << "  \"n_tokens\": " << result.n_tokens << ",\n";
    json << "  \"tokens\": [\n";
    
    for (int32_t i = 0; i < result.n_tokens; ++i) {
        json << "    {";
        json << "\"id\": " << result.token_ids[i] << ", ";
        json << "\"confidence\": " << std::fixed << std::setprecision(4) << result.token_confs[i];
        json << "}";
        if (i + 1 < result.n_tokens) json << ",";
        json << "\n";
    }
    
    json << "  ]\n";
    json << "}\n";
    
    return json.str();
}

static std::string build_error_response(const std::string& error_msg) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": false,\n";
    json << "  \"error\": \"" << escape_json_string(error_msg) << "\"\n";
    json << "}\n";
    return json.str();
}

ASRServer::ASRServer(const ASRServerConfig& config)
    : config_(config)
    , handle_(nullptr)
    , model_loaded_(false)
{
}

ASRServer::~ASRServer() {
    if (handle_) {
        qwen3asr_free(handle_);
        handle_ = nullptr;
    }
}

bool ASRServer::init() {
    LOG_INFO("Initializing ASRServer...");
    LOG_INFO("  Model path: {}", config_.asr_model_path);
    LOG_INFO("  Threads: {}", config_.n_threads);
    LOG_INFO("  Max tokens: {}", config_.max_tokens);
    LOG_INFO("  Device: {}", config_.device.empty() ? "(auto)" : config_.device);
    
    int ret;
    if (!config_.device.empty()) {
        ret = qwen3asr_init_with_device_name(&handle_, config_.device.c_str());
    } else {
        ret = qwen3asr_init(&handle_);
    }
    
    if (ret != 0) {
        LOG_ERROR("Failed to init ASR handle: {}", qwen3_get_last_error());
        return false;
    }
    
    ret = qwen3asr_load_model(handle_, config_.asr_model_path.c_str());
    if (ret != 0) {
        LOG_ERROR("Failed to load ASR model: {}", qwen3_get_last_error());
        qwen3asr_free(handle_);
        handle_ = nullptr;
        return false;
    }
    
    model_loaded_ = true;
    LOG_INFO("ASRServer initialized successfully on device: {}", qwen3asr_get_device_name(handle_));
    return true;
}

std::string ASRServer::handle_transcribe(
    const std::vector<int16_t>& pcm_data,
    const std::string& language,
    const std::string& context,
    int max_tokens
) {
    std::lock_guard<std::mutex> lock(asr_mutex_);
    
    LOG_INFO("Processing transcribe request: pcm_samples={}, lang={}, context_len={}, max_tokens={}",
             pcm_data.size(), language.empty() ? "(auto)" : language, 
             context.size(), max_tokens);
    
    qwen3asr_params params;
    params.max_tokens = max_tokens > 0 ? max_tokens : config_.max_tokens;
    params.language = language.empty() ? nullptr : language.c_str();
    params.context = context.empty() ? nullptr : context.c_str();
    params.n_threads = config_.n_threads;
    
    qwen3asr_result result;
    int ret = qwen3asr_transcribe_pcm(
        handle_,
        pcm_data.data(),
        static_cast<int32_t>(pcm_data.size()),
        &params,
        &result
    );
    
    if (ret != 0) {
        LOG_ERROR("Transcribe failed: {}", qwen3_get_last_error());
        return build_error_response(qwen3_get_last_error());
    }
    
    std::string json_response = build_success_response(result);
    
    LOG_INFO("Transcribe completed: {} tokens, {} ms", result.n_tokens, result.t_total_ms);
    
    qwen3asr_free_result(&result);
    
    return json_response;
}

}