#include "align_server.h"
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

static std::string build_success_response(const qwen3alignment_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"processing_time_ms\": " << result.t_total_ms << ",\n";
    json << "  \"n_utterances\": " << result.n_utterances << ",\n";
    json << "  \"utterances\": [\n";
    
    for (int i = 0; i < result.n_utterances; ++i) {
        const auto& utt = result.utterances[i];
        json << "    {\n";
        json << "      \"start\": " << std::fixed << std::setprecision(3) << utt.start << ",\n";
        json << "      \"end\": " << std::fixed << std::setprecision(3) << utt.end << ",\n";
        json << "      \"text\": \"" << escape_json_string(utt.text) << "\",\n";
        json << "      \"n_words\": " << utt.n_words << ",\n";
        json << "      \"words\": [\n";
        
        for (int j = 0; j < utt.n_words; ++j) {
            const auto& w = utt.words[j];
            json << "        {";
            json << "\"word\": \"" << escape_json_string(w.word) << "\", ";
            json << "\"start\": " << std::fixed << std::setprecision(3) << w.start << ", ";
            json << "\"end\": " << std::fixed << std::setprecision(3) << w.end << ", ";
            json << "\"conf_word\": " << std::fixed << std::setprecision(4) << w.conf_word << ", ";
            json << "\"conf_start_time\": " << std::fixed << std::setprecision(4) << w.conf_start_time << ", ";
            json << "\"conf_end_time\": " << std::fixed << std::setprecision(4) << w.conf_end_time;
            json << "}";
            if (j + 1 < utt.n_words) json << ",";
            json << "\n";
        }
        
        json << "      ]\n";
        json << "    }";
        if (i + 1 < result.n_utterances) json << ",";
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

AlignServer::AlignServer(const ServerConfig& config)
    : config_(config)
    , handle_(nullptr)
    , model_loaded_(false)
{
}

AlignServer::~AlignServer() {
    if (handle_) {
        qwen3aligner_free(handle_);
        handle_ = nullptr;
    }
}

bool AlignServer::init() {
    LOG_INFO("Initializing AlignServer...");
    LOG_INFO("  Model path: {}", config_.aligner_model_path);
    LOG_INFO("  Korean dict: {}", config_.korean_dict_path.empty() ? "(none)" : config_.korean_dict_path);
    LOG_INFO("  Threads: {}", config_.n_threads);
    LOG_INFO("  Device: {}", config_.device.empty() ? "(auto)" : config_.device);
    
    int ret;
    if (!config_.device.empty()) {
        ret = qwen3aligner_init_with_device_name(&handle_, config_.device.c_str());
    } else {
        ret = qwen3aligner_init(&handle_);
    }
    
    if (ret != 0) {
        LOG_ERROR("Failed to init aligner handle: {}", qwen3_get_last_error());
        return false;
    }
    
    ret = qwen3aligner_load_model(handle_, config_.aligner_model_path.c_str());
    if (ret != 0) {
        LOG_ERROR("Failed to load aligner model: {}", qwen3_get_last_error());
        qwen3aligner_free(handle_);
        handle_ = nullptr;
        return false;
    }
    
    if (!config_.korean_dict_path.empty()) {
        ret = qwen3aligner_load_korean_dict(handle_, config_.korean_dict_path.c_str());
        if (ret != 0) {
            LOG_WARN("Failed to load Korean dict: {}", qwen3_get_last_error());
        }
    }
    
    model_loaded_ = true;
    LOG_INFO("AlignServer initialized successfully on device: {}", qwen3aligner_get_device_name(handle_));
    return true;
}

std::string AlignServer::handle_align(
    const std::string& text,
    const std::vector<int16_t>& pcm_data,
    const std::string& language
) {
    std::lock_guard<std::mutex> lock(align_mutex_);
    
    LOG_INFO("Processing align request: text_len={}, pcm_samples={}, lang={}",
             text.size(), pcm_data.size(), language.empty() ? "(auto)" : language);
    
    qwen3aligner_params params;
    params.language = language.empty() ? nullptr : language.c_str();
    params.n_threads = config_.n_threads;
    
    qwen3alignment_result result;
    int ret = qwen3aligner_align_pcm(
        handle_,
        pcm_data.data(),
        static_cast<int32_t>(pcm_data.size()),
        text.c_str(),
        &params,
        &result
    );
    
    if (ret != 0) {
        LOG_ERROR("Align failed: {}", qwen3_get_last_error());
        return build_error_response(qwen3_get_last_error());
    }
    
    std::string json_response = build_success_response(result);
    
    LOG_INFO("Align completed: {} utterances, {} ms", result.n_utterances, result.t_total_ms);
    
    qwen3aligner_free_result(&result);
    
    return json_response;
}

}