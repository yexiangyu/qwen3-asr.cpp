#include "asr_server.h"
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

static std::string build_transcribe_response(const qwen3asr_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"text\": \"" << escape_json_string(result.text) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string(result.text_content) << "\",\n";
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
    
    json << "  ],\n";
    json << "  \"mel_ms\": " << result.t_mel_ms << ",\n";
    json << "  \"encode_ms\": " << result.t_encode_ms << ",\n";
    json << "  \"decode_ms\": " << result.t_decode_ms << ",\n";
    json << "  \"total_ms\": " << result.t_total_ms << "\n";
    json << "}\n";
    
    return json.str();
}

static std::string build_align_response(const qwen3alignment_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
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
    
    json << "  ],\n";
    json << "  \"mel_ms\": " << result.t_mel_ms << ",\n";
    json << "  \"encode_ms\": " << result.t_encode_ms << ",\n";
    json << "  \"decode_ms\": " << result.t_decode_ms << ",\n";
    json << "  \"total_ms\": " << result.t_total_ms << "\n";
    json << "}\n";
    
    return json.str();
}

static std::string build_combined_response(
    const qwen3asr_result& asr_result,
    const qwen3alignment_result& align_result
) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"text\": \"" << escape_json_string(asr_result.text) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string(asr_result.text_content) << "\",\n";
    json << "  \"n_tokens\": " << asr_result.n_tokens << ",\n";
    json << "  \"n_utterances\": " << align_result.n_utterances << ",\n";
    json << "  \"utterances\": [\n";
    
    for (int i = 0; i < align_result.n_utterances; ++i) {
        const auto& utt = align_result.utterances[i];
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
        if (i + 1 < align_result.n_utterances) json << ",";
        json << "\n";
    }
    
    json << "  ],\n";
    json << "  \"transcribe_mel_ms\": " << asr_result.t_mel_ms << ",\n";
    json << "  \"transcribe_encode_ms\": " << asr_result.t_encode_ms << ",\n";
    json << "  \"transcribe_decode_ms\": " << asr_result.t_decode_ms << ",\n";
    json << "  \"transcribe_total_ms\": " << asr_result.t_total_ms << ",\n";
    json << "  \"alignment_mel_ms\": " << align_result.t_mel_ms << ",\n";
    json << "  \"alignment_encode_ms\": " << align_result.t_encode_ms << ",\n";
    json << "  \"alignment_decode_ms\": " << align_result.t_decode_ms << ",\n";
    json << "  \"alignment_total_ms\": " << align_result.t_total_ms << ",\n";
    json << "  \"total_ms\": " << (asr_result.t_total_ms + align_result.t_total_ms) << "\n";
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

CombinedASRServer::CombinedASRServer(const CombinedServerConfig& config)
    : config_(config)
    , asr_handle_(nullptr)
    , aligner_handle_(nullptr)
    , models_loaded_(false)
{
}

CombinedASRServer::~CombinedASRServer() {
    if (asr_handle_) {
        qwen3asr_free(asr_handle_);
        asr_handle_ = nullptr;
    }
    if (aligner_handle_) {
        qwen3aligner_free(aligner_handle_);
        aligner_handle_ = nullptr;
    }
}

bool CombinedASRServer::init() {
    LOG_INFO("Initializing CombinedASRServer...");
    LOG_INFO("  ASR model: {}", config_.asr_model_path);
    LOG_INFO("  Aligner model: {}", config_.aligner_model_path);
    LOG_INFO("  Korean dict: {}", config_.korean_dict_path.empty() ? "(none)" : config_.korean_dict_path);
    LOG_INFO("  Threads: {}", config_.n_threads);
    LOG_INFO("  ASR device: {}", config_.asr_device.empty() ? "(auto)" : config_.asr_device);
    LOG_INFO("  Aligner device: {}", config_.aligner_device.empty() ? "(auto)" : config_.aligner_device);
    
    int ret;
    
    if (!config_.asr_device.empty()) {
        ret = qwen3asr_init_with_device_name(&asr_handle_, config_.asr_device.c_str());
    } else {
        ret = qwen3asr_init(&asr_handle_);
    }
    
    if (ret != 0) {
        LOG_ERROR("Failed to init ASR handle: {}", qwen3_get_last_error());
        return false;
    }
    
    ret = qwen3asr_load_model(asr_handle_, config_.asr_model_path.c_str());
    if (ret != 0) {
        LOG_ERROR("Failed to load ASR model: {}", qwen3_get_last_error());
        qwen3asr_free(asr_handle_);
        asr_handle_ = nullptr;
        return false;
    }
    
    if (!config_.aligner_device.empty()) {
        ret = qwen3aligner_init_with_device_name(&aligner_handle_, config_.aligner_device.c_str());
    } else {
        ret = qwen3aligner_init(&aligner_handle_);
    }
    
    if (ret != 0) {
        LOG_ERROR("Failed to init aligner handle: {}", qwen3_get_last_error());
        qwen3asr_free(asr_handle_);
        asr_handle_ = nullptr;
        return false;
    }
    
    ret = qwen3aligner_load_model(aligner_handle_, config_.aligner_model_path.c_str());
    if (ret != 0) {
        LOG_ERROR("Failed to load aligner model: {}", qwen3_get_last_error());
        qwen3asr_free(asr_handle_);
        asr_handle_ = nullptr;
        qwen3aligner_free(aligner_handle_);
        aligner_handle_ = nullptr;
        return false;
    }
    
    if (!config_.korean_dict_path.empty()) {
        ret = qwen3aligner_load_korean_dict(aligner_handle_, config_.korean_dict_path.c_str());
        if (ret != 0) {
            LOG_WARN("Failed to load Korean dict: {}", qwen3_get_last_error());
        }
    }
    
    models_loaded_ = true;
    LOG_INFO("CombinedASRServer initialized successfully");
    LOG_INFO("  ASR device: {}", qwen3asr_get_device_name(asr_handle_));
    LOG_INFO("  Aligner device: {}", qwen3aligner_get_device_name(aligner_handle_));
    return true;
}

std::string CombinedASRServer::handle_transcribe(
    const std::vector<int16_t>& pcm_data,
    const std::string& language,
    const std::string& context,
    int max_tokens
) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    
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
        asr_handle_,
        pcm_data.data(),
        static_cast<int32_t>(pcm_data.size()),
        &params,
        &result
    );
    
    if (ret != 0) {
        LOG_ERROR("Transcribe failed: {}", qwen3_get_last_error());
        return build_error_response(qwen3_get_last_error());
    }
    
    std::string json_response = build_transcribe_response(result);
    
    LOG_INFO("Transcribe completed: {} tokens, {} ms", result.n_tokens, result.t_total_ms);
    
    qwen3asr_free_result(&result);
    
    return json_response;
}

std::string CombinedASRServer::handle_align(
    const std::string& text,
    const std::vector<int16_t>& pcm_data,
    const std::string& language
) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    
    LOG_INFO("Processing align request: text_len={}, pcm_samples={}, lang={}",
             text.size(), pcm_data.size(), language.empty() ? "(auto)" : language);
    
    qwen3aligner_params params;
    params.language = language.empty() ? nullptr : language.c_str();
    params.n_threads = config_.n_threads;
    
    qwen3alignment_result result;
    int ret = qwen3aligner_align_pcm(
        aligner_handle_,
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
    
    std::string json_response = build_align_response(result);
    
    LOG_INFO("Align completed: {} utterances, {} ms", result.n_utterances, result.t_total_ms);
    
    qwen3aligner_free_result(&result);
    
    return json_response;
}

std::string CombinedASRServer::handle_transcribe_align(
    const std::vector<int16_t>& pcm_data,
    const std::string& language,
    const std::string& context,
    int max_tokens
) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    
    LOG_INFO("Processing transcribe-align request: pcm_samples={}, lang={}, context_len={}, max_tokens={}",
             pcm_data.size(), language.empty() ? "(auto)" : language, 
             context.size(), max_tokens);
    
    qwen3asr_params asr_params;
    asr_params.max_tokens = max_tokens > 0 ? max_tokens : config_.max_tokens;
    asr_params.language = language.empty() ? nullptr : language.c_str();
    asr_params.context = context.empty() ? nullptr : context.c_str();
    asr_params.n_threads = config_.n_threads;
    
    qwen3aligner_params align_params;
    align_params.language = language.empty() ? nullptr : language.c_str();
    align_params.n_threads = config_.n_threads;
    
    qwen3combined_result combined_result;
    int ret = qwen3asr_transcribe_align_pcm_combined(
        asr_handle_,
        aligner_handle_,
        pcm_data.data(),
        static_cast<int32_t>(pcm_data.size()),
        &asr_params,
        &align_params,
        &combined_result
    );
    
    if (ret != 0) {
        LOG_ERROR("Transcribe-align failed: {}", qwen3_get_last_error());
        return build_error_response(qwen3_get_last_error());
    }
    
    std::string json_response = build_combined_response(combined_result.transcription, combined_result.alignment);
    
    LOG_INFO("Transcribe-align completed: {} tokens, {} utterances, total {} ms",
             combined_result.transcription.n_tokens, combined_result.alignment.n_utterances,
             combined_result.transcription.t_total_ms + combined_result.alignment.t_total_ms);
    
    qwen3asr_free_combined_result(&combined_result);
    
    return json_response;
}

}