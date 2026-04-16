#include "asr_server.h"
#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "logger.h"

#include <sstream>
#include <iomanip>
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

static std::string build_transcribe_response(const transcribe_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"text\": \"" << escape_json_string(result.text) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string(result.text_content) << "\",\n";
    json << "  \"n_tokens\": " << result.tokens.size() << ",\n";
    json << "  \"mel_ms\": " << result.t_mel_ms << ",\n";
    json << "  \"encode_ms\": " << result.t_encode_ms << ",\n";
    json << "  \"decode_ms\": " << result.t_decode_ms << ",\n";
    json << "  \"total_ms\": " << result.t_total_ms << "\n";
    json << "}\n";
    
    return json.str();
}

static std::string build_align_response(const alignment_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"n_utterances\": " << result.utterances.size() << ",\n";
    json << "  \"utterances\": [\n";
    
    for (size_t i = 0; i < result.utterances.size(); ++i) {
        const auto& utt = result.utterances[i];
        json << "    {\n";
        json << "      \"start\": " << std::fixed << std::setprecision(3) << utt.start << ",\n";
        json << "      \"end\": " << std::fixed << std::setprecision(3) << utt.end << ",\n";
        json << "      \"text\": \"" << escape_json_string(utt.text) << "\",\n";
        json << "      \"n_words\": " << utt.words.size() << ",\n";
        json << "      \"words\": [\n";
        
        for (size_t j = 0; j < utt.words.size(); ++j) {
            const auto& w = utt.words[j];
            json << "        {";
            json << "\"word\": \"" << escape_json_string(w.word) << "\", ";
            json << "\"start\": " << std::fixed << std::setprecision(3) << w.start << ", ";
            json << "\"end\": " << std::fixed << std::setprecision(3) << w.end << ", ";
            json << "\"conf_word\": " << std::fixed << std::setprecision(4) << w.conf_word << ", ";
            json << "\"conf_start_time\": " << std::fixed << std::setprecision(4) << w.conf_start_time << ", ";
            json << "\"conf_end_time\": " << std::fixed << std::setprecision(4) << w.conf_end_time;
            json << "}";
            if (j + 1 < utt.words.size()) json << ",";
            json << "\n";
        }
        
        json << "      ]\n";
        json << "    }";
        if (i + 1 < result.utterances.size()) json << ",";
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
    , models_loaded_(false)
{
}

CombinedASRServer::~CombinedASRServer() {
    if (batch_scheduler_) {
        batch_scheduler_->stop();
        batch_scheduler_.reset();
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
    
    // Create ASR object
    asr_ = std::make_unique<Qwen3ASR>();
    
    // Load ASR model with device selection
    if (!asr_->load_model(config_.asr_model_path, config_.asr_device)) {
        LOG_ERROR("Failed to load ASR model: {}", asr_->get_error());
        asr_.reset();
        return false;
    }
    
    // Create Aligner object
    aligner_ = std::make_unique<ForcedAligner>();
    
    // Load Aligner model with device selection
    if (!aligner_->load_model(config_.aligner_model_path, config_.aligner_device)) {
        LOG_ERROR("Failed to load aligner model: {}", aligner_->get_error());
        asr_.reset();
        aligner_.reset();
        return false;
    }
    
    // Load Korean dictionary if specified
    if (!config_.korean_dict_path.empty()) {
        if (!aligner_->load_korean_dict(config_.korean_dict_path)) {
            LOG_WARN("Failed to load Korean dict: {}", aligner_->get_error());
        }
    }
    
    // Initialize batch scheduler with C++ objects
    batch_scheduler_ = std::make_unique<BatchScheduler>();
    batch_scheduler_->set_asr(asr_.get());
    batch_scheduler_->set_aligner(aligner_.get());
    batch_scheduler_->set_batch_size(config_.max_batch_size);
    batch_scheduler_->set_timeout_ms(config_.batch_timeout_ms);
    batch_scheduler_->start();
    
    models_loaded_ = true;
    LOG_INFO("CombinedASRServer initialized successfully");
    LOG_INFO("  Batch scheduler: size={}, timeout={}ms", config_.max_batch_size, config_.batch_timeout_ms);
    return true;
}

std::string CombinedASRServer::handle_transcribe(
    const std::vector<int16_t>& pcm_data,
    const std::string& language,
    const std::string& context,
    int max_tokens
) {
    LOG_INFO("Processing transcribe request: pcm_samples={}, lang={}, context_len={}, max_tokens={}",
             pcm_data.size(), language.empty() ? "(auto)" : language, 
             context.size(), max_tokens);
    
    // Submit to batch scheduler
    auto future = batch_scheduler_->submit_request(
        pcm_data, language, context, max_tokens,
        RequestType::TRANSCRIBE
    );
    return future.get();
}

std::string CombinedASRServer::handle_align(
    const std::string& text,
    const std::vector<int16_t>& pcm_data,
    const std::string& language
) {
    LOG_INFO("Processing align request: text_len={}, pcm_samples={}, lang={}",
             text.size(), pcm_data.size(), language.empty() ? "(auto)" : language);
    
    // Convert PCM to float
    std::vector<float> float_samples(pcm_data.size());
    for (size_t i = 0; i < pcm_data.size(); ++i) {
        float_samples[i] = pcm_data[i] / 32768.0f;
    }
    
    align_params ap;
    
    auto result = aligner_->align(
        float_samples.data(),
        static_cast<int>(float_samples.size()),
        text,
        language,
        ap
    );
    
    if (!result.success) {
        LOG_ERROR("Align failed: {}", result.error_msg);
        return build_error_response(result.error_msg);
    }
    
    LOG_INFO("Align completed: {} utterances, {} ms", result.utterances.size(), result.t_total_ms);
    return build_align_response(result);
}

std::string CombinedASRServer::handle_transcribe_align(
    const std::vector<int16_t>& pcm_data,
    const std::string& language,
    const std::string& context,
    int max_tokens
) {
    LOG_INFO("Processing transcribe-align request: pcm_samples={}, lang={}, context_len={}, max_tokens={}",
             pcm_data.size(), language.empty() ? "(auto)" : language, 
             context.size(), max_tokens);
    
    auto future = batch_scheduler_->submit_request(
        pcm_data, language, context, max_tokens,
        RequestType::TRANSCRIBE_ALIGN
    );
    return future.get();
}

}
