#include "batch_scheduler.h"
#include "qwen3asr_c_api.h"
#include "logger.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace qwen3_asr {

static std::string escape_json_string_local(const char* s) {
    if (!s) return "";
    std::string result;
    while (*s) {
        char c = *s++;
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
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

static std::string build_transcribe_json_response(const qwen3asr_result& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"success\": true,\n";
    json << "  \"text\": \"" << escape_json_string_local(result.text) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string_local(result.text_content) << "\",\n";
    json << "  \"n_tokens\": " << result.n_tokens << ",\n";
    json << "  \"mel_ms\": " << result.t_mel_ms << ",\n";
    json << "  \"encode_ms\": " << result.t_encode_ms << ",\n";
    json << "  \"decode_ms\": " << result.t_decode_ms << ",\n";
    json << "  \"total_ms\": " << result.t_total_ms << "\n";
    json << "}\n";
    return json.str();
}

static std::string build_transcribe_align_json_response(const qwen3alignment_result& result) {
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
        json << "      \"text\": \"" << escape_json_string_local(utt.text) << "\",\n";
        json << "      \"n_words\": " << utt.n_words << ",\n";
        json << "      \"words\": [\n";
        
        for (int j = 0; j < utt.n_words; ++j) {
            const auto& w = utt.words[j];
            json << "        {";
            json << "\"word\": \"" << escape_json_string_local(w.word) << "\", ";
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

BatchScheduler::BatchScheduler() 
    : asr_handle_(nullptr)
    , aligner_handle_(nullptr)
    , running_(false)
    , next_request_id_(0)
{
}

BatchScheduler::~BatchScheduler() {
    stop();
}

void BatchScheduler::set_asr(void* asr_handle) {
    asr_handle_ = asr_handle;
}

void BatchScheduler::set_aligner(void* aligner_handle) {
    aligner_handle_ = aligner_handle;
}

void BatchScheduler::start() {
    if (running_) return;
    
    running_ = true;
    worker_thread_ = std::thread(&BatchScheduler::batch_worker, this);
    
    LOG_INFO("BatchScheduler started: batch_size={}, timeout={}ms", 
             config_.max_batch_size, config_.batch_timeout_ms);
}

void BatchScheduler::stop() {
    if (!running_) return;
    
    running_ = false;
    batch_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    LOG_INFO("BatchScheduler stopped");
}

void BatchScheduler::set_batch_size(int size) {
    config_.max_batch_size = std::max(1, std::min(32, size));
    LOG_INFO("Batch size set to {}", config_.max_batch_size);
}

void BatchScheduler::set_timeout_ms(int ms) {
    config_.batch_timeout_ms = std::max(10, std::min(1000, ms));
    LOG_INFO("Batch timeout set to {}ms", config_.batch_timeout_ms);
}

std::future<std::string> BatchScheduler::submit_request(
    const std::vector<int16_t>& pcm,
    const std::string& language,
    const std::string& context,
    int max_tokens,
    RequestType type
) {
    ASRRequest req;
    req.pcm_data = pcm;
    req.language = language;
    req.context = context;
    req.max_tokens = max_tokens;
    req.request_id = next_request_id_.fetch_add(1);
    req.type = type;
    
    std::future<std::string> future = req.result_promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_queue_.push_back(std::move(req));
    }
    
    batch_cv_.notify_one();
    
    LOG_DEBUG("Request {} submitted (type={}), pending count: {}", 
              req.request_id, static_cast<int>(type), pending_queue_.size());
    
    return future;
}

int BatchScheduler::get_pending_count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mutex_));
    return static_cast<int>(pending_queue_.size());
}

bool BatchScheduler::is_running() const {
    return running_;
}

void BatchScheduler::batch_worker() {
    LOG_INFO("Batch worker thread started");
    
    while (running_) {
        std::vector<ASRRequest> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            bool timeout_reached = !batch_cv_.wait_for(
                lock, 
                std::chrono::milliseconds(config_.batch_timeout_ms),
                [this] { 
                    return !running_ || pending_queue_.size() >= static_cast<size_t>(config_.max_batch_size); 
                }
            );
            
            if (!running_) break;
            
            int batch_size = std::min(
                static_cast<int>(pending_queue_.size()),
                config_.max_batch_size
            );
            
            for (int i = 0; i < batch_size; ++i) {
                batch.push_back(std::move(pending_queue_.front()));
                pending_queue_.pop_front();
            }
        }
        
        if (!batch.empty()) {
            LOG_INFO("Processing batch of {} requests", batch.size());
            process_batch(batch);
        }
    }
    
    LOG_INFO("Batch worker thread exiting");
}

void BatchScheduler::process_batch(std::vector<ASRRequest>& batch) {
    if (!asr_handle_) {
        LOG_ERROR("ASR handle not set");
        for (auto& req : batch) {
            req.result_promise.set_value("{\"success\":false,\"error\":\"ASR handle not set\"}");
        }
        return;
    }
    
    auto batch_start = std::chrono::steady_clock::now();
    
    qwen3asr_handle asr = static_cast<qwen3asr_handle>(asr_handle_);
    qwen3aligner_handle aligner = aligner_handle_ ? static_cast<qwen3aligner_handle>(aligner_handle_) : nullptr;
    
    // Process requests sequentially within the batch
    // Note: True parallel batch processing would require refactoring to use 
    // transcribe_batch() API which processes multiple audio files in parallel.
    // Current implementation batches requests to reduce overhead but processes
    // them sequentially for correctness.
    LOG_INFO("Processing {} requests in batch (sequential within batch)", batch.size());
    
    for (auto& req : batch) {
        try {
            LOG_INFO("Processing request {} type={}", req.request_id, static_cast<int>(req.type));
            
            if (req.type == RequestType::TRANSCRIBE_ALIGN && aligner) {
                // Full transcribe-align pipeline
                LOG_INFO("Request {}: Running transcribe-align pipeline", req.request_id);
                LOG_INFO("Request {}: language={}, context={}, max_tokens={}", 
                         req.request_id, req.language, req.context, req.max_tokens);
                // Full transcribe-align pipeline
                qwen3asr_params asr_params;
                asr_params.max_tokens = req.max_tokens > 0 ? req.max_tokens : 1024;
                asr_params.language = req.language.empty() ? nullptr : req.language.c_str();
                asr_params.context = req.context.empty() ? nullptr : req.context.c_str();
                asr_params.n_threads = 4;
                
                qwen3aligner_params align_params;
                align_params.language = req.language.empty() ? nullptr : req.language.c_str();
                align_params.n_threads = 4;
                
                qwen3alignment_result align_result;
                LOG_INFO("Request {}: Calling qwen3asr_transcribe_and_align_pcm with max_tokens={}", 
                         req.request_id, asr_params.max_tokens);
                int ret = qwen3asr_transcribe_and_align_pcm(
                    asr,
                    aligner,
                    req.pcm_data.data(),
                    static_cast<int32_t>(req.pcm_data.size()),
                    &asr_params,
                    &align_params,
                    &align_result
                );
                
                LOG_INFO("Request {}: qwen3asr_transcribe_and_align_pcm returned {}", req.request_id, ret);
                
                if (ret != 0) {
                    std::string error = qwen3_get_last_error();
                    LOG_ERROR("Request {} transcribe-align failed: {}", req.request_id, error);
                    req.result_promise.set_value("{\"success\":false,\"error\":\"" + error + "\"}");
                } else {
                    std::string json_response = build_transcribe_align_json_response(align_result);
                    req.result_promise.set_value(json_response);
                    qwen3aligner_free_result(&align_result);
                    LOG_DEBUG("Request {} transcribe-align completed", req.request_id);
                }
            } else {
                // Transcription only
                qwen3asr_params params;
                params.max_tokens = req.max_tokens > 0 ? req.max_tokens : 1024;
                params.language = req.language.empty() ? nullptr : req.language.c_str();
                params.context = req.context.empty() ? nullptr : req.context.c_str();
                params.n_threads = 4;
                
                qwen3asr_result result;
                int ret = qwen3asr_transcribe_pcm(
                    asr,
                    req.pcm_data.data(),
                    static_cast<int32_t>(req.pcm_data.size()),
                    &params,
                    &result
                );
                
                if (ret != 0) {
                    std::string error = qwen3_get_last_error();
                    LOG_ERROR("Request {} failed: {}", req.request_id, error);
                    req.result_promise.set_value("{\"success\":false,\"error\":\"" + error + "\"}");
                } else {
                    std::string json_response = build_transcribe_json_response(result);
                    req.result_promise.set_value(json_response);
                    qwen3asr_free_result(&result);
                    LOG_DEBUG("Request {} completed", req.request_id);
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception processing request {}: {}", req.request_id, e.what());
            req.result_promise.set_value("{\"success\":false,\"error\":\"Internal error\"}");
        }
    }
    
    auto batch_end = std::chrono::steady_clock::now();
    auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
    
    LOG_INFO("Batch of {} requests processed in {}ms (avg {}ms per request)",
             batch.size(), batch_ms, batch.size() > 0 ? batch_ms / batch.size() : 0);
}

}