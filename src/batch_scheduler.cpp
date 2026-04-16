#include "batch_scheduler.h"
#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "logger.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace qwen3_asr {

static std::string escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
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

BatchScheduler::BatchScheduler() 
    : asr_(nullptr)
    , aligner_(nullptr)
    , running_(false)
    , next_request_id_(0)
{
}

BatchScheduler::~BatchScheduler() {
    stop();
}

void BatchScheduler::set_asr(Qwen3ASR* asr) {
    asr_ = asr;
}

void BatchScheduler::set_aligner(ForcedAligner* aligner) {
    aligner_ = aligner;
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
    config_.batch_timeout_ms = std::max(10, std::min(60000, ms));
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
        bool batch_full = false;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Step 1: Wait for at least one request
            batch_cv_.wait(lock, [this] { 
                return !running_ || !pending_queue_.empty(); 
            });
            
            if (!running_) break;
            
            auto wait_start = std::chrono::steady_clock::now();
            
            // Step 2: Wait for batch full OR timeout
            // If batch is already full (max_batch_size=1), skip waiting
            if (pending_queue_.size() < static_cast<size_t>(config_.max_batch_size)) {
                while (running_) {
                    auto elapsed = std::chrono::steady_clock::now() - wait_start;
                    auto remaining = std::chrono::milliseconds(config_.batch_timeout_ms) - 
                                     std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
                    
                    if (remaining.count() <= 0) {
                        // Timeout expired
                        LOG_INFO("Batch timeout after {}ms (queue: {}/{})", 
                                 config_.batch_timeout_ms, pending_queue_.size(), config_.max_batch_size);
                        break;
                    }
                    
                    // Wait for new request or timeout
                    auto status = batch_cv_.wait_for(lock, remaining);
                    
                    if (!running_) break;
                    
                    // Check if batch is full
                    if (pending_queue_.size() >= static_cast<size_t>(config_.max_batch_size)) {
                        batch_full = true;
                        LOG_INFO("Batch full after {}ms (queue: {}/{})", 
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - wait_start).count(),
                                 pending_queue_.size(), config_.max_batch_size);
                        break;
                    }
                    
                    // If notified but not full, continue waiting
                    if (status == std::cv_status::no_timeout) {
                        LOG_DEBUG("New request arrived (queue: {}), continuing to wait for batch fill", 
                                  pending_queue_.size());
                    }
                }
            } else {
                batch_full = true;
                LOG_INFO("Batch immediately full (queue: {}/{})", 
                         pending_queue_.size(), config_.max_batch_size);
            }
            
            if (!running_) break;
            
            // Take all available requests (up to max_batch_size)
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
    if (!asr_) {
        LOG_ERROR("ASR object not set");
        for (auto& req : batch) {
            req.result_promise.set_value("{\"success\":false,\"error\":\"ASR not initialized\"}");
        }
        return;
    }
    
    auto batch_start = std::chrono::steady_clock::now();
    
    // Check if all requests are TRANSCRIBE_ALIGN
    bool all_transcribe_align = true;
    for (const auto& req : batch) {
        if (req.type != RequestType::TRANSCRIBE_ALIGN || !aligner_) {
            all_transcribe_align = false;
            break;
        }
    }
    
    if (all_transcribe_align && batch.size() > 1) {
        process_batch_transcribe_align(batch);
    } else {
        // Mixed types or single request - process sequentially
        for (auto& req : batch) {
            if (req.type == RequestType::TRANSCRIBE_ALIGN && aligner_) {
                // Single transcribe-align
                process_batch_transcribe_align(batch);
                break;
            } else {
                // Transcribe only
                process_batch_transcribe_only(batch);
                break;
            }
        }
    }
    
    auto batch_end = std::chrono::steady_clock::now();
    auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
    
    LOG_INFO("Batch of {} requests processed in {}ms (avg {}ms per request)",
             batch.size(), batch_ms, batch.size() > 0 ? batch_ms / batch.size() : 0);
}

void BatchScheduler::process_batch_transcribe_align(std::vector<ASRRequest>& batch) {
    if (!asr_ || !aligner_) {
        LOG_ERROR("ASR or Aligner not set");
        for (auto& req : batch) {
            req.result_promise.set_value("{\"success\":false,\"error\":\"Model not initialized\"}");
        }
        return;
    }
    
    // Step 1: Prepare float samples for batch transcription
    std::vector<std::vector<float>> float_samples_list(batch.size());
    std::vector<const float*> audio_samples(batch.size());
    std::vector<int> n_samples_list(batch.size());
    
    for (size_t i = 0; i < batch.size(); ++i) {
        float_samples_list[i].resize(batch[i].pcm_data.size());
        for (size_t j = 0; j < batch[i].pcm_data.size(); ++j) {
            float_samples_list[i][j] = batch[i].pcm_data[j] / 32768.0f;
        }
        audio_samples[i] = float_samples_list[i].data();
        n_samples_list[i] = static_cast<int>(batch[i].pcm_data.size());
    }
    
    // Step 2: Batch transcribe using transcribe_batch()
    transcribe_params tp;
    tp.max_tokens = 1024;
    tp.n_threads = 4;
    if (!batch.empty() && !batch[0].language.empty()) {
        tp.language = batch[0].language;
    }
    
    LOG_INFO("Calling transcribe_batch for {} requests", batch.size());
    auto transcribe_results = asr_->transcribe_batch(audio_samples, n_samples_list, tp);
    LOG_INFO("transcribe_batch completed, {} results returned", transcribe_results.size());
    
    // Step 3: Align each result
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& req = batch[i];
        try {
            if (i < transcribe_results.size() && transcribe_results[i].success) {
                const std::string& text = transcribe_results[i].text_content;
                
                // Align
                align_params ap;
                
                LOG_INFO("Request {}: Aligning text: {}", req.request_id, text.substr(0, 50));
                auto align_result = aligner_->align(
                    float_samples_list[i].data(),
                    n_samples_list[i],
                    text,
                    req.language,
                    ap
                );
                
                if (!align_result.success) {
                    LOG_ERROR("Request {} align failed: {}", req.request_id, align_result.error_msg);
                    req.result_promise.set_value("{\"success\":false,\"error\":\"" + escape_json_string(align_result.error_msg) + "\"}");
                } else {
                    // Build JSON response
                    std::ostringstream json;
                    json << "{\n";
                    json << "  \"success\": true,\n";
                    json << "  \"n_utterances\": " << align_result.utterances.size() << ",\n";
                    json << "  \"utterances\": [\n";
                    
                    for (size_t u = 0; u < align_result.utterances.size(); ++u) {
                        const auto& utt = align_result.utterances[u];
                        json << "    {\n";
                        json << "      \"start\": " << std::fixed << std::setprecision(3) << utt.start << ",\n";
                        json << "      \"end\": " << std::fixed << std::setprecision(3) << utt.end << ",\n";
                        json << "      \"text\": \"" << escape_json_string(utt.text) << "\",\n";
                        json << "      \"n_words\": " << utt.words.size() << ",\n";
                        json << "      \"words\": [\n";
                        
                        for (size_t w = 0; w < utt.words.size(); ++w) {
                            const auto& word = utt.words[w];
                            json << "        {";
                            json << "\"word\": \"" << escape_json_string(word.word) << "\", ";
                            json << "\"start\": " << std::fixed << std::setprecision(3) << word.start << ", ";
                            json << "\"end\": " << std::fixed << std::setprecision(3) << word.end << ", ";
                            json << "\"conf_word\": " << std::fixed << std::setprecision(4) << word.conf_word << ", ";
                            json << "\"conf_start_time\": " << std::fixed << std::setprecision(4) << word.conf_start_time << ", ";
                            json << "\"conf_end_time\": " << std::fixed << std::setprecision(4) << word.conf_end_time;
                            json << "}";
                            if (w + 1 < utt.words.size()) json << ",";
                            json << "\n";
                        }
                        
                        json << "      ]\n";
                        json << "    }";
                        if (u + 1 < align_result.utterances.size()) json << ",";
                        json << "\n";
                    }
                    
                    json << "  ],\n";
                    json << "  \"transcribe_ms\": " << transcribe_results[i].t_total_ms << ",\n";
                    json << "  \"align_ms\": " << align_result.t_total_ms << ",\n";
                    json << "  \"total_ms\": " << (transcribe_results[i].t_total_ms + align_result.t_total_ms) << "\n";
                    json << "}\n";
                    
                    req.result_promise.set_value(json.str());
                    LOG_INFO("Request {} transcribe-align completed", req.request_id);
                }
            } else {
                std::string error = i < transcribe_results.size() ? transcribe_results[i].error_msg : "No result";
                LOG_ERROR("Request {} transcribe failed: {}", req.request_id, error);
                req.result_promise.set_value("{\"success\":false,\"error\":\"" + escape_json_string(error) + "\"}");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception processing request {}: {}", req.request_id, e.what());
            req.result_promise.set_value("{\"success\":false,\"error\":\"Internal error\"}");
        }
    }
}

void BatchScheduler::process_batch_transcribe_only(std::vector<ASRRequest>& batch) {
    if (!asr_) {
        LOG_ERROR("ASR not set");
        for (auto& req : batch) {
            req.result_promise.set_value("{\"success\":false,\"error\":\"ASR not initialized\"}");
        }
        return;
    }
    
    // Prepare float samples
    std::vector<std::vector<float>> float_samples_list(batch.size());
    std::vector<const float*> audio_samples(batch.size());
    std::vector<int> n_samples_list(batch.size());
    
    for (size_t i = 0; i < batch.size(); ++i) {
        float_samples_list[i].resize(batch[i].pcm_data.size());
        for (size_t j = 0; j < batch[i].pcm_data.size(); ++j) {
            float_samples_list[i][j] = batch[i].pcm_data[j] / 32768.0f;
        }
        audio_samples[i] = float_samples_list[i].data();
        n_samples_list[i] = static_cast<int>(batch[i].pcm_data.size());
    }
    
    // Batch transcribe
    transcribe_params tp;
    tp.max_tokens = 1024;
    tp.n_threads = 4;
    if (!batch.empty() && !batch[0].language.empty()) {
        tp.language = batch[0].language;
    }
    
    LOG_INFO("Calling transcribe_batch for {} requests", batch.size());
    auto results = asr_->transcribe_batch(audio_samples, n_samples_list, tp);
    LOG_INFO("transcribe_batch completed, {} results returned", results.size());
    
    // Return results
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& req = batch[i];
        try {
            if (i < results.size() && results[i].success) {
                std::ostringstream json;
                json << "{\n";
                json << "  \"success\": true,\n";
                json << "  \"text\": \"" << escape_json_string(results[i].text) << "\",\n";
                json << "  \"text_content\": \"" << escape_json_string(results[i].text_content) << "\",\n";
                json << "  \"n_tokens\": " << results[i].tokens.size() << ",\n";
                json << "  \"total_ms\": " << results[i].t_total_ms << "\n";
                json << "}\n";
                
                req.result_promise.set_value(json.str());
                LOG_INFO("Request {} transcribe completed", req.request_id);
            } else {
                std::string error = i < results.size() ? results[i].error_msg : "No result";
                LOG_ERROR("Request {} failed: {}", req.request_id, error);
                req.result_promise.set_value("{\"success\":false,\"error\":\"" + escape_json_string(error) + "\"}");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception processing request {}: {}", req.request_id, e.what());
            req.result_promise.set_value("{\"success\":false,\"error\":\"Internal error\"}");
        }
    }
}

}
