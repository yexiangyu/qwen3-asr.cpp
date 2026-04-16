#pragma once

#include <deque>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstdint>
#include <string>

namespace qwen3_asr {

// Forward declarations
class Qwen3ASR;
class ForcedAligner;

enum class RequestType {
    TRANSCRIBE,
    TRANSCRIBE_ALIGN
};

struct ASRRequest {
    std::vector<int16_t> pcm_data;
    std::string language;
    std::string context;
    int max_tokens;
    std::promise<std::string> result_promise;
    int request_id;
    RequestType type;
    
    ASRRequest() : max_tokens(1024), request_id(-1), type(RequestType::TRANSCRIBE) {}
};

struct BatchConfig {
    int max_batch_size = 2;
    int batch_timeout_ms = 100;
    
    BatchConfig() = default;
    BatchConfig(int batch_size, int timeout) 
        : max_batch_size(batch_size), batch_timeout_ms(timeout) {}
};

class BatchScheduler {
public:
    BatchScheduler();
    ~BatchScheduler();
    
    void set_asr(Qwen3ASR* asr);
    void set_aligner(ForcedAligner* aligner);
    
    std::future<std::string> submit_request(
        const std::vector<int16_t>& pcm,
        const std::string& language = "",
        const std::string& context = "",
        int max_tokens = 1024,
        RequestType type = RequestType::TRANSCRIBE
    );
    
    void start();
    void stop();
    
    void set_batch_size(int size);
    void set_timeout_ms(int ms);
    
    int get_pending_count() const;
    bool is_running() const;
    
private:
    void batch_worker();
    void process_batch(std::vector<ASRRequest>& batch);
    void process_batch_transcribe_align(std::vector<ASRRequest>& batch);
    void process_batch_transcribe_only(std::vector<ASRRequest>& batch);
    
    std::string build_transcribe_json(const std::string& text, const std::string& text_content,
                                       int n_tokens, int64_t mel_ms, int64_t encode_ms, 
                                       int64_t decode_ms, int64_t total_ms);
    std::string build_transcribe_align_json(const std::string& text, const std::string& text_content,
                                             const std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>>& words_with_times,
                                             int64_t total_ms);
    
    BatchConfig config_;
    Qwen3ASR* asr_;
    ForcedAligner* aligner_;
    
    std::deque<ASRRequest> pending_queue_;
    std::mutex queue_mutex_;
    std::condition_variable batch_cv_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    std::atomic<int> next_request_id_;
};

}
