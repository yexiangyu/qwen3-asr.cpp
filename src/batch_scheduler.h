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
    
    void set_asr(void* asr_handle);
    void set_aligner(void* aligner_handle);
    
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
    
    BatchConfig config_;
    void* asr_handle_;
    void* aligner_handle_;
    
    std::deque<ASRRequest> pending_queue_;
    std::mutex queue_mutex_;
    std::condition_variable batch_cv_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    std::atomic<int> next_request_id_;
};

}