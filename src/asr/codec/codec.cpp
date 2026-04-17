#include "asr/codec/codec.hpp"
#include "asr/common/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace qwen3_asr {
namespace asr { namespace codec {

bool ffmpeg_decode(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), 
        "ffmpeg -i \"%s\" -ar 16000 -ac 1 -f s16le - 2>/dev/null",
        path);
    
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        if (error) error->message = "Failed to execute ffmpeg";
        return false;
    }
    
    samples.clear();
    sample_rate = 16000;
    
    const int chunk_size = 4096;
    int16_t buffer[chunk_size];
    
    while (true) {
        size_t n = fread(buffer, sizeof(int16_t), chunk_size, pipe);
        if (n == 0) break;
        
        size_t offset = samples.size();
        samples.resize(offset + n);
        
        for (size_t i = 0; i < n; i++) {
            samples[offset + i] = buffer[i] / 32768.0f;
        }
    }
    
    int status = pclose(pipe);
    if (status != 0) {
        if (error) error->message = "ffmpeg exited with error (code " + std::to_string(status) + ")";
        return false;
    }
    
    if (samples.empty()) {
        if (error) error->message = "No audio data decoded";
        return false;
    }
    
    normalize_audio(samples);
    
    return true;
}

bool decode_file(const char* path, std::vector<float>& samples, int& sample_rate, ErrorInfo* error) {
    if (load_wav(path, samples, sample_rate, error)) {
        normalize_audio(samples);
        return true;
    }
    
    if (ffmpeg_decode(path, samples, sample_rate, error)) {
        return true;
    }
    
    if (error && error->message.empty()) {
        error->message = "Failed to decode audio file: " + std::string(path);
    }
    
    return false;
}

} // namespace codec
} // namespace asr
} // namespace qwen3_asr