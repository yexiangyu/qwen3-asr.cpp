#include "audio_utils.h"
#include <sstream>
#include <cctype>

namespace qwen3_asr {

static int find_low_energy_boundary(
    const float * wav, int total_len, int cut_pos, int expand, int win) {
    
    int left = std::max(0, cut_pos - expand);
    int right = std::min(total_len, cut_pos + expand);
    
    if (right - left <= win) {
        return cut_pos;
    }
    
    std::vector<float> seg_abs(right - left);
    for (int i = left; i < right; ++i) {
        seg_abs[i - left] = std::fabs(wav[i]);
    }
    
    std::vector<float> window_sums(right - left - win + 1);
    for (int i = 0; i < static_cast<int>(window_sums.size()); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < win; ++j) {
            sum += seg_abs[i + j];
        }
        window_sums[i] = sum;
    }
    
    float min_sum = window_sums[0];
    int min_pos = 0;
    for (int i = 1; i < static_cast<int>(window_sums.size()); ++i) {
        if (window_sums[i] < min_sum) {
            min_sum = window_sums[i];
            min_pos = i;
        }
    }
    
    int wstart = min_pos;
    int wend = min_pos + win;
    float local_min = seg_abs[wstart];
    int inner_pos = wstart;
    for (int i = wstart + 1; i < wend; ++i) {
        if (seg_abs[i] < local_min) {
            local_min = seg_abs[i];
            inner_pos = i;
        }
    }
    
    int boundary = left + inner_pos;
    
    return std::max(std::min(boundary, total_len), 1);
}

std::vector<AudioChunk> split_audio_into_chunks(
    const std::vector<float> & wav,
    float max_chunk_sec,
    float search_expand_sec,
    float min_window_ms) {
    
    std::vector<AudioChunk> chunks;
    
    int total_len = static_cast<int>(wav.size());
    float total_sec = static_cast<float>(total_len) / QWEN_SAMPLE_RATE;
    
    if (total_sec <= max_chunk_sec) {
        AudioChunk chunk;
        chunk.orig_index = 0;
        chunk.chunk_index = 0;
        chunk.wav = wav;
        chunk.sr = QWEN_SAMPLE_RATE;
        chunk.offset_sec = 0.0f;
        chunks.push_back(chunk);
        return chunks;
    }
    
    int max_len = static_cast<int>(max_chunk_sec * QWEN_SAMPLE_RATE);
    int expand = static_cast<int>(search_expand_sec * QWEN_SAMPLE_RATE);
    int win = std::max(4, static_cast<int>(min_window_ms / 1000.0f * QWEN_SAMPLE_RATE));
    
    int start = 0;
    float offset_sec = 0.0f;
    int chunk_index = 0;
    
    while (total_len - start > max_len) {
        int cut = start + max_len;
        
        int boundary = find_low_energy_boundary(wav.data(), total_len, cut, expand, win);
        boundary = std::max(boundary, start + 1);
        boundary = std::min(boundary, total_len);
        
        AudioChunk chunk;
        chunk.orig_index = 0;
        chunk.chunk_index = chunk_index;
        chunk.wav.assign(wav.begin() + start, wav.begin() + boundary);
        chunk.sr = QWEN_SAMPLE_RATE;
        chunk.offset_sec = offset_sec;
        chunks.push_back(chunk);
        
        offset_sec += static_cast<float>(boundary - start) / QWEN_SAMPLE_RATE;
        start = boundary;
        ++chunk_index;
    }
    
    AudioChunk tail_chunk;
    tail_chunk.orig_index = 0;
    tail_chunk.chunk_index = chunk_index;
    tail_chunk.wav.assign(wav.begin() + start, wav.end());
    tail_chunk.sr = QWEN_SAMPLE_RATE;
    tail_chunk.offset_sec = offset_sec;
    chunks.push_back(tail_chunk);
    
    for (AudioChunk & c : chunks) {
        pad_audio_to_min_length(c.wav, MIN_ASR_INPUT_SECONDS);
    }
    
    return chunks;
}

static std::string fix_char_repeats(const std::string & s, int thresh) {
    std::string result;
    size_t i = 0;
    size_t n = s.size();
    
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i]) {
            ++count;
        }
        
        if (count > static_cast<size_t>(thresh)) {
            result += s[i];
            i += count;
        } else {
            result += s.substr(i, count);
            i += count;
        }
    }
    
    return result;
}

static std::string fix_pattern_repeats(const std::string & s, int thresh, int max_len = 20) {
    size_t n = s.size();
    size_t min_repeat_chars = static_cast<size_t>(thresh) * 2;
    
    if (n < min_repeat_chars) {
        return s;
    }
    
    size_t i = 0;
    std::string result;
    bool found = false;
    
    while (i <= n - min_repeat_chars) {
        found = false;
        
        for (size_t k = 1; k <= static_cast<size_t>(max_len) && !found; ++k) {
            if (i + k * thresh > n) break;
            
            std::string pattern = s.substr(i, k);
            bool valid = true;
            
            for (int rep = 1; rep < thresh; ++rep) {
                size_t start_idx = i + static_cast<size_t>(rep) * k;
                if (s.substr(start_idx, k) != pattern) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                size_t end_index = i + static_cast<size_t>(thresh) * k;
                
                while (end_index + k <= n && s.substr(end_index, k) == pattern) {
                    end_index += k;
                }
                
                result += pattern;
                result += fix_pattern_repeats(s.substr(end_index), thresh, max_len);
                i = n;
                found = true;
            }
        }
        
        if (!found) {
            result += s[i];
            ++i;
        }
    }
    
    if (!found && i < n) {
        result += s.substr(i);
    }
    
    return result;
}

std::string detect_and_fix_repetitions(const std::string & text, int threshold) {
    std::string result = fix_char_repeats(text, threshold);
    result = fix_pattern_repeats(result, threshold);
    return result;
}

std::pair<std::string, std::string> parse_asr_output(
    const std::string & raw,
    const std::string & user_language) {
    
    if (raw.empty()) {
        return {"", ""};
    }
    
    std::string s = raw;
    
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) {
        --end;
    }
    s = s.substr(start, end - start);
    
    if (s.empty()) {
        return {"", ""};
    }
    
    s = detect_and_fix_repetitions(s);
    
    if (!user_language.empty()) {
        return {user_language, s};
    }
    
    const std::string asr_text_tag = "<asr_text>";
    const std::string lang_prefix = "language ";
    
    std::string lang;
    std::string text_part;
    
    if (s.find(asr_text_tag) != std::string::npos) {
        size_t tag_pos = s.find(asr_text_tag);
        std::string meta_part = s.substr(0, tag_pos);
        text_part = s.substr(tag_pos + asr_text_tag.size());
        
        std::string meta_lower;
        for (char c : meta_part) {
            meta_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        
        if (meta_lower.find("language none") != std::string::npos) {
            size_t start = 0;
            while (start < text_part.size() && (text_part[start] == ' ' || text_part[start] == '\t' || text_part[start] == '\n' || text_part[start] == '\r')) {
                ++start;
            }
            text_part = text_part.substr(start);
            if (text_part.empty()) {
                return {"", ""};
            }
            return {"", text_part};
        }
        
        std::istringstream iss(meta_part);
        std::string line;
        while (std::getline(iss, line)) {
            size_t start = 0;
            while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
                ++start;
            }
            line = line.substr(start);
            
            if (line.empty()) continue;
            
            std::string line_lower;
            for (char c : line) {
                line_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            
            if (line_lower.find(lang_prefix) == 0) {
                std::string val = line.substr(lang_prefix.size());
                start = 0;
                while (start < val.size() && (val[start] == ' ' || val[start] == '\t')) {
                    ++start;
                }
                val = val.substr(start);
                if (!val.empty()) {
                    lang = normalize_language_name(val);
                }
                break;
            }
        }
    } else {
        if (s.find(lang_prefix) == 0) {
            size_t pos = lang_prefix.size();
            size_t lang_end = pos;
            while (lang_end < s.size() && std::isalpha(static_cast<unsigned char>(s[lang_end]))) {
                ++lang_end;
            }
            lang = normalize_language_name(s.substr(pos, lang_end - pos));
            
            while (lang_end < s.size() && (s[lang_end] == ' ' || s[lang_end] == '\t')) {
                ++lang_end;
            }
            text_part = s.substr(lang_end);
        } else {
            text_part = s;
        }
    }
    
    start = 0;
    while (start < text_part.size() && (text_part[start] == ' ' || text_part[start] == '\t' || text_part[start] == '\n' || text_part[start] == '\r')) {
        ++start;
    }
    text_part = text_part.substr(start);
    
    return {lang, text_part};
}

std::string normalize_language_name(const std::string & language) {
    if (language.empty()) {
        return "";
    }
    
    std::string s;
    for (char c : language) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            s += c;
        }
    }
    
    if (s.empty()) {
        return "";
    }
    
    s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    for (size_t i = 1; i < s.size(); ++i) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    }
    
    return s;
}

}