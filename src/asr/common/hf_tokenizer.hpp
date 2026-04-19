#pragma once

#include <string>
#include <vector>
#include <map>

namespace asr {

struct HfTokenizerData {
    std::vector<std::string> vocab;
    std::map<std::string, int> token_to_id;
    std::map<std::string, int> bpe_ranks;
};

inline std::string hf_json_extract_string(const std::string& json, size_t pos) {
    if (pos >= json.size() || json[pos] != '"') return "";
    size_t end = pos + 1;
    while (end < json.size()) {
        if (json[end] == '\\') { end += 2; continue; }
        if (json[end] == '"') break;
        end++;
    }
    return json.substr(pos + 1, end - pos - 1);
}

inline std::string hf_json_unescape(const std::string& s) {
    std::string result;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            if (next == 'n') { result += '\n'; i++; }
            else if (next == 't') { result += '\t'; i++; }
            else if (next == 'r') { result += '\r'; i++; }
            else if (next == '\\') { result += '\\'; i++; }
            else if (next == '"') { result += '"'; i++; }
            else if (next == 'u' && i + 5 < s.size()) {
                unsigned int code = 0;
                for (int j = 0; j < 4; ++j) {
                    char c = s[i + 2 + j];
                    if (c >= '0' && c <= '9') code = (code << 4) | (c - '0');
                    else if (c >= 'a' && c <= 'f') code = (code << 4) | (c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') code = (code << 4) | (c - 'A' + 10);
                }
                if (code < 0x80) result += (char)code;
                else if (code < 0x800) { result += (char)(0xC0 | (code >> 6)); result += (char)(0x80 | (code & 0x3F)); }
                else { result += (char)(0xE0 | (code >> 12)); result += (char)(0x80 | ((code >> 6) & 0x3F)); result += (char)(0x80 | (code & 0x3F)); }
                i += 5;
            } else { result += next; i++; }
        } else result += s[i];
    }
    return result;
}

inline int hf_json_extract_int(const std::string& json, size_t pos) {
    size_t end = pos;
    bool negative = false;
    if (end < json.size() && json[end] == '-') { negative = true; end++; }
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++;
    int val = 0;
    for (size_t i = (negative ? pos + 1 : pos); i < end; ++i) val = val * 10 + (json[i] - '0');
    return negative ? -val : val;
}

inline size_t hf_json_skip_value(const std::string& json, size_t pos) {
    if (pos >= json.size()) return pos;
    char c = json[pos];
    if (c == '"') {
        size_t end = pos + 1;
        while (end < json.size()) {
            if (json[end] == '\\') { end += 2; continue; }
            if (json[end] == '"') return end + 1;
            end++;
        }
        return end;
    }
    if (c == '{') {
        int depth = 1;
        size_t end = pos + 1;
        while (end < json.size() && depth > 0) {
            if (json[end] == '{') depth++;
            else if (json[end] == '}') depth--;
            else if (json[end] == '"') { end++; while (end < json.size()) { if (json[end] == '\\') { end += 2; continue; } if (json[end] == '"') break; end++; } }
            end++;
        }
        return end;
    }
    if (c == '[') {
        int depth = 1;
        size_t end = pos + 1;
        while (end < json.size() && depth > 0) {
            if (json[end] == '[') depth++;
            else if (json[end] == ']') depth--;
            else if (json[end] == '"') { end++; while (end < json.size()) { if (json[end] == '\\') { end += 2; continue; } if (json[end] == '"') break; end++; } }
            end++;
        }
        return end;
    }
    if (c == 't') return pos + 4;
    if (c == 'f') return pos + 5;
    if (c == 'n') return pos + 4;
    if (c == '-' || (c >= '0' && c <= '9')) {
        size_t end = pos;
        if (json[end] == '-') end++;
        while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++;
        if (end < json.size() && json[end] == '.') { end++; while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++; }
        if (end < json.size() && (json[end] == 'e' || json[end] == 'E')) { end++; if (end < json.size() && (json[end] == '+' || json[end] == '-')) end++; while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++; }
        return end;
    }
    return pos + 1;
}

inline bool load_tokenizer_from_hf_json(const char* json_str, int vocab_size, HfTokenizerData& data) {
    std::string json(json_str);

    size_t model_pos = json.find("\"model\"");
    if (model_pos == std::string::npos) return false;

    size_t vocab_pos = json.find("\"vocab\"", model_pos);
    if (vocab_pos == std::string::npos) return false;

    size_t vocab_obj_start = json.find('{', vocab_pos);
    if (vocab_obj_start == std::string::npos) return false;

    data.vocab.resize(vocab_size);
    std::fill(data.vocab.begin(), data.vocab.end(), std::string(""));

    size_t pos = vocab_obj_start + 1;
    int vocab_count = 0;
    while (pos < json.size() && json[pos] != '}') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == '}') break;

        std::string key = hf_json_unescape(hf_json_extract_string(json, pos));
        pos += key.size() + 2;
        while (pos < json.size() && json[pos] != ':') pos++;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;

        int id = hf_json_extract_int(json, pos);
        pos = hf_json_skip_value(json, pos);

        if (id >= 0 && id < vocab_size) {
            data.vocab[id] = key;
            data.token_to_id[key] = id;
            vocab_count++;
        }
    }

    size_t merges_pos = json.find("\"merges\"", vocab_pos);
    if (merges_pos == std::string::npos) return false;

    size_t merges_arr_start = json.find('[', merges_pos);
    if (merges_arr_start == std::string::npos) return false;

    pos = merges_arr_start + 1;
    int merge_count = 0;
    while (pos < json.size() && json[pos] != ']') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == ']') break;

        size_t arr_start = json.find('[', pos);
        if (arr_start == std::string::npos) break;

        size_t arr_pos = arr_start + 1;
        while (arr_pos < json.size() && (json[arr_pos] == ' ' || json[arr_pos] == '\n')) arr_pos++;

        std::string first = hf_json_unescape(hf_json_extract_string(json, arr_pos));
        arr_pos += first.size() + 2;
        while (arr_pos < json.size() && (json[arr_pos] == ' ' || json[arr_pos] == ',' || json[arr_pos] == '\n')) arr_pos++;

        std::string second = hf_json_unescape(hf_json_extract_string(json, arr_pos));
        arr_pos += second.size() + 2;

        std::string merge = first + second;
        data.bpe_ranks[merge] = merge_count;
        merge_count++;

        size_t arr_end = json.find(']', arr_pos);
        if (arr_end == std::string::npos) break;
        pos = arr_end + 1;
    }

    size_t added_pos = json.find("\"added_tokens\"");
    if (added_pos != std::string::npos) {
        size_t added_arr_start = json.find('[', added_pos);
        if (added_arr_start != std::string::npos) {
            pos = added_arr_start + 1;
            while (pos < json.size() && json[pos] != ']') {
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == ',')) pos++;
                if (pos >= json.size() || json[pos] == ']') break;
                if (json[pos] != '{') { pos = hf_json_skip_value(json, pos); continue; }

                size_t obj_start = pos;
                std::string content;
                int id = -1;

                size_t inner = pos + 1;
                while (inner < json.size() && json[inner] != '}') {
                    while (inner < json.size() && (json[inner] == ' ' || json[inner] == '\n' || json[inner] == '\t' || json[inner] == '\r' || json[inner] == ',')) inner++;
                    if (inner >= json.size() || json[inner] == '}') break;

                    std::string field_key = hf_json_unescape(hf_json_extract_string(json, inner));
                    inner += field_key.size() + 2;
                    while (inner < json.size() && json[inner] != ':') inner++;
                    inner++;
                    while (inner < json.size() && (json[inner] == ' ' || json[inner] == '\n')) inner++;

                    if (field_key == "id") id = hf_json_extract_int(json, inner);
                    else if (field_key == "content") content = hf_json_unescape(hf_json_extract_string(json, inner));

                    inner = hf_json_skip_value(json, inner);
                }

                if (id >= 0 && id < vocab_size && !content.empty()) {
                    data.vocab[id] = content;
                    data.token_to_id[content] = id;
                }

                pos = json.find('}', obj_start) + 1;
            }
        }
    }

    fprintf(stderr, "Loaded tokenizer from HF JSON: %d vocab, %d merges\n", vocab_count, merge_count);
    return vocab_count > 0 && merge_count > 0;
}

} // namespace asr