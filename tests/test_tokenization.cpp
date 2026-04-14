#include <iostream>
#include <vector>
#include <string>
#include <cassert>

static uint32_t utf8_to_codepoint(const std::string & s, size_t & i) {
    if (i >= s.size()) return 0;
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0;
    if ((c & 0x80) == 0) {
        cp = c;
        i += 1;
    } else if ((c & 0xE0) == 0xC0) {
        if (i + 1 < s.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
            i += 2;
        } else { i += 1; }
    } else if ((c & 0xF0) == 0xE0) {
        if (i + 2 < s.size()) {
            cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6)
                 | (static_cast<unsigned char>(s[i + 2]) & 0x3F);
            i += 3;
        } else { i += 1; }
    } else if ((c & 0xF8) == 0xF0) {
        if (i + 3 < s.size()) {
            cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12)
                 | ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6)
                 | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
            i += 4;
        } else { i += 1; }
    } else {
        i += 1;
    }
    return cp;
}

static bool is_cjk_char(uint32_t code) {
    return (0x4E00 <= code && code <= 0x9FFF)
        || (0x3400 <= code && code <= 0x4DBF)
        || (0x20000 <= code && code <= 0x2A6DF)
        || (0x2A700 <= code && code <= 0x2B73F)
        || (0x2B740 <= code && code <= 0x2B81F)
        || (0x2B820 <= code && code <= 0x2CEAF)
        || (0xF900 <= code && code <= 0xFAFF);
}

static bool is_kept_char(uint32_t code) {
    if (code == '\'') return true;
    if (code < 0x80) {
        if (('A' <= code && code <= 'Z') || ('a' <= code && code <= 'z') ||
            ('0' <= code && code <= '9')) return true;
        if (code == '.' || code == '!' || code == '?') return true;
        return false;
    }
    if (code == 0x3002) return true;
    if (code == 0xFF01) return true;
    if (code == 0xFF1F) return true;
    if (0x4E00 <= code && code <= 0x9FFF) return true;
    if (0x3400 <= code && code <= 0x4DBF) return true;
    if (0xAC00 <= code && code <= 0xD7AF) return true;
    if (0x3040 <= code && code <= 0x30FF) return true;
    return false;
}

static std::string clean_token(const std::string & token) {
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        size_t start = i;
        uint32_t cp = utf8_to_codepoint(token, i);
        if (is_kept_char(cp)) {
            result += token.substr(start, i - start);
        }
    }
    return result;
}

static bool is_end_punctuation(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F;
}

static std::vector<std::string> split_segment_with_cjk(const std::string & seg) {
    std::vector<std::string> tokens;
    std::string buf;
    size_t i = 0;
    while (i < seg.size()) {
        size_t start = i;
        uint32_t cp = utf8_to_codepoint(seg, i);
        if (is_cjk_char(cp)) {
            if (!buf.empty()) {
                std::string cleaned = clean_token(buf);
                if (!cleaned.empty()) tokens.push_back(cleaned);
                buf.clear();
            }
            std::string cjk_char = seg.substr(start, i - start);
            std::string cleaned = clean_token(cjk_char);
            if (!cleaned.empty()) tokens.push_back(cleaned);
        } else if (is_end_punctuation(cp)) {
            if (!buf.empty()) {
                std::string cleaned = clean_token(buf);
                if (!cleaned.empty()) tokens.push_back(cleaned);
                buf.clear();
            }
            std::string punct = seg.substr(start, i - start);
            tokens.push_back(punct);
        } else {
            buf += seg.substr(start, i - start);
        }
    }
    if (!buf.empty()) {
        std::string cleaned = clean_token(buf);
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
    return tokens;
}

static std::vector<std::string> tokenize_space_lang(const std::string & text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                                    text[i] == '\n' || text[i] == '\r')) ++i;
        if (i >= text.size()) break;
        size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t' &&
               text[i] != '\n' && text[i] != '\r') ++i;
        std::string seg = text.substr(start, i - start);
        std::string cleaned_seg = clean_token(seg);
        if (!cleaned_seg.empty()) {
            auto sub_tokens = split_segment_with_cjk(cleaned_seg);
            for (const auto & t : sub_tokens) {
                tokens.push_back(t);
            }
        }
    }
    return tokens;
}

void print_tokens(const std::string & text, const std::vector<std::string> & tokens) {
    std::cout << "Input: \"" << text << "\"\n";
    std::cout << "Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << tokens[i] << "\"";
    }
    std::cout << "]\n\n";
}

int main() {
    std::cout << "=== Tokenization Tests ===\n\n";

    // Test 1: Pure Chinese
    auto t1 = tokenize_space_lang("你好世界");
    print_tokens("你好世界", t1);
    assert(t1.size() == 4);
    assert(t1[0] == "你");
    assert(t1[1] == "好");
    assert(t1[2] == "世");
    assert(t1[3] == "界");

    // Test 2: Mixed Chinese and English (no spaces)
    auto t2 = tokenize_space_lang("Hello你好World");
    print_tokens("Hello你好World", t2);
    assert(t2.size() == 3);
    assert(t2[0] == "Hello");
    assert(t2[1] == "你");
    assert(t2[2] == "好");

    // Test 3: Mixed with spaces
    auto t3 = tokenize_space_lang("你好 Hello 世界");
    print_tokens("你好 Hello 世界", t3);
    assert(t3.size() == 4);
    assert(t3[0] == "你");
    assert(t3[1] == "好");
    assert(t3[2] == "Hello");
    assert(t3[3] == "世");

    // Test 4: Pure English
    auto t4 = tokenize_space_lang("Hello World");
    print_tokens("Hello World", t4);
    assert(t4.size() == 2);
    assert(t4[0] == "Hello");
    assert(t4[1] == "World");

    // Test 5: With punctuation (should be kept for utterance splitting)
    auto t5 = tokenize_space_lang("你好，世界！");
    print_tokens("你好，世界！", t5);
    // "，" is filtered (comma), "！" is kept (exclamation)
    assert(t5.size() == 5);
    assert(t5[0] == "你");
    assert(t5[1] == "好");
    assert(t5[2] == "世");
    assert(t5[3] == "界");
    assert(t5[4] == "！");

    // Test 5b: Chinese period (should be separate token)
    auto t5b = tokenize_space_lang("你好。世界");
    print_tokens("你好。世界", t5b);
    assert(t5b.size() == 6);
    assert(t5b[0] == "你");
    assert(t5b[1] == "好");
    assert(t5b[2] == "。");
    assert(t5b[3] == "世");
    assert(t5b[4] == "界");

    // Test 6: Numbers
    auto t6 = tokenize_space_lang("123你好456");
    print_tokens("123你好456", t6);
    assert(t6.size() == 3);
    assert(t6[0] == "123");
    assert(t6[1] == "你");
    assert(t6[2] == "好");

    // Test 7: Complex mixed
    auto t7 = tokenize_space_lang("我是AI工程师");
    print_tokens("我是AI工程师", t7);
    assert(t7.size() == 5);
    assert(t7[0] == "我");
    assert(t7[1] == "是");
    assert(t7[2] == "AI");
    assert(t7[3] == "工");
    assert(t7[4] == "程");

    // Test 8: English with end punctuation
    auto t8 = tokenize_space_lang("Hello World.");
    print_tokens("Hello World.", t8);
    assert(t8.size() == 3);
    assert(t8[0] == "Hello");
    assert(t8[1] == "World");
    assert(t8[2] == ".");
    
    // Test 9: English with question mark
    auto t9 = tokenize_space_lang("How are you?");
    print_tokens("How are you?", t9);
    assert(t9.size() == 4);
    assert(t9[0] == "How");
    assert(t9[1] == "are");
    assert(t9[2] == "you");
    assert(t9[3] == "?");
    
    // Test 10: Mixed English and Chinese with punctuation
    auto t10 = tokenize_space_lang("你好Hello世界.");
    print_tokens("你好Hello世界.", t10);
    assert(t10.size() == 6);
    assert(t10[0] == "你");
    assert(t10[1] == "好");
    assert(t10[2] == "Hello");
    assert(t10[3] == "世");
    assert(t10[4] == "界");
    assert(t10[5] == ".");

    std::cout << "All tests passed!\n";
    return 0;
}