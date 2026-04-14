#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "timing.h"
#include "logger.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>

static bool parse_int(const char* str, int32_t& out) {
    if (!str || str[0] == '\0') return false;
    
    char* endptr = nullptr;
    long val = std::strtol(str, &endptr, 10);
    
    if (endptr == str || *endptr != '\0') return false;
    if (val < INT32_MIN || val > INT32_MAX) return false;
    
    out = static_cast<int32_t>(val);
    return true;
}

struct cli_params {
    std::string model_path = "models/qwen3-asr-0.6b-f16.gguf";
    std::string aligner_model_path = "";
    std::string audio_path = "";
    std::string output_path = "";
    std::string language = "";
    std::string align_text = "";
    std::string context = "";
    int32_t max_tokens = 1024;
    int32_t n_threads = 4;
    bool print_progress = false;
    bool print_timing = true;
    bool print_tokens = false;
    bool align_mode = false;
    bool transcribe_align_mode = false;
    bool profile = false;
    bool json_output = false;
    bool debug_input = false;
    bool debug_output = false;
    bool arabic_numbers = false;
};

static void print_usage(const char * prog) {
    LOG_INFO("Usage: {} [options]", prog);
    
    LOG_INFO("Options:");
    LOG_INFO("  -m, --model <path>     Path to GGUF model (default: models/qwen3-asr-0.6b-f16.gguf)");
    LOG_INFO("  -f, --audio <path>     Path to audio file (WAV, 16kHz mono) [required]");
    LOG_INFO("  -o, --output <path>    Output file path (default: stdout)");
    LOG_INFO("  -l, --language <code>  Language code (optional, e.g. 'korean' for Korean word splitting)");
    LOG_INFO("  -t, --threads <n>      Number of threads (default: 4)");
    LOG_INFO("  --max-tokens <n>       Maximum tokens to generate (default: 1024)");
    LOG_INFO("  --context <text>       Context/prompt text for ASR (domain hints, key terms)");
    LOG_INFO("  --progress             Print progress during transcription");
    LOG_INFO("  --no-timing            Don't print timing information");
    LOG_INFO("  --tokens               Print token IDs");
    LOG_INFO("  --profile              Print detailed timing profile (requires QWEN3_ASR_TIMING build)");
    LOG_INFO("  --json                 Output JSON format to stdout (default: log format)");
    LOG_INFO("  --debug-input          Print decoded input tokens");
    LOG_INFO("  --debug-output         Print decoded output tokens");
    LOG_INFO("  --arabic-numbers       Convert Chinese numbers to Arabic numerals");
    
    LOG_INFO("Forced Alignment:");
    LOG_INFO("  --align                Enable forced alignment mode");
    LOG_INFO("  --text <text>          Reference transcript for alignment");
    
    LOG_INFO("Transcribe + Align:");
    LOG_INFO("  -a, --transcribe-align Run ASR then forced alignment");
    LOG_INFO("  --aligner-model <path> Path to forced aligner GGUF model (required with --transcribe-align)");
    
    LOG_INFO("  -h, --help             Show this help message");
    
    LOG_INFO("Examples:");
    LOG_INFO("  Transcription:");
    LOG_INFO("    {} -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav", prog);
    
    LOG_INFO("  Forced Alignment:");
    LOG_INFO("    {} -m models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --align --text \"Hello world\"", prog);
    
    LOG_INFO("  Transcribe + Align:");
    LOG_INFO("    {} -m models/qwen3-asr-0.6b-f16.gguf --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --transcribe-align", prog);
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        
        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--audio") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.audio_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--language") == 0 || strcmp(arg, "--lang") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.language = argv[++i];
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            if (!parse_int(argv[++i], params.n_threads)) {
                LOG_ERROR("Invalid integer value for {}", arg);
                return false;
            }
        } else if (strcmp(arg, "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            if (!parse_int(argv[++i], params.max_tokens)) {
                LOG_ERROR("Invalid integer value for {}", arg);
                return false;
            }
        } else if (strcmp(arg, "--context") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.context = argv[++i];
        } else if (strcmp(arg, "--progress") == 0) {
            params.print_progress = true;
        } else if (strcmp(arg, "--no-timing") == 0) {
            params.print_timing = false;
        } else if (strcmp(arg, "--tokens") == 0) {
            params.print_tokens = true;
        } else if (strcmp(arg, "--profile") == 0) {
            params.profile = true;
        } else if (strcmp(arg, "--json") == 0) {
            params.json_output = true;
        } else if (strcmp(arg, "--debug-input") == 0) {
            params.debug_input = true;
        } else if (strcmp(arg, "--debug-output") == 0) {
            params.debug_output = true;
        } else if (strcmp(arg, "--arabic-numbers") == 0) {
            params.arabic_numbers = true;
        } else if (strcmp(arg, "--align") == 0) {
            params.align_mode = true;
        } else if (strcmp(arg, "-a") == 0 || strcmp(arg, "--transcribe-align") == 0) {
            params.transcribe_align_mode = true;
        } else if (strcmp(arg, "--aligner-model") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.aligner_model_path = argv[++i];
        } else if (strcmp(arg, "--text") == 0) {
            if (i + 1 >= argc) {
                LOG_ERROR("{} requires an argument", arg);
                return false;
            }
            params.align_text = argv[++i];
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            LOG_ERROR("Unknown argument: {}", arg);
            return false;
        }
    }
    
    if (params.audio_path.empty()) {
        LOG_ERROR("Audio file path is required (-f/--audio)");
        return false;
    }
    
    if (params.align_mode && params.align_text.empty()) {
        LOG_ERROR("Reference text is required for alignment mode (--text)");
        return false;
    }

    if (params.align_mode && params.transcribe_align_mode) {
        LOG_ERROR("--align and --transcribe-align cannot be used together");
        return false;
    }

    if (params.transcribe_align_mode && params.aligner_model_path.empty()) {
        LOG_ERROR("--aligner-model is required for --transcribe-align");
        return false;
    }
    
    return true;
}

static int64_t parse_chinese_number(const std::string & s) {
    static const std::unordered_map<std::string, int64_t> digits = {
        {"零", 0}, {"一", 1}, {"二", 2}, {"两", 2}, {"三", 3}, {"四", 4},
        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}
    };
    static const std::unordered_map<std::string, int64_t> units = {
        {"十", 10}, {"百", 100}, {"千", 1000}
    };
    static const std::unordered_map<std::string, int64_t> big_units = {
        {"万", 10000LL}, {"亿", 100000000LL}
    };
    
    if (s.empty()) return -1;
    
    int64_t total = 0;
    int64_t section = 0;
    int64_t current = 0;
    
    for (size_t i = 0; i < s.size(); ) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(s[i]);
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) break;
        
        std::string ch = s.substr(i, len);
        i += len;
        
        if (digits.count(ch)) {
            current = digits.at(ch);
        } else if (units.count(ch)) {
            int64_t u = units.at(ch);
            if (current == 0) current = 1;
            section += current * u;
            current = 0;
        } else if (big_units.count(ch)) {
            int64_t u = big_units.at(ch);
            if (current == 0 && section == 0) {
                section = 1;
            }
            section += current;
            total += section * u;
            section = 0;
            current = 0;
        } else {
            return -1;
        }
    }
    
    section += current;
    total += section;
    
    return total;
}

static bool is_chinese_digit_char(const std::string & ch) {
    static const std::unordered_set<std::string> digit_chars = {
        "零", "一", "二", "两", "三", "四", "五", "六", "七", "八", "九",
        "十", "百", "千", "万", "亿"
    };
    return digit_chars.count(ch) > 0;
}

static std::string convert_chinese_numbers_to_arabic(const std::string & text) {
    static const std::unordered_set<std::string> skip_words = {
        "一起", "一下", "一个", "一次", "第一", "个别", "一些", "一样", "一方面",
        "试一试", "看一看", "想一想", "走一走", "说一说", "听一听",
        "万一", "千万", "百万", "十万", "百般", "千方百计", "百分之",
        "亿万", "亿万富翁", "十全十美", "九九归一", "独一无二",
        "三心二意", "四面八方", "五颜六色", "七上八下", "十万火急"
    };
    
    std::string result;
    size_t i = 0;
    
    while (i < text.size()) {
        bool skipped = false;
        for (const auto & w : skip_words) {
            if (text.substr(i, w.size()) == w) {
                result += w;
                i += w.size();
                skipped = true;
                break;
            }
        }
        if (skipped) continue;
        
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) break;
        
        std::string ch = text.substr(i, len);
        
        if (!is_chinese_digit_char(ch)) {
            result += ch;
            i += len;
            continue;
        }
        
        std::string num_str;
        size_t j = i;
        while (j < text.size()) {
            size_t j_len = 1;
            unsigned char jc = static_cast<unsigned char>(text[j]);
            if ((jc & 0xE0) == 0xC0) j_len = 2;
            else if ((jc & 0xF0) == 0xE0) j_len = 3;
            else if ((jc & 0xF8) == 0xF0) j_len = 4;
            if (j + j_len > text.size()) break;
            
            std::string jch = text.substr(j, j_len);
            if (is_chinese_digit_char(jch)) {
                num_str += jch;
                j += j_len;
            } else {
                break;
            }
        }
        
        int64_t num = parse_chinese_number(num_str);
        if (num > 0) {
            result += std::to_string(num);
            i = j;
        } else {
            result += ch;
            i += len;
        }
    }
    
    return result;
}

static std::string detect_language(const std::string & asr_text) {
    const std::string prefix = "language ";
    if (asr_text.size() < prefix.size() || asr_text.compare(0, prefix.size(), prefix) != 0) {
        return "";
    }

    size_t pos = prefix.size();
    if (pos >= asr_text.size()) {
        return "";
    }

    unsigned char first = static_cast<unsigned char>(asr_text[pos]);
    if (!std::isupper(first)) {
        return "";
    }

    ++pos;
    while (pos < asr_text.size()) {
        unsigned char c = static_cast<unsigned char>(asr_text[pos]);
        if (!std::islower(c)) {
            break;
        }
        ++pos;
    }

    std::string lang = asr_text.substr(prefix.size(), pos - prefix.size());
    std::transform(lang.begin(), lang.end(), lang.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lang;
}

static std::string escape_json_string(const std::string & s) {
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

static void print_alignment_simple(const qwen3_asr::alignment_result & result, bool arabic_numbers = false) {
    for (const auto & utt : result.utterances) {
        std::string text = utt.text;
        if (arabic_numbers) {
            text = convert_chinese_numbers_to_arabic(text);
        }
        LOG_INFO("{:.3f}->{:.3f}: {}", utt.start, utt.end, text);
    }
}

static std::string transcribe_to_json(const qwen3_asr::transcribe_result & result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"text\": \"" << escape_json_string(result.text) << "\",\n";
    json << "  \"text_prefix\": \"" << escape_json_string(result.text_prefix) << "\",\n";
    json << "  \"text_content\": \"" << escape_json_string(result.text_content) << "\",\n";
    json << "  \"tokens\": [\n";
    for (size_t i = 0; i < result.tokens.size(); ++i) {
        json << "    {\n";
        json << "      \"id\": " << result.tokens[i] << ",\n";
        json << "      \"string\": \"" << escape_json_string(result.token_strings[i]) << "\",\n";
        json << "      \"confidence\": " << std::fixed << std::setprecision(4) << result.token_confidences[i] << "\n";
        json << "    }";
        if (i + 1 < result.tokens.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return json.str();
}

static std::string alignment_to_json_simple(const qwen3_asr::alignment_result & result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"utterances\": [\n";
    
    for (size_t ui = 0; ui < result.utterances.size(); ++ui) {
        const auto & utt = result.utterances[ui];
        json << "    {\n";
        json << "      \"start\": " << std::fixed << std::setprecision(3) << utt.start << ",\n";
        json << "      \"end\": " << std::fixed << std::setprecision(3) << utt.end << ",\n";
        json << "      \"text\": \"" << escape_json_string(utt.text) << "\",\n";
        json << "      \"words\": [\n";
        
        for (size_t wi = 0; wi < utt.words.size(); ++wi) {
            const auto & w = utt.words[wi];
            json << "        {\n";
            json << "          \"word\": \"" << escape_json_string(w.word) << "\",\n";
            json << "          \"start\": " << std::fixed << std::setprecision(3) << w.start << ",\n";
            json << "          \"end\": " << std::fixed << std::setprecision(3) << w.end << ",\n";
            json << "          \"conf_word\": " << std::fixed << std::setprecision(4) << w.conf_word << ",\n";
            json << "          \"conf_start_time\": " << std::fixed << std::setprecision(4) << w.conf_start_time << ",\n";
            json << "          \"conf_end_time\": " << std::fixed << std::setprecision(4) << w.conf_end_time << "\n";
            json << "        }";
            if (wi + 1 < utt.words.size()) json << ",";
            json << "\n";
        }
        
        json << "      ]\n";
        json << "    }";
        if (ui + 1 < result.utterances.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    
    return json.str();
}

static std::string alignment_to_json(
    const qwen3_asr::alignment_result & result,
    const std::string & asr_prefix,
    const std::vector<int32_t> & asr_tokens,
    const std::vector<float> & asr_token_confs,
    const std::vector<std::string> & asr_token_strings) {
    
    std::ostringstream json;
    json << "{\n";
    json << "  \"asr_info\": {\n";
    json << "    \"text_prefix\": \"" << escape_json_string(asr_prefix) << "\",\n";
    json << "    \"tokens\": [\n";
    
    for (size_t i = 0; i < asr_tokens.size(); ++i) {
        json << "      {\n";
        json << "        \"id\": " << asr_tokens[i] << ",\n";
        json << "        \"string\": \"" << escape_json_string(asr_token_strings[i]) << "\",\n";
        json << "        \"confidence\": " << std::fixed << std::setprecision(4) << asr_token_confs[i] << "\n";
        json << "      }";
        if (i + 1 < asr_tokens.size()) json << ",";
        json << "\n";
    }
    json << "    ]\n";
    json << "  },\n";
    
    json << "  \"utterances\": [\n";
    for (size_t ui = 0; ui < result.utterances.size(); ++ui) {
        const auto & utt = result.utterances[ui];
        json << "    {\n";
        json << "      \"start\": " << std::fixed << std::setprecision(3) << utt.start << ",\n";
        json << "      \"end\": " << std::fixed << std::setprecision(3) << utt.end << ",\n";
        json << "      \"text\": \"" << escape_json_string(utt.text) << "\",\n";
        json << "      \"words\": [\n";
        
        for (size_t wi = 0; wi < utt.words.size(); ++wi) {
            const auto & w = utt.words[wi];
            json << "        {\n";
            json << "          \"word\": \"" << escape_json_string(w.word) << "\",\n";
            json << "          \"start\": " << std::fixed << std::setprecision(3) << w.start << ",\n";
            json << "          \"end\": " << std::fixed << std::setprecision(3) << w.end << ",\n";
            json << "          \"conf_word\": " << std::fixed << std::setprecision(4) << w.conf_word << ",\n";
            json << "          \"conf_start_time\": " << std::fixed << std::setprecision(4) << w.conf_start_time << ",\n";
            json << "          \"conf_end_time\": " << std::fixed << std::setprecision(4) << w.conf_end_time << "\n";
            json << "        }";
            if (wi + 1 < utt.words.size()) json << ",";
            json << "\n";
        }
        
        json << "      ]\n";
        json << "    }";
        if (ui + 1 < result.utterances.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    
    return json.str();
}

static std::string find_korean_dict(const std::string & model_path) {
    auto dir_of = [](const std::string & path) -> std::string {
        size_t pos = path.find_last_of("/\\");
        return (pos != std::string::npos) ? path.substr(0, pos) : ".";
    };

    std::vector<std::string> candidates = {
        dir_of(model_path) + "/../assets/korean_dict_jieba.dict",
        dir_of(model_path) + "/assets/korean_dict_jieba.dict",
        "assets/korean_dict_jieba.dict",
    };

    for (const auto & p : candidates) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
}

static int run_alignment(const cli_params & params) {
    LOG_INFO("qwen3-asr-cli (Forced Alignment Mode)");
    LOG_INFO("  Model: {}", params.model_path);
    LOG_INFO("  Audio: {}", params.audio_path);
    LOG_INFO("  Text: {}", params.align_text);
    if (!params.language.empty()) {
        LOG_INFO("  Language: {}", params.language);
    }
    
    qwen3_asr::ForcedAligner aligner;
    
    if (!aligner.load_model(params.model_path)) {
        LOG_ERROR("{}", aligner.get_error());
        return 1;
    }
    
    if (params.language == "korean") {
        std::string dict_path = find_korean_dict(params.model_path);
        if (dict_path.empty()) {
            LOG_WARN("Korean dictionary not found. Falling back to whitespace splitting.");
        } else {
            if (!aligner.load_korean_dict(dict_path)) {
                LOG_WARN("Failed to load Korean dictionary from {}", dict_path);
            }
        }
    }
    
    LOG_INFO("Model loaded. Running alignment...");
    
    qwen3_asr::align_params ap;
    ap.print_progress = params.print_progress;
    ap.print_timing = params.print_timing;
    
    auto result = aligner.align(params.audio_path, params.align_text, params.language, ap);
    
    if (!result.success) {
        LOG_ERROR("{}", result.error_msg);
        return 1;
    }
    
    if (params.print_timing) {
        LOG_INFO("Timing:");
        LOG_INFO("  Mel spectrogram: {} ms", (long long)result.t_mel_ms);
        LOG_INFO("  Audio encoding:  {} ms", (long long)result.t_encode_ms);
        LOG_INFO("  Text decoding:   {} ms", (long long)result.t_decode_ms);
        LOG_INFO("  Total:           {} ms", (long long)result.t_total_ms);
        LOG_INFO("  Utterances aligned: {}", result.utterances.size());
    }
    
    if (params.json_output) {
        std::string json_output = alignment_to_json_simple(result);
        if (params.output_path.empty()) {
            printf("%s\n", json_output.c_str());
        } else {
            std::ofstream out(params.output_path);
            if (!out) {
                LOG_ERROR("Failed to open output file: {}", params.output_path);
                return 1;
            }
            out << json_output << "\n";
            LOG_INFO("Output written to: {}", params.output_path);
        }
    } else {
        print_alignment_simple(result, params.arabic_numbers);
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

static int run_transcription(const cli_params & params) {
    LOG_INFO("qwen3-asr-cli");
    LOG_INFO("  Model: {}", params.model_path);
    LOG_INFO("  Audio: {}", params.audio_path);
    LOG_INFO("  Threads: {}", params.n_threads);
    
    qwen3_asr::Qwen3ASR asr;
    
    if (!asr.load_model(params.model_path)) {
        LOG_ERROR("{}", asr.get_error());
        return 1;
    }
    
    qwen3_asr::transcribe_params tp;
    tp.max_tokens = params.max_tokens;
    tp.language = params.language;
    tp.context = params.context;
    tp.n_threads = params.n_threads;
    tp.print_progress = params.print_progress;
    tp.print_timing = params.print_timing;
    tp.debug_input = params.debug_input;
    tp.debug_output = params.debug_output;
    
    auto result = asr.transcribe(params.audio_path, tp);
    
    if (!result.success) {
        LOG_ERROR("{}", result.error_msg);
        return 1;
    }
    
    if (params.print_tokens) {
        LOG_INFO("Tokens ({}):", result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            LOG_INFO("  [{}] {}", i, result.tokens[i]);
        }
    }
    
    if (params.json_output) {
        std::string json_output = transcribe_to_json(result);
        if (params.output_path.empty()) {
            printf("%s\n", json_output.c_str());
        } else {
            std::ofstream out(params.output_path);
            if (!out) {
                LOG_ERROR("Failed to open output file: {}", params.output_path);
                return 1;
            }
            out << json_output << "\n";
            LOG_INFO("Output written to: {}", params.output_path);
        }
    } else {
        std::string output_text = result.text_content;
        if (params.arabic_numbers) {
            output_text = convert_chinese_numbers_to_arabic(output_text);
        }
        LOG_INFO("{}", output_text);
        if (!params.output_path.empty()) {
            std::ofstream out(params.output_path);
            if (!out) {
                LOG_ERROR("Failed to open output file: {}", params.output_path);
                return 1;
            }
            out << output_text << "\n";
            LOG_INFO("Output written to: {}", params.output_path);
        }
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

static int run_transcribe_and_align(const cli_params & params) {
    LOG_INFO("qwen3-asr-cli (Transcribe + Align Mode)");
    LOG_INFO("  ASR Model: {}", params.model_path);
    LOG_INFO("  Aligner Model: {}", params.aligner_model_path);
    LOG_INFO("  Audio: {}", params.audio_path);
    LOG_INFO("  Threads: {}", params.n_threads);

    LOG_INFO("--- Phase 1: Transcription ---");
    qwen3_asr::Qwen3ASR asr;
    if (!asr.load_model(params.model_path)) {
        LOG_ERROR("(ASR) {}", asr.get_error());
        return 1;
    }

    qwen3_asr::transcribe_params tp;
    tp.max_tokens = params.max_tokens;
    tp.language = params.language;
    tp.context = params.context;
    tp.n_threads = params.n_threads;
    tp.print_progress = params.print_progress;
    tp.print_timing = params.print_timing;
    tp.debug_input = params.debug_input;
    tp.debug_output = params.debug_output;

    auto asr_result = asr.transcribe(params.audio_path, tp);
    if (!asr_result.success) {
        LOG_ERROR("(ASR) {}", asr_result.error_msg);
        return 1;
    }

    std::string detected_lang = detect_language(asr_result.text);
    std::string align_lang = params.language.empty() ? detected_lang : params.language;

    LOG_INFO("  Detected language: {}", detected_lang.empty() ? "(none)" : detected_lang);
    if (!params.language.empty()) {
        LOG_INFO("  Language override: {}", params.language);
    }
    LOG_INFO("  Alignment language: {}", align_lang.empty() ? "(none)" : align_lang);
    LOG_INFO("  Transcript: {}", asr_result.text_content);

    LOG_INFO("--- Phase 2: Forced Alignment ---");
    qwen3_asr::ForcedAligner aligner;
    if (!aligner.load_model(params.aligner_model_path)) {
        LOG_ERROR("(Aligner) {}", aligner.get_error());
        return 1;
    }

    if (align_lang == "korean") {
        std::string dict_path = find_korean_dict(params.aligner_model_path);
        if (dict_path.empty()) {
            LOG_WARN("Korean dictionary not found. Falling back to whitespace splitting.");
        } else if (!aligner.load_korean_dict(dict_path)) {
            LOG_WARN("Failed to load Korean dictionary from {}", dict_path);
        }
    }

    qwen3_asr::align_params ap;
    ap.print_progress = params.print_progress;
    ap.print_timing = params.print_timing;

    auto align_result = aligner.align_with_asr_tokens(
        params.audio_path,
        asr_result.text_content,
        asr_result.tokens,
        asr_result.token_confidences,
        asr_result.token_strings,
        align_lang,
        ap
    );
    if (!align_result.success) {
        LOG_ERROR("(Aligner) {}", align_result.error_msg);
        return 1;
    }

    if (params.print_timing) {
        LOG_INFO("Combined Timing:");
        LOG_INFO("  ASR:           {} ms", (long long) asr_result.t_total_ms);
        LOG_INFO("  Alignment:     {} ms", (long long) align_result.t_total_ms);
        LOG_INFO("  Total:         {} ms", (long long) (asr_result.t_total_ms + align_result.t_total_ms));
        LOG_INFO("  Utterances aligned: {}", align_result.utterances.size());
    }

    if (params.json_output) {
        std::string json_output = alignment_to_json(
            align_result,
            asr_result.text_prefix,
            asr_result.tokens,
            asr_result.token_confidences,
            asr_result.token_strings
        );
        if (params.output_path.empty()) {
            printf("%s\n", json_output.c_str());
        } else {
            std::ofstream out(params.output_path);
            if (!out) {
                LOG_ERROR("Failed to open output file: {}", params.output_path);
                return 1;
            }
            out << json_output << "\n";
            LOG_INFO("Output written to: {}", params.output_path);
        }
    } else {
        print_alignment_simple(align_result, params.arabic_numbers);
        if (!params.output_path.empty()) {
            std::ofstream out(params.output_path);
            if (!out) {
                LOG_ERROR("Failed to open output file: {}", params.output_path);
                return 1;
            }
            for (const auto & utt : align_result.utterances) {
                std::string text = utt.text;
                if (params.arabic_numbers) {
                    text = convert_chinese_numbers_to_arabic(text);
                }
                out << utt.start << "->" << utt.end << ": " << text << "\n";
            }
            LOG_INFO("Output written to: {}", params.output_path);
        }
    }

    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }

    return 0;
}

static void ggml_log_handler(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERROR("{}", msg);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN("{}", msg);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO("{}", msg);
            break;
        default:
            break;
    }
}

int main(int argc, char ** argv) {
    qwen3_asr::init_logger();
    ggml_log_set(ggml_log_handler, nullptr);

    cli_params params;
    
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (params.transcribe_align_mode) {
        return run_transcribe_and_align(params);
    }

    if (params.align_mode) {
        return run_alignment(params);
    } else {
        return run_transcription(params);
    }
}
