#include "asr_server.h"
#include "logger.h"

#include <crow.h>
#include <cstring>
#include <vector>
#include <cstdlib>

namespace qwen3_asr {

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --asr-model <path> --aligner-model <path> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --asr-model <path>       ASR model path (required)\n");
    fprintf(stderr, "  --aligner-model <path>   Aligner model path (required)\n");
    fprintf(stderr, "  --port <num>             HTTP port (default: 8082)\n");
    fprintf(stderr, "  --threads <num>          Processing threads (default: 4)\n");
    fprintf(stderr, "  --max-tokens <num>       Max output tokens (default: 1024)\n");
    fprintf(stderr, "  --korean-dict <path>     Korean dictionary path\n");
    fprintf(stderr, "  --default-language <lang>  Default language\n");
    fprintf(stderr, "  --asr-device <name>      ASR GPU device (e.g. CUDA0, Metal)\n");
    fprintf(stderr, "  --aligner-device <name>  Aligner GPU device (e.g. CUDA1)\n");
    fprintf(stderr, "  --batch-size <num>       Max batch size for scheduler (default: 2)\n");
    fprintf(stderr, "  --batch-timeout <ms>     Batch timeout in milliseconds (default: 100)\n");
    fprintf(stderr, "  --help                   Show this message\n");
}

static bool parse_int(const char* str, int& out) {
    if (!str || str[0] == '\0') return false;
    char* endptr = nullptr;
    long val = std::strtol(str, &endptr, 10);
    if (endptr == str || *endptr != '\0') return false;
    out = static_cast<int>(val);
    return true;
}

int run_combined_server(int argc, char** argv) {
    CombinedServerConfig config;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--asr-model") == 0 && i + 1 < argc) {
            config.asr_model_path = argv[++i];
        } else if (strcmp(argv[i], "--aligner-model") == 0 && i + 1 < argc) {
            config.aligner_model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], config.port)) {
                fprintf(stderr, "Error: Invalid port number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], config.n_threads)) {
                fprintf(stderr, "Error: Invalid threads number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], config.max_tokens)) {
                fprintf(stderr, "Error: Invalid max-tokens number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--korean-dict") == 0 && i + 1 < argc) {
            config.korean_dict_path = argv[++i];
        } else if (strcmp(argv[i], "--default-language") == 0 && i + 1 < argc) {
            config.default_language = argv[++i];
        } else if (strcmp(argv[i], "--asr-device") == 0 && i + 1 < argc) {
            config.asr_device = argv[++i];
        } else if (strcmp(argv[i], "--aligner-device") == 0 && i + 1 < argc) {
            config.aligner_device = argv[++i];
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], config.max_batch_size)) {
                fprintf(stderr, "Error: Invalid batch-size number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--batch-timeout") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], config.batch_timeout_ms)) {
                fprintf(stderr, "Error: Invalid batch-timeout number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (config.asr_model_path.empty()) {
        fprintf(stderr, "Error: --asr-model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (config.aligner_model_path.empty()) {
        fprintf(stderr, "Error: --aligner-model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    init_logger();
    
    CombinedASRServer server(config);
    
    if (!server.init()) {
        LOG_ERROR("Failed to initialize combined ASR server");
        return 1;
    }
    
    crow::SimpleApp app;
    
    CROW_ROUTE(app, "/health")
    ([&server]() {
        crow::json::wvalue json;
        json["status"] = "ok";
        json["asr_model_loaded"] = server.is_models_loaded();
        json["aligner_model_loaded"] = server.is_models_loaded();
        return json;
    });
    
    CROW_ROUTE(app, "/transcribe").methods("POST"_method)
    ([&server](const crow::request& req) {
        try {
            crow::multipart::message_view msg(req);
            
            auto audio_part = msg.get_part_by_name("audio");
            if (audio_part.body.empty()) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Missing required field: audio";
                return crow::response(400, err);
            }
            
            size_t n_samples = audio_part.body.size() / 2;
            
            const size_t MAX_PCM_SAMPLES = 5000000;
            if (n_samples > MAX_PCM_SAMPLES) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Audio too large (max 5M samples)";
                return crow::response(400, err);
            }
            
            if (n_samples == 0) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Empty audio data";
                return crow::response(400, err);
            }
            
            auto lang_part = msg.get_part_by_name("language");
            std::string language = lang_part.body.empty() ? 
                                    server.get_config().default_language : 
                                    std::string(lang_part.body);
            
            auto context_part = msg.get_part_by_name("context");
            std::string context(context_part.body);
            
            auto max_tokens_part = msg.get_part_by_name("max_tokens");
            int max_tokens = 0;
            if (!max_tokens_part.body.empty()) {
                max_tokens = std::stoi(std::string(max_tokens_part.body));
            }
            
            std::vector<int16_t> pcm_data(n_samples);
            std::memcpy(pcm_data.data(), audio_part.body.data(), audio_part.body.size());
            
            LOG_INFO("Received transcribe request: pcm_len={} samples, lang={}, context_len={}, max_tokens={}",
                     n_samples, language.empty() ? "(auto)" : language, 
                     context.size(), max_tokens);
            
            std::string json_result = server.handle_transcribe(pcm_data, language, context, max_tokens);
            
            return crow::response(200, json_result);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Request error: {}", e.what());
            crow::json::wvalue err;
            err["success"] = false;
            err["error"] = std::string("Internal error: ") + e.what();
            return crow::response(500, err);
        }
    });
    
    CROW_ROUTE(app, "/align").methods("POST"_method)
    ([&server](const crow::request& req) {
        try {
            crow::multipart::message_view msg(req);
            
            auto text_part = msg.get_part_by_name("text");
            if (text_part.body.empty()) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Missing required field: text";
                return crow::response(400, err);
            }
            
            auto audio_part = msg.get_part_by_name("audio");
            if (audio_part.body.empty()) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Missing required field: audio";
                return crow::response(400, err);
            }
            
            std::string text(text_part.body);
            
            auto lang_part = msg.get_part_by_name("language");
            std::string language = lang_part.body.empty() ? 
                                    server.get_config().default_language : 
                                    std::string(lang_part.body);
            
            size_t n_samples = audio_part.body.size() / 2;
            
            const size_t MAX_PCM_SAMPLES = 5000000;
            if (n_samples > MAX_PCM_SAMPLES) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Audio too large (max 5M samples)";
                return crow::response(400, err);
            }
            
            if (n_samples == 0) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Empty audio data";
                return crow::response(400, err);
            }
            
            std::vector<int16_t> pcm_data(n_samples);
            std::memcpy(pcm_data.data(), audio_part.body.data(), audio_part.body.size());
            
            LOG_INFO("Received align request: text_len={}, pcm_len={} samples, lang={}",
                     text.size(), n_samples, language.empty() ? "(auto)" : language);
            
            std::string json_result = server.handle_align(text, pcm_data, language);
            
            return crow::response(200, json_result);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Request error: {}", e.what());
            crow::json::wvalue err;
            err["success"] = false;
            err["error"] = std::string("Internal error: ") + e.what();
            return crow::response(500, err);
        }
    });
    
    CROW_ROUTE(app, "/transcribe-align").methods("POST"_method)
    ([&server](const crow::request& req) {
        try {
            crow::multipart::message_view msg(req);
            
            auto audio_part = msg.get_part_by_name("audio");
            if (audio_part.body.empty()) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Missing required field: audio";
                return crow::response(400, err);
            }
            
            size_t n_samples = audio_part.body.size() / 2;
            
            const size_t MAX_PCM_SAMPLES = 5000000;
            if (n_samples > MAX_PCM_SAMPLES) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Audio too large (max 5M samples)";
                return crow::response(400, err);
            }
            
            if (n_samples == 0) {
                crow::json::wvalue err;
                err["success"] = false;
                err["error"] = "Empty audio data";
                return crow::response(400, err);
            }
            
            auto lang_part = msg.get_part_by_name("language");
            std::string language = lang_part.body.empty() ? 
                                    server.get_config().default_language : 
                                    std::string(lang_part.body);
            
            auto context_part = msg.get_part_by_name("context");
            std::string context(context_part.body);
            
            auto max_tokens_part = msg.get_part_by_name("max_tokens");
            int max_tokens = 0;
            if (!max_tokens_part.body.empty()) {
                max_tokens = std::stoi(std::string(max_tokens_part.body));
            }
            
            std::vector<int16_t> pcm_data(n_samples);
            std::memcpy(pcm_data.data(), audio_part.body.data(), audio_part.body.size());
            
            LOG_INFO("Received transcribe-align request: pcm_len={} samples, lang={}, context_len={}, max_tokens={}",
                     n_samples, language.empty() ? "(auto)" : language, 
                     context.size(), max_tokens);
            
            std::string json_result = server.handle_transcribe_align(pcm_data, language, context, max_tokens);
            
            return crow::response(200, json_result);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Request error: {}", e.what());
            crow::json::wvalue err;
            err["success"] = false;
            err["error"] = std::string("Internal error: ") + e.what();
            return crow::response(500, err);
        }
    });
    
    LOG_INFO("Starting Combined ASR HTTP server on port {}", config.port);
    LOG_INFO("Endpoints:");
    LOG_INFO("  GET  /health          - Health check");
    LOG_INFO("  POST /transcribe      - Transcription only");
    LOG_INFO("  POST /align           - Alignment only (requires text)");
    LOG_INFO("  POST /transcribe-align - Combined transcription + alignment");
    
    app.port(config.port).multithreaded();
    app.run();
    
    return 0;
}

}

int main(int argc, char** argv) {
    return qwen3_asr::run_combined_server(argc, argv);
}