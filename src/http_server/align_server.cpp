#include "align_server.h"
#include "logger.h"

#include <crow.h>
#include <cstring>
#include <vector>
#include <cstdlib>

namespace qwen3_asr {

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --model <path> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model <path>         Aligner model path (required)\n");
    fprintf(stderr, "  --port <num>           HTTP port (default: 8080)\n");
    fprintf(stderr, "  --threads <num>        Processing threads (default: 4)\n");
    fprintf(stderr, "  --korean-dict <path>   Korean dictionary path\n");
    fprintf(stderr, "  --default-language <lang>  Default language\n");
    fprintf(stderr, "  --help                 Show this message\n");
}

static bool parse_int(const char* str, int& out) {
    if (!str || str[0] == '\0') return false;
    char* endptr = nullptr;
    long val = std::strtol(str, &endptr, 10);
    if (endptr == str || *endptr != '\0') return false;
    out = static_cast<int>(val);
    return true;
}

int run_server(int argc, char** argv) {
    ServerConfig config;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
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
        } else if (strcmp(argv[i], "--korean-dict") == 0 && i + 1 < argc) {
            config.korean_dict_path = argv[++i];
        } else if (strcmp(argv[i], "--default-language") == 0 && i + 1 < argc) {
            config.default_language = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (config.aligner_model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    init_logger();
    
    AlignServer server(config);
    
    if (!server.init()) {
        LOG_ERROR("Failed to initialize server");
        return 1;
    }
    
    crow::SimpleApp app;
    
    CROW_ROUTE(app, "/health")
    ([&server]() {
        crow::json::wvalue json;
        json["status"] = "ok";
        json["model_loaded"] = server.is_model_loaded();
        return json;
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
    
    LOG_INFO("Starting HTTP server on port {}", config.port);
    LOG_INFO("Endpoints:");
    LOG_INFO("  GET  /health - Health check");
    LOG_INFO("  POST /align  - Alignment (multipart/form-data)");
    
    app.port(config.port).multithreaded();
    app.run();
    
    return 0;
}

}

int main(int argc, char** argv) {
    return qwen3_asr::run_server(argc, argv);
}