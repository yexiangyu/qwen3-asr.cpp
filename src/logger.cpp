#include "logger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/pattern_formatter.h>

namespace qwen3_asr {

void init_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console_sink->set_pattern("%^%l%$: %v");
    
    auto logger = std::make_shared<spdlog::logger>("qwen3_asr", console_sink);
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
}

void set_log_level(int level) {
    auto logger = spdlog::get("qwen3_asr");
    if (logger) {
        logger->set_level(static_cast<spdlog::level::level_enum>(level));
    }
}

}