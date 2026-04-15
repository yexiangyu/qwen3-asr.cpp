cmake_minimum_required(VERSION 3.14)
project(qwen3-asr-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# spdlog (header-only via FetchContent)
include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    
    if(APPLE)
        # Apple Silicon optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
        
        # Enable LTO for Release builds
        include(CheckIPOSupported)
        check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
        if(LTO_SUPPORTED)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        endif()
    else()
        # Generic Linux/Unix optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    endif()
endif()

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Timing instrumentation option
option(QWEN3_ASR_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_ASR_TIMING)
    add_compile_definitions(QWEN3_ASR_TIMING)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Logger library
add_library(logger STATIC
    src/logger.cpp
)
target_include_directories(logger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(logger PUBLIC spdlog::spdlog_header_only)

# Audio utils library
add_library(audio_utils STATIC
    src/audio_utils.cpp
)
target_include_directories(audio_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Mel spectrogram library
add_library(mel_spectrogram STATIC
    src/mel_spectrogram.cpp
)
target_include_directories(mel_spectrogram PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(APPLE)
    target_link_libraries(mel_spectrogram PRIVATE
        Threads::Threads
        "-framework Accelerate"
    )
else()
    target_link_libraries(mel_spectrogram PRIVATE
        Threads::Threads
    )
endif()

# Audio encoder library (GGML-based)
add_library(audio_encoder STATIC
    src/gguf_loader.cpp
    src/audio_encoder.cpp
)
target_include_directories(audio_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Text decoder library (GGML-based)
add_library(text_decoder STATIC
    src/text_decoder.cpp
)
target_include_directories(text_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(text_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Audio injection library (pure C++, no GGML dependency)
add_library(audio_injection STATIC
    src/audio_injection.cpp
)
target_include_directories(audio_injection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Qwen3 ASR library (full pipeline)
add_library(qwen3_asr STATIC
    src/qwen3_asr.cpp
)
target_include_directories(qwen3_asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_asr PUBLIC
    mel_spectrogram
    audio_encoder
    text_decoder
    audio_injection
    audio_utils
    logger
    Threads::Threads
)

# ForcedAligner library
add_library(forced_aligner STATIC
    src/forced_aligner.cpp
)
target_include_directories(forced_aligner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(forced_aligner PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(forced_aligner PUBLIC
    mel_spectrogram
    audio_utils
    ggml
    ggml-base
    ggml-cpu
    logger
    Threads::Threads
)

# CLI executable
add_executable(qwen3-asr-cli
    src/main.cpp
)
target_link_libraries(qwen3-asr-cli PRIVATE
    qwen3_asr
    forced_aligner
    logger
)

# Test executable for mel spectrogram
add_executable(test_mel
    tests/test_mel.cpp
)
target_link_libraries(test_mel PRIVATE
    mel_spectrogram
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_encoder
    Threads::Threads
)

# Test executable for text decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    text_decoder
    Threads::Threads
)

# Test executable for audio injection
add_executable(test_injection
    tests/test_injection.cpp
)
target_link_libraries(test_injection PRIVATE
    audio_injection
)

# Simple decoder test
add_executable(test_decoder_simple
    tests/test_decoder_simple.cpp
)
target_link_libraries(test_decoder_simple PRIVATE
    text_decoder
    Threads::Threads
)

# Embedding test
add_executable(test_embedding
    tests/test_embedding.cpp
)
target_include_directories(test_embedding PRIVATE
    ${GGML_DIR}/include
)
target_link_directories(test_embedding PRIVATE
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(test_embedding PRIVATE
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Debug decoder test
add_executable(test_decoder_debug
    tests/test_decoder_debug.cpp
)
target_link_libraries(test_decoder_debug PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder with reference audio
add_executable(test_decoder_with_audio
    tests/test_decoder_with_audio.cpp
)
target_link_libraries(test_decoder_with_audio PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder without audio (long sequence)
add_executable(test_decoder_no_audio
    tests/test_decoder_no_audio.cpp
)
target_link_libraries(test_decoder_no_audio PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder last position
add_executable(test_decoder_last_pos
    tests/test_decoder_last_pos.cpp
)
target_link_libraries(test_decoder_last_pos PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder lengths
add_executable(test_decoder_lengths
    tests/test_decoder_lengths.cpp
)
target_link_libraries(test_decoder_lengths PRIVATE
    text_decoder
    Threads::Threads
)

# Test attention comparison
add_executable(test_attention_compare
    tests/test_attention_compare.cpp
)
target_link_libraries(test_attention_compare PRIVATE
    text_decoder
    Threads::Threads
)

# Test Q/K comparison
add_executable(test_qk_compare
    tests/test_qk_compare.cpp
)
target_link_libraries(test_qk_compare PRIVATE
    text_decoder
    Threads::Threads
)

# Test conv only (no chunking)
add_executable(test_conv_only
    tests/test_conv_only.cpp
)
target_link_libraries(test_conv_only PRIVATE
    audio_encoder
    Threads::Threads
)

# Test encoder without chunking
add_executable(test_encoder_no_chunk
    tests/test_encoder_no_chunk.cpp
)
target_link_libraries(test_encoder_no_chunk PRIVATE
    audio_encoder
    Threads::Threads
)

# ============================================================================
# Shared Library (C API)
# ============================================================================

set(QWEN3ASR_ALL_SOURCES
    src/qwen3asr_c_api.cpp
    src/qwen3_asr.cpp
    src/forced_aligner.cpp
    src/mel_spectrogram.cpp
    src/audio_encoder.cpp
    src/text_decoder.cpp
    src/audio_injection.cpp
    src/audio_utils.cpp
    src/gguf_loader.cpp
    src/logger.cpp
)

add_library(qwen3asr SHARED ${QWEN3ASR_ALL_SOURCES})

target_include_directories(qwen3asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
    ${spdlog_SOURCE_DIR}/include
)

target_link_directories(qwen3asr PRIVATE
    ${GGML_BUILD_DIR}/src
)

target_link_libraries(qwen3asr PRIVATE
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Link all available backends - GGML scheduler will auto-select at runtime
if(APPLE)
    # macOS: Metal + CPU
    target_link_directories(qwen3asr PRIVATE
        ${GGML_BUILD_DIR}/src
        ${GGML_BUILD_DIR}/src/ggml-metal
    )
    target_link_libraries(qwen3asr PRIVATE
        ggml-metal
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate"
    )
elseif(WIN32)
    # Windows: CUDA (if available) + CPU
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_directories(qwen3asr PRIVATE ${GGML_BUILD_DIR}/src)
        target_link_libraries(qwen3asr PRIVATE ggml-cuda CUDA::cudart)
        message(STATUS "GGML CUDA backend: enabled")
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux: CUDA/Vulkan (if available) + CPU
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_directories(qwen3asr PRIVATE ${GGML_BUILD_DIR}/src)
        target_link_libraries(qwen3asr PRIVATE ggml-cuda CUDA::cudart)
        message(STATUS "GGML CUDA backend: enabled")
    endif()
endif()

# Common platform settings
if(WIN32)
    target_compile_definitions(qwen3asr PRIVATE QWEN3ASR_EXPORTS)
    set_target_properties(qwen3asr PROPERTIES
        OUTPUT_NAME "qwen3asr"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
elseif(APPLE)
    set_target_properties(qwen3asr PROPERTIES
        OUTPUT_NAME "qwen3asr"
        MACOSX_RPATH ON
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(qwen3asr PROPERTIES
        OUTPUT_NAME "qwen3asr"
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Install targets
install(TARGETS qwen3asr mel_spectrogram audio_encoder text_decoder audio_injection qwen3_asr forced_aligner qwen3-asr-cli
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
install(FILES src/qwen3asr_c_api.h src/mel_spectrogram.h src/audio_encoder.h src/gguf_loader.h src/text_decoder.h src/audio_injection.h src/qwen3_asr.h src/forced_aligner.h
    DESTINATION include
)

# C API test
add_executable(test_c_api tests/test_c_api.cpp)
target_link_libraries(test_c_api PRIVATE qwen3asr)

if(APPLE)
    set_target_properties(test_c_api PROPERTIES
        BUILD_RPATH "${GGML_BUILD_DIR}/src;${GGML_BUILD_DIR}/src/ggml-metal"
        INSTALL_RPATH "@executable_path/../lib"
    )
elseif(UNIX AND NOT APPLE)
    set_target_properties(test_c_api PROPERTIES
        BUILD_RPATH "${GGML_BUILD_DIR}/src"
    )
endif()

# CTest support
enable_testing()
add_test(NAME mel_spectrogram_test
    COMMAND test_mel
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME audio_encoder_test
    COMMAND test_encoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME text_decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME audio_injection_test
    COMMAND test_injection
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test tokenization
add_executable(test_tokenization
    tests/test_tokenization.cpp
)
target_link_libraries(test_tokenization PRIVATE
    Threads::Threads
)

# Test audio utils
add_executable(test_audio_utils
    tests/test_audio_utils.cpp
)
target_link_libraries(test_audio_utils PRIVATE
    audio_utils
    Threads::Threads
)

# Test conv1 output
add_executable(test_conv1
    tests/test_conv1.cpp
)
target_include_directories(test_conv1 PRIVATE
    ${GGML_DIR}/include
)
target_link_libraries(test_conv1 PRIVATE
    audio_encoder
    Threads::Threads
)

# ============================================================================
# HTTP Server (optional)
# ============================================================================

option(QWEN3_ASR_HTTP_SERVER "Build HTTP server" ON)

if(QWEN3_ASR_HTTP_SERVER)
    # ASIO (required by Crow)
    FetchContent_Declare(
        asio
        GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
        GIT_TAG asio-1-28-1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(asio)
    
    # Add ASIO include directory to target
    add_library(asio_header INTERFACE)
    target_include_directories(asio_header INTERFACE "${asio_SOURCE_DIR}/asio/include")
    
    # Crow HTTP framework
    FetchContent_Declare(
        crow
        GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
        GIT_TAG v1.3.2
        GIT_SHALLOW TRUE
    )
    set(CROW_ENABLE_TESTS OFF CACHE BOOL "Disable Crow tests")
    FetchContent_MakeAvailable(crow)
    
    # HTTP server executable
    add_executable(qwen3-asr-server
        src/http_server/server.cpp
        src/http_server/align_handler.cpp
    )
    target_include_directories(qwen3-asr-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${crow_SOURCE_DIR}/include
        "${asio_SOURCE_DIR}/asio/include"
    )
    target_link_libraries(qwen3-asr-server PRIVATE
        qwen3asr
        logger
        asio_header
        Crow::Crow
        Threads::Threads
    )
    
    # Install target
    install(TARGETS qwen3-asr-server DESTINATION bin)
endif()
