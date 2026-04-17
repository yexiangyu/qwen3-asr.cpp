cmake_minimum_required(VERSION 3.14)
project(qwen3-asr-ggml VERSION 0.1.0 LANGUAGES CXX)

# Auto-initialize GGML submodule if not present
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
if(NOT EXISTS "${GGML_DIR}/CMakeLists.txt")
    message(STATUS "GGML submodule not found, initializing...")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to initialize GGML submodule. Please run 'git submodule update --init --recursive' manually.")
        endif()
    else()
        message(FATAL_ERROR "Git not found. Please initialize GGML submodule manually.")
    endif()
endif()

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

# CUDA detection for Linux: auto-enable if found at /usr/local/cuda, unless explicitly disabled
if(UNIX AND NOT APPLE)
    option(GGML_CUDA "Enable GGML CUDA backend" ON)
    
    if(GGML_CUDA)
        # Set CUDA toolkit hint paths before find_package (only if not already set)
        if(NOT DEFINED ENV{CUDAToolkit_ROOT} AND NOT DEFINED CUDAToolkit_ROOT)
            set(ENV{CUDAToolkit_ROOT} "/usr/local/cuda")
        endif()
        find_package(CUDAToolkit QUIET)
        
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit found at ${CUDAToolkit_ROOT_DIR}")
            message(STATUS "GGML CUDA backend: enabled")
            set(GGML_CUDA_FOUND TRUE)
            # Set CUDA compiler path for GGML subdirectory
            if(NOT CMAKE_CUDA_COMPILER)
                set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE PATH "CUDA compiler" FORCE)
            endif()
        else()
            message(STATUS "CUDA not found, building with CPU only")
            set(GGML_CUDA_FOUND FALSE)
        endif()
    else()
        message(STATUS "CUDA explicitly disabled")
        set(GGML_CUDA_FOUND FALSE)
    endif()
endif()

# Build GGML as part of this project (GGML_CUDA must be set before add_subdirectory)
set(GGML_BUILD_DIR "${GGML_DIR}/build")
add_subdirectory(${GGML_DIR} ${GGML_BUILD_DIR} EXCLUDE_FROM_ALL)

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
    target_compile_definitions(mel_spectrogram PRIVATE HAVE_BLAS)
else()
    # Try to find BLAS for Linux: prefer MKL, then OpenBLAS, then generic BLAS
    set(BLAS_FOUND FALSE)
    set(BLAS_LIBRARIES "")

    # 1. Try Intel MKL at /opt/intel/oneapi/mkl/latest
    if(NOT BLAS_FOUND)
        set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
        if(EXISTS "${MKL_ROOT}")
            message(STATUS "Intel MKL root found at ${MKL_ROOT}")
            find_path(MKL_INCLUDE_DIR mkl.h PATHS "${MKL_ROOT}/include" NO_DEFAULT_PATH)
            # Use mkl_rt single dynamic library for simpler linking
            find_library(MKL_RT_LIB NAMES mkl_rt PATHS "${MKL_ROOT}/lib" "${MKL_ROOT}/lib/intel64" NO_DEFAULT_PATH)

            if(MKL_RT_LIB)
                set(BLAS_FOUND TRUE)
                set(HAVE_MKL TRUE)
                set(BLAS_LIBRARIES ${MKL_RT_LIB})
                message(STATUS "Intel MKL found: ${BLAS_LIBRARIES}")
                target_include_directories(mel_spectrogram PRIVATE "${MKL_ROOT}/include")
                target_compile_definitions(mel_spectrogram PRIVATE HAVE_MKL)
            else()
                message(STATUS "MKL root found but mkl_rt not found, trying other BLAS providers")
            endif()
        endif()
    endif()

    # 2. Try OpenBLAS
    if(NOT BLAS_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENBLAS QUIET openblas)
        endif()
        if(OPENBLAS_FOUND)
            set(BLAS_FOUND TRUE)
            set(BLAS_LIBRARIES ${OPENBLAS_LINK_LIBRARIES})
            message(STATUS "OpenBLAS found via pkg-config: ${BLAS_LIBRARIES}")
            if(OPENBLAS_INCLUDE_DIRS)
                target_include_directories(mel_spectrogram PRIVATE ${OPENBLAS_INCLUDE_DIRS})
            endif()
        else()
            # Try finding OpenBLAS directly
            find_library(OPENBLAS_LIB NAMES openblas PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/OpenBLAS/lib)
            find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS /usr/include /usr/include/openblas /usr/local/include /opt/OpenBLAS/include)
            if(OPENBLAS_LIB)
                set(BLAS_FOUND TRUE)
                set(BLAS_LIBRARIES ${OPENBLAS_LIB})
                message(STATUS "OpenBLAS found: ${BLAS_LIBRARIES}")
                if(OPENBLAS_INCLUDE_DIR)
                    target_include_directories(mel_spectrogram PRIVATE ${OPENBLAS_INCLUDE_DIR})
                endif()
            endif()
        endif()
    endif()

    # 3. Fallback to generic BLAS
    if(NOT BLAS_FOUND)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            message(STATUS "Generic BLAS found: ${BLAS_LIBRARIES}")
        endif()
    endif()

    if(BLAS_FOUND)
        target_link_libraries(mel_spectrogram PRIVATE
            Threads::Threads
            ${BLAS_LIBRARIES}
        )
        target_compile_definitions(mel_spectrogram PRIVATE HAVE_BLAS)
    else()
        message(STATUS "BLAS not found, using naive implementation")
        target_link_libraries(mel_spectrogram PRIVATE
            Threads::Threads
        )
    endif()
endif()

# ============================================================================
# Audio Codec Library (New modular component)
# ============================================================================

add_library(audio_codec STATIC
    src/modules/audio_codec/audio_codec.cpp
    src/modules/audio_codec/wav_loader.cpp
)
target_include_directories(audio_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/audio_codec
)
target_link_libraries(audio_codec PUBLIC
    Threads::Threads
)

# Mel spectrogram module (new)
add_library(mel_module STATIC
    src/modules/mel/mel.cpp
)
target_include_directories(mel_module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/mel
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/audio_codec
)
target_link_libraries(mel_module PUBLIC
    audio_codec
    Threads::Threads
)

# Encoder module (new)
add_library(encoder_module STATIC
    src/modules/encoder/encoder.cpp
)
target_include_directories(encoder_module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/encoder
    ${GGML_DIR}/include
)
target_link_directories(encoder_module PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(encoder_module PUBLIC
    mel_module
    audio_codec
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_CUDA_FOUND)
    target_link_directories(encoder_module PUBLIC ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(encoder_module PUBLIC ggml-cuda CUDA::cudart)
    target_compile_definitions(encoder_module PUBLIC GGML_USE_CUDA)
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
if(GGML_CUDA_FOUND)
    target_link_directories(audio_encoder PUBLIC ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(audio_encoder PUBLIC ggml-cuda CUDA::cudart)
    target_compile_definitions(audio_encoder PUBLIC GGML_USE_CUDA)
endif()

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
if(GGML_CUDA_FOUND)
    target_link_directories(text_decoder PUBLIC ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(text_decoder PUBLIC ggml-cuda CUDA::cudart)
    target_compile_definitions(text_decoder PUBLIC GGML_USE_CUDA)
endif()

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
if(GGML_CUDA_FOUND)
    target_link_directories(forced_aligner PUBLIC ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(forced_aligner PUBLIC ggml-cuda CUDA::cudart)
    target_compile_definitions(forced_aligner PUBLIC GGML_USE_CUDA)
endif()

# CLI executable
add_executable(qwen3-asr-cli
    src/main.cpp
)
target_link_libraries(qwen3-asr-cli PRIVATE
    qwen3_asr
    forced_aligner
    logger
)

# Test executables (controlled by QWEN3_ASR_BUILD_TESTS option)
option(QWEN3_ASR_BUILD_TESTS "Build test executables" OFF)

if(QWEN3_ASR_BUILD_TESTS)
    # Test audio codec (new module)
    add_executable(test_audio_codec
        src/modules/audio_codec/test_audio_codec.cpp
    )
    target_link_libraries(test_audio_codec PRIVATE
        audio_codec
        Threads::Threads
    )

    # Test mel module
    add_executable(test_mel_module
        src/modules/mel/test_mel.cpp
    )
    target_link_libraries(test_mel_module PRIVATE
        mel_module
        audio_codec
        Threads::Threads
    )

    # Test encoder module
    add_executable(test_encoder_module
        src/modules/encoder/test_encoder.cpp
    )
    target_link_libraries(test_encoder_module PRIVATE
        encoder_module
        mel_module
        audio_codec
        ggml
        ggml-base
        ggml-cpu
        Threads::Threads
    )
    if(GGML_CUDA_FOUND)
        target_link_directories(test_encoder_module PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
        target_link_libraries(test_encoder_module PRIVATE ggml-cuda CUDA::cudart)
    endif()

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
    if(GGML_CUDA_FOUND)
        target_link_directories(test_embedding PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
        target_link_libraries(test_embedding PRIVATE ggml-cuda CUDA::cudart)
    endif()

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
    add_executable(test_qkcompare
        tests/test_qkcompare.cpp
    )
    target_link_libraries(test_qkcompare PRIVATE
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
endif()

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
elseif(GGML_CUDA_FOUND)
    # Linux/Windows: CUDA + CPU
    target_link_directories(qwen3asr PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(qwen3asr PRIVATE ggml-cuda CUDA::cudart)
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

# Additional test executables (only built when QWEN3_ASR_BUILD_TESTS is ON)
if(QWEN3_ASR_BUILD_TESTS)
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
        if(GGML_CUDA_FOUND)
            set(CUDA_RPATH "${CUDAToolkit_LIBRARY_DIR}")
            if(NOT CUDA_RPATH)
                set(CUDA_RPATH "/usr/local/cuda/lib64")
            endif()
            set_target_properties(test_c_api PROPERTIES
                BUILD_RPATH "${GGML_BUILD_DIR}/src;${CUDA_RPATH}"
            )
        endif()
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
endif()

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
    
    # Align HTTP server executable
    add_executable(qwen3-align-server
        src/http_server/align_server.cpp
        src/http_server/align_handler.cpp
    )
    target_include_directories(qwen3-align-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${crow_SOURCE_DIR}/include
        "${asio_SOURCE_DIR}/asio/include"
    )
    target_link_libraries(qwen3-align-server PRIVATE
        qwen3asr
        logger
        asio_header
        Crow::Crow
        Threads::Threads
    )
    
    # Transcribe HTTP server executable
    add_executable(qwen3-transcribe-server
        src/http_server/transcribe_server.cpp
        src/http_server/transcribe_handler.cpp
    )
    target_include_directories(qwen3-transcribe-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${crow_SOURCE_DIR}/include
        "${asio_SOURCE_DIR}/asio/include"
    )
    target_link_libraries(qwen3-transcribe-server PRIVATE
        qwen3asr
        logger
        asio_header
        Crow::Crow
        Threads::Threads
    )
    
    # Combined ASR server executable (transcription + alignment)
    add_executable(qwen3-asr-server
        src/http_server/asr_server.cpp
        src/http_server/asr_handler.cpp
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
    
    # Install targets
    install(TARGETS qwen3-align-server qwen3-transcribe-server qwen3-asr-server DESTINATION bin)
endif()
