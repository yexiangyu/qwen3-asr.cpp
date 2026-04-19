cmake_minimum_required(VERSION 3.14)
project(qwen3-asr-ggml VERSION 0.1.0 LANGUAGES CXX)

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
            message(FATAL_ERROR "Failed to initialize GGML submodule.")
        endif()
    else()
        message(FATAL_ERROR "Git not found. Please initialize GGML submodule manually.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    if(APPLE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
        include(CheckIPOSupported)
        check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
        if(LTO_SUPPORTED)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        endif()
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(QWEN3_ASR_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_ASR_TIMING)
    add_compile_definitions(QWEN3_ASR_TIMING)
endif()

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
    option(GGML_CUDA "Enable GGML CUDA backend" ON)
    if(GGML_CUDA)
        if(NOT DEFINED ENV{CUDAToolkit_ROOT} AND NOT DEFINED CUDAToolkit_ROOT)
            set(ENV{CUDAToolkit_ROOT} "/usr/local/cuda")
        endif()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit found at ${CUDAToolkit_ROOT_DIR}")
            message(STATUS "GGML CUDA backend: enabled")
            set(GGML_CUDA_FOUND TRUE)
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

set(GGML_BUILD_DIR "${GGML_DIR}/build")
add_subdirectory(${GGML_DIR} ${GGML_BUILD_DIR} EXCLUDE_FROM_ALL)

add_library(logger STATIC
    src/logger.cpp
)
target_include_directories(logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(logger PUBLIC spdlog::spdlog_header_only)

add_library(asr_common INTERFACE)
target_include_directories(asr_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(asr_codec STATIC
    src/asr/codec/codec.cpp
    src/asr/codec/wav_loader.cpp
)
target_include_directories(asr_codec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(asr_codec PUBLIC asr_common Threads::Threads)

add_library(asr_mel STATIC
    src/asr/mel/mel.cpp
)
target_include_directories(asr_mel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(asr_mel PUBLIC asr_common asr_codec Threads::Threads)

function(add_asr_module target_name src_file)
    add_library(${target_name} STATIC ${src_file})
    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GGML_DIR}/include
    )
    target_link_directories(${target_name} PUBLIC ${GGML_BUILD_DIR}/src)
    target_link_libraries(${target_name} PUBLIC
        asr_common
        ggml
        ggml-base
        ggml-cpu
        Threads::Threads
    )
    if(GGML_CUDA_FOUND)
        target_link_directories(${target_name} PUBLIC ${GGML_BUILD_DIR}/src/ggml-cuda)
        target_link_libraries(${target_name} PUBLIC ggml-cuda CUDA::cudart)
        target_compile_definitions(${target_name} PUBLIC GGML_USE_CUDA)
    endif()
endfunction()

add_asr_module(asr_transcribe_encoder src/asr/transcribe/encoder.cpp)
target_link_libraries(asr_transcribe_encoder PUBLIC asr_mel)

add_asr_module(asr_transcribe_decoder src/asr/transcribe/decoder.cpp)

add_asr_module(asr_aligner_encoder src/asr/aligner/encoder.cpp)
target_link_libraries(asr_aligner_encoder PUBLIC asr_mel)

add_asr_module(asr_aligner_decoder src/asr/aligner/decoder.cpp)

add_executable(asr-cli src/tools/asr_cli.cpp)
target_include_directories(asr-cli PRIVATE ${GGML_DIR}/include)
target_link_directories(asr-cli PRIVATE ${GGML_BUILD_DIR}/src)
target_link_libraries(asr-cli PRIVATE
    asr_codec
    asr_mel
    asr_transcribe_encoder
    asr_transcribe_decoder
    asr_aligner_encoder
    asr_aligner_decoder
    logger
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_CUDA_FOUND)
    target_link_directories(asr-cli PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(asr-cli PRIVATE ggml-cuda CUDA::cudart)
endif()

add_executable(ggml-quantize src/tools/quantize.cpp)
target_include_directories(ggml-quantize PRIVATE ${GGML_DIR}/include)
target_link_directories(ggml-quantize PRIVATE ${GGML_BUILD_DIR}/src)
target_link_libraries(ggml-quantize PRIVATE
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_CUDA_FOUND)
    target_link_directories(ggml-quantize PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
    target_link_libraries(ggml-quantize PRIVATE ggml-cuda CUDA::cudart)
endif()

option(QWEN3_ASR_BUILD_TESTS "Build test executables" OFF)

if(QWEN3_ASR_BUILD_TESTS)
    function(add_asr_test test_name src_file)
        add_executable(${test_name} ${src_file})
        target_include_directories(${test_name} PRIVATE ${GGML_DIR}/include)
        target_link_directories(${test_name} PRIVATE ${GGML_BUILD_DIR}/src)
        target_link_libraries(${test_name} PRIVATE
            ggml
            ggml-base
            ggml-cpu
            Threads::Threads
        )
        if(GGML_CUDA_FOUND)
            target_link_directories(${test_name} PRIVATE ${GGML_BUILD_DIR}/src/ggml-cuda)
            target_link_libraries(${test_name} PRIVATE ggml-cuda CUDA::cudart)
        endif()
    endfunction()

    add_asr_test(test_codec tests/test_codec_module.cpp)
    target_link_libraries(test_codec PRIVATE asr_codec)

    add_asr_test(test_mel tests/test_mel_module.cpp)
    target_link_libraries(test_mel PRIVATE asr_mel asr_codec)

    add_asr_test(test_transcribe_encoder tests/test_transcribe_encoder.cpp)
    target_link_libraries(test_transcribe_encoder PRIVATE asr_transcribe_encoder asr_mel asr_codec)

    add_asr_test(test_batch_encode tests/test_batch_encode.cpp)
    target_link_libraries(test_batch_encode PRIVATE asr_transcribe_encoder asr_mel asr_codec)

    add_asr_test(test_transcribe_decoder tests/test_transcribe_decoder.cpp)
    target_link_libraries(test_transcribe_decoder PRIVATE asr_transcribe_decoder)

    add_asr_test(test_batch_decode tests/test_batch_decode.cpp)
    target_link_libraries(test_batch_decode PRIVATE asr_transcribe_decoder)

    add_asr_test(test_e2e_batch_transcribe tests/test_e2e_batch_transcribe.cpp)
    target_link_libraries(test_e2e_batch_transcribe PRIVATE
        asr_transcribe_encoder asr_transcribe_decoder asr_mel asr_codec)

    add_asr_test(test_batch_performance tests/test_batch_performance.cpp)
    target_link_libraries(test_batch_performance PRIVATE
        asr_transcribe_encoder asr_transcribe_decoder asr_mel asr_codec)

    add_asr_test(test_aligner_encoder tests/test_aligner_encoder.cpp)
    target_link_libraries(test_aligner_encoder PRIVATE asr_aligner_encoder asr_mel asr_codec)

    add_asr_test(test_aligner_decoder tests/test_aligner_decoder.cpp)
    target_link_libraries(test_aligner_decoder PRIVATE asr_aligner_decoder)

    add_asr_test(test_aligner_decoder_simple tests/test_aligner_decoder_simple.cpp)
    target_link_libraries(test_aligner_decoder_simple PRIVATE asr_aligner_decoder)

    add_asr_test(test_batch_align tests/test_batch_align.cpp)
    target_link_libraries(test_batch_align PRIVATE asr_aligner_encoder asr_aligner_decoder asr_mel asr_codec)

    add_asr_test(integration_test tests/integration_test.cpp)
    target_link_libraries(integration_test PRIVATE
        asr_transcribe_encoder asr_transcribe_decoder asr_mel asr_codec)

    add_asr_test(test_conv2d tests/test_conv2d.cpp)

    add_asr_test(test_embedding tests/test_embedding.cpp)

    add_asr_test(test_mel_load tests/test_mel_load.cpp)

    add_executable(test_tokenization tests/test_tokenization.cpp)
    target_link_libraries(test_tokenization PRIVATE Threads::Threads)

    enable_testing()
    add_test(NAME codec_test COMMAND test_codec WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME mel_test COMMAND test_mel WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME transcribe_encoder_test COMMAND test_transcribe_encoder WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME transcribe_decoder_test COMMAND test_transcribe_decoder WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME integration_test COMMAND integration_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

install(TARGETS asr-cli asr_codec asr_mel asr_transcribe_encoder asr_transcribe_decoder asr_aligner_encoder asr_aligner_decoder logger
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)