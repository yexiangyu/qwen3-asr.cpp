#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #ifdef QWEN3ASR_EXPORTS
        #define QWEN3ASR_API __declspec(dllexport)
    #else
        #define QWEN3ASR_API __declspec(dllimport)
    #endif
#else
    #define QWEN3ASR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QWEN3ASR_API const char* qwen3asr_get_version(void);

typedef enum {
    QWEN3_DEVICE_TYPE_CPU = 0,
    QWEN3_DEVICE_TYPE_GPU = 1,
    QWEN3_DEVICE_TYPE_IGPU = 2,
    QWEN3_DEVICE_TYPE_ACCEL = 3
} qwen3_device_type;

typedef struct {
    char* name;
    char* description;
    qwen3_device_type type;
    size_t memory_free;
    size_t memory_total;
} qwen3_device_info;

QWEN3ASR_API int qwen3_get_device_count(void);
QWEN3ASR_API int qwen3_get_device_info(int index, qwen3_device_info* info);
QWEN3ASR_API void qwen3_free_device_info(qwen3_device_info* info);

typedef enum {
    QWEN3_LOG_TRACE = 0,
    QWEN3_LOG_DEBUG = 1,
    QWEN3_LOG_INFO  = 2,
    QWEN3_LOG_WARN  = 3,
    QWEN3_LOG_ERROR = 4
} qwen3_log_level;

typedef void (*qwen3_log_callback)(qwen3_log_level level, const char* message);

QWEN3ASR_API void qwen3_set_log_callback(qwen3_log_callback callback);
QWEN3ASR_API void qwen3_set_log_level(qwen3_log_level level);

QWEN3ASR_API const char* qwen3_get_last_error(void);

typedef struct qwen3asr_handle_t* qwen3asr_handle;
typedef struct qwen3aligner_handle_t* qwen3aligner_handle;

QWEN3ASR_API int qwen3asr_init(qwen3asr_handle* handle);
QWEN3ASR_API int qwen3asr_init_with_device(qwen3asr_handle* handle, int device_index);
QWEN3ASR_API int qwen3asr_init_with_device_name(qwen3asr_handle* handle, const char* device_name);
QWEN3ASR_API void qwen3asr_free(qwen3asr_handle handle);
QWEN3ASR_API const char* qwen3asr_get_device_name(qwen3asr_handle handle);

QWEN3ASR_API int qwen3asr_load_model(qwen3asr_handle handle, const char* model_path);

typedef struct {
    int32_t max_tokens;
    const char* language;
    const char* context;
    int32_t n_threads;
} qwen3asr_params;

typedef void (*qwen3_progress_callback)(int current, int total, const char* message);

QWEN3ASR_API void qwen3asr_set_progress_callback(qwen3asr_handle handle, qwen3_progress_callback callback);

typedef struct {
    char* text;
    char* text_content;
    int32_t* token_ids;
    float* token_confs;
    int32_t n_tokens;
    int64_t t_total_ms;
} qwen3asr_result;

QWEN3ASR_API void qwen3asr_free_result(qwen3asr_result* result);

QWEN3ASR_API int qwen3asr_transcribe_wav_file(
    qwen3asr_handle handle,
    const char* wav_path,
    const qwen3asr_params* params,
    qwen3asr_result* result
);

QWEN3ASR_API int qwen3asr_transcribe_pcm(
    qwen3asr_handle handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const qwen3asr_params* params,
    qwen3asr_result* result
);

QWEN3ASR_API int qwen3aligner_init(qwen3aligner_handle* handle);
QWEN3ASR_API int qwen3aligner_init_with_device(qwen3aligner_handle* handle, int device_index);
QWEN3ASR_API int qwen3aligner_init_with_device_name(qwen3aligner_handle* handle, const char* device_name);
QWEN3ASR_API void qwen3aligner_free(qwen3aligner_handle handle);
QWEN3ASR_API const char* qwen3aligner_get_device_name(qwen3aligner_handle handle);

QWEN3ASR_API int qwen3aligner_load_model(qwen3aligner_handle handle, const char* model_path);
QWEN3ASR_API int qwen3aligner_load_korean_dict(qwen3aligner_handle handle, const char* dict_path);

typedef struct {
    const char* language;
    int32_t n_threads;
} qwen3aligner_params;

QWEN3ASR_API void qwen3aligner_set_progress_callback(qwen3aligner_handle handle, qwen3_progress_callback callback);

typedef struct {
    char* word;
    float start;
    float end;
    float conf_word;
    float conf_start_time;
    float conf_end_time;
} qwen3aligned_word;

typedef struct {
    float start;
    float end;
    char* text;
    qwen3aligned_word* words;
    int32_t n_words;
} qwen3aligned_utterance;

typedef struct {
    qwen3aligned_utterance* utterances;
    int32_t n_utterances;
    int64_t t_total_ms;
} qwen3alignment_result;

QWEN3ASR_API void qwen3aligner_free_result(qwen3alignment_result* result);

QWEN3ASR_API int qwen3aligner_align_wav_file(
    qwen3aligner_handle handle,
    const char* wav_path,
    const char* text,
    const qwen3aligner_params* params,
    qwen3alignment_result* result
);

QWEN3ASR_API int qwen3aligner_align_pcm(
    qwen3aligner_handle handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const char* text,
    const qwen3aligner_params* params,
    qwen3alignment_result* result
);

QWEN3ASR_API int qwen3asr_transcribe_and_align_wav_file(
    qwen3asr_handle asr_handle,
    qwen3aligner_handle aligner_handle,
    const char* wav_path,
    const qwen3asr_params* asr_params,
    const qwen3aligner_params* align_params,
    qwen3alignment_result* result
);

QWEN3ASR_API int qwen3asr_transcribe_and_align_pcm(
    qwen3asr_handle asr_handle,
    qwen3aligner_handle aligner_handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const qwen3asr_params* asr_params,
    const qwen3aligner_params* align_params,
    qwen3alignment_result* result
);

#ifdef __cplusplus
}
#endif