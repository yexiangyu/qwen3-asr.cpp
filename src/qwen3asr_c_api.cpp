#include "qwen3asr_c_api.h"
#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "logger.h"
#include "batch_scheduler.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <future>

namespace {

static thread_local std::string g_last_error;
static thread_local qwen3_log_callback g_log_callback = nullptr;
static bool g_logger_initialized = false;

static void default_log_handler(qwen3_log_level level, const char* message) {
    if (g_log_callback) {
        g_log_callback(level, message);
    }
}

static std::vector<float> pcm_to_float(const int16_t* pcm, int32_t n) {
    std::vector<float> out(n);
    for (int32_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }
    return out;
}

}

extern "C" {

const char* qwen3asr_get_version(void) {
    return "0.1.0";
}

int qwen3_get_device_count(void) {
    return static_cast<int>(ggml_backend_dev_count());
}

int qwen3_get_device_info(int index, qwen3_device_info* info) {
    if (index < 0 || index >= static_cast<int>(ggml_backend_dev_count())) {
        g_last_error = "Device index out of range";
        return -1;
    }
    
    ggml_backend_dev_t dev = ggml_backend_dev_get(index);
    if (!dev) {
        g_last_error = "Failed to get device";
        return -1;
    }
    
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    
    info->name = strdup(props.name);
    info->description = strdup(props.description);
    info->type = static_cast<qwen3_device_type>(props.type);
    info->memory_free = props.memory_free;
    info->memory_total = props.memory_total;
    
    return 0;
}

void qwen3_free_device_info(qwen3_device_info* info) {
    if (info) {
        if (info->name) free(info->name);
        if (info->description) free(info->description);
        info->name = nullptr;
        info->description = nullptr;
    }
}

void qwen3_set_log_callback(qwen3_log_callback callback) {
    g_log_callback = callback;
}

void qwen3_set_log_level(qwen3_log_level level) {
    qwen3_asr::set_log_level(level);
}

const char* qwen3_get_last_error(void) {
    return g_last_error.c_str();
}

struct qwen3asr_handle_t {
    qwen3_asr::Qwen3ASR* asr;
    std::string device_name;
    qwen3_progress_callback progress_cb;
    
    qwen3asr_handle_t() : asr(nullptr), progress_cb(nullptr) {}
};

struct qwen3aligner_handle_t {
    qwen3_asr::ForcedAligner* aligner;
    std::string device_name;
    qwen3_progress_callback progress_cb;
    
    qwen3aligner_handle_t() : aligner(nullptr), progress_cb(nullptr) {}
};

int qwen3asr_init(qwen3asr_handle* handle) {
    if (!g_logger_initialized) {
        qwen3_asr::init_logger();
        g_logger_initialized = true;
    }
    
    *handle = new qwen3asr_handle_t();
    (*handle)->asr = new qwen3_asr::Qwen3ASR();
    (*handle)->device_name = "auto";
    
    return 0;
}

int qwen3asr_init_with_device(qwen3asr_handle* handle, int device_index) {
    int ret = qwen3asr_init(handle);
    if (ret != 0) return ret;
    
    qwen3_device_info info;
    if (qwen3_get_device_info(device_index, &info) != 0) {
        qwen3asr_free(*handle);
        *handle = nullptr;
        return -1;
    }
    
    (*handle)->device_name = info.name;
    qwen3_free_device_info(&info);
    
    return 0;
}

int qwen3asr_init_with_device_name(qwen3asr_handle* handle, const char* device_name) {
    int ret = qwen3asr_init(handle);
    if (ret != 0) return ret;
    
    if (device_name) {
        (*handle)->device_name = device_name;
    }
    
    return 0;
}

void qwen3asr_free(qwen3asr_handle handle) {
    if (handle) {
        if (handle->asr) {
            delete handle->asr;
        }
        delete handle;
    }
}

const char* qwen3asr_get_device_name(qwen3asr_handle handle) {
    if (!handle) return nullptr;
    return handle->device_name.c_str();
}

int qwen3asr_load_model(qwen3asr_handle handle, const char* model_path) {
    if (!handle || !handle->asr) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!model_path) {
        g_last_error = "Model path is null";
        return -1;
    }
    
    std::string device = handle->device_name;
    if (device == "auto") {
        device = "";
    }
    
    if (!handle->asr->load_model(model_path, device)) {
        g_last_error = handle->asr->get_error();
        return -1;
    }
    
    return 0;
}

void qwen3asr_set_progress_callback(qwen3asr_handle handle, qwen3_progress_callback callback) {
    if (handle) {
        handle->progress_cb = callback;
    }
}

void qwen3asr_free_result(qwen3asr_result* result) {
    if (result) {
        if (result->text) free(result->text);
        if (result->text_content) free(result->text_content);
        if (result->token_ids) free(result->token_ids);
        if (result->token_confs) free(result->token_confs);
        result->text = nullptr;
        result->text_content = nullptr;
        result->token_ids = nullptr;
        result->token_confs = nullptr;
        result->n_tokens = 0;
    }
}

int qwen3asr_transcribe_wav_file(
    qwen3asr_handle handle,
    const char* wav_path,
    const qwen3asr_params* params,
    qwen3asr_result* result
) {
    if (!handle || !handle->asr) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!wav_path) {
        g_last_error = "WAV path is null";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    qwen3_asr::transcribe_params tp;
    if (params) {
        tp.max_tokens = params->max_tokens;
        tp.language = params->language ? params->language : "";
        tp.context = params->context ? params->context : "";
        tp.n_threads = params->n_threads;
    }
    
    auto internal_result = handle->asr->transcribe(wav_path, tp);
    
    if (!internal_result.success) {
        g_last_error = internal_result.error_msg;
        return -1;
    }
    
    result->text = strdup(internal_result.text.c_str());
    result->text_content = strdup(internal_result.text_content.c_str());
    
    if (!result->text || !result->text_content) {
        if (result->text) free(result->text);
        if (result->text_content) free(result->text_content);
        g_last_error = "Failed to allocate memory for result strings";
        return -1;
    }
    
    result->n_tokens = static_cast<int32_t>(internal_result.tokens.size());
    result->t_mel_ms = internal_result.t_mel_ms;
    result->t_encode_ms = internal_result.t_encode_ms;
    result->t_decode_ms = internal_result.t_decode_ms;
    result->t_total_ms = internal_result.t_total_ms;
    
    if (internal_result.tokens.size() > 0) {
        result->token_ids = (int32_t*)malloc(internal_result.tokens.size() * sizeof(int32_t));
        result->token_confs = (float*)malloc(internal_result.token_confidences.size() * sizeof(float));
        
        if (!result->token_ids || !result->token_confs) {
            free(result->text);
            free(result->text_content);
            if (result->token_ids) free(result->token_ids);
            if (result->token_confs) free(result->token_confs);
            g_last_error = "Failed to allocate memory for token arrays";
            return -1;
        }
        
        for (size_t i = 0; i < internal_result.tokens.size(); ++i) {
            result->token_ids[i] = internal_result.tokens[i];
            result->token_confs[i] = internal_result.token_confidences[i];
        }
    } else {
        result->token_ids = nullptr;
        result->token_confs = nullptr;
    }
    
    return 0;
}

int qwen3asr_transcribe_pcm(
    qwen3asr_handle handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const qwen3asr_params* params,
    qwen3asr_result* result
) {
    if (!handle || !handle->asr) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!pcm_samples) {
        g_last_error = "PCM samples pointer is null";
        return -1;
    }
    
    if (n_samples <= 0) {
        g_last_error = "Invalid number of samples";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    std::vector<float> float_samples = pcm_to_float(pcm_samples, n_samples);
    
    qwen3_asr::transcribe_params tp;
    if (params) {
        tp.max_tokens = params->max_tokens;
        tp.language = params->language ? params->language : "";
        tp.context = params->context ? params->context : "";
        tp.n_threads = params->n_threads;
    }
    
    auto internal_result = handle->asr->transcribe(float_samples.data(), n_samples, tp);
    
    if (!internal_result.success) {
        g_last_error = internal_result.error_msg;
        return -1;
    }
    
    result->text = strdup(internal_result.text.c_str());
    result->text_content = strdup(internal_result.text_content.c_str());
    
    if (!result->text || !result->text_content) {
        if (result->text) free(result->text);
        if (result->text_content) free(result->text_content);
        g_last_error = "Failed to allocate memory for result strings";
        return -1;
    }
    
    result->n_tokens = static_cast<int32_t>(internal_result.tokens.size());
    result->t_mel_ms = internal_result.t_mel_ms;
    result->t_encode_ms = internal_result.t_encode_ms;
    result->t_decode_ms = internal_result.t_decode_ms;
    result->t_total_ms = internal_result.t_total_ms;
    
    if (internal_result.tokens.size() > 0) {
        result->token_ids = (int32_t*)malloc(internal_result.tokens.size() * sizeof(int32_t));
        result->token_confs = (float*)malloc(internal_result.token_confidences.size() * sizeof(float));
        
        if (!result->token_ids || !result->token_confs) {
            free(result->text);
            free(result->text_content);
            if (result->token_ids) free(result->token_ids);
            if (result->token_confs) free(result->token_confs);
            g_last_error = "Failed to allocate memory for token arrays";
            return -1;
        }
        
        for (size_t i = 0; i < internal_result.tokens.size(); ++i) {
            result->token_ids[i] = internal_result.tokens[i];
            result->token_confs[i] = internal_result.token_confidences[i];
        }
    } else {
        result->token_ids = nullptr;
        result->token_confs = nullptr;
    }
    
    return 0;
}

int qwen3aligner_init(qwen3aligner_handle* handle) {
    if (!g_logger_initialized) {
        qwen3_asr::init_logger();
        g_logger_initialized = true;
    }
    
    *handle = new qwen3aligner_handle_t();
    (*handle)->aligner = new qwen3_asr::ForcedAligner();
    (*handle)->device_name = "auto";
    
    return 0;
}

int qwen3aligner_init_with_device(qwen3aligner_handle* handle, int device_index) {
    int ret = qwen3aligner_init(handle);
    if (ret != 0) return ret;
    
    qwen3_device_info info;
    if (qwen3_get_device_info(device_index, &info) != 0) {
        qwen3aligner_free(*handle);
        *handle = nullptr;
        return -1;
    }
    
    (*handle)->device_name = info.name;
    qwen3_free_device_info(&info);
    
    return 0;
}

int qwen3aligner_init_with_device_name(qwen3aligner_handle* handle, const char* device_name) {
    int ret = qwen3aligner_init(handle);
    if (ret != 0) return ret;
    
    if (device_name) {
        (*handle)->device_name = device_name;
    }
    
    return 0;
}

void qwen3aligner_free(qwen3aligner_handle handle) {
    if (handle) {
        if (handle->aligner) {
            delete handle->aligner;
        }
        delete handle;
    }
}

const char* qwen3aligner_get_device_name(qwen3aligner_handle handle) {
    if (!handle) return nullptr;
    return handle->device_name.c_str();
}

int qwen3aligner_load_model(qwen3aligner_handle handle, const char* model_path) {
    if (!handle || !handle->aligner) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!model_path) {
        g_last_error = "Model path is null";
        return -1;
    }
    
    std::string device = handle->device_name;
    if (device == "auto") {
        device = "";
    }
    
    if (!handle->aligner->load_model(model_path, device)) {
        g_last_error = handle->aligner->get_error();
        return -1;
    }
    
    return 0;
}

int qwen3aligner_load_korean_dict(qwen3aligner_handle handle, const char* dict_path) {
    if (!handle || !handle->aligner) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!dict_path) {
        g_last_error = "Dictionary path is null";
        return -1;
    }
    
    if (!handle->aligner->load_korean_dict(dict_path)) {
        g_last_error = "Failed to load Korean dictionary";
        return -1;
    }
    
    return 0;
}

void qwen3aligner_set_progress_callback(qwen3aligner_handle handle, qwen3_progress_callback callback) {
    if (handle) {
        handle->progress_cb = callback;
    }
}

void qwen3aligner_free_result(qwen3alignment_result* result) {
    if (result) {
        if (result->utterances) {
            for (int32_t i = 0; i < result->n_utterances; ++i) {
                qwen3aligned_utterance* utt = &result->utterances[i];
                if (utt->text) free(utt->text);
                if (utt->words) {
                    for (int32_t j = 0; j < utt->n_words; ++j) {
                        if (utt->words[j].word) free(utt->words[j].word);
                    }
                    free(utt->words);
                }
            }
            free(result->utterances);
        }
        result->utterances = nullptr;
        result->n_utterances = 0;
    }
}

static bool fill_alignment_result(const qwen3_asr::alignment_result& internal, qwen3alignment_result* result) {
    result->n_utterances = static_cast<int32_t>(internal.utterances.size());
    result->t_mel_ms = internal.t_mel_ms;
    result->t_encode_ms = internal.t_encode_ms;
    result->t_decode_ms = internal.t_decode_ms;
    result->t_total_ms = internal.t_total_ms;
    
    if (internal.utterances.size() > 0) {
        result->utterances = (qwen3aligned_utterance*)malloc(internal.utterances.size() * sizeof(qwen3aligned_utterance));
        if (!result->utterances) {
            g_last_error = "Failed to allocate memory for utterances";
            return false;
        }
        
        for (size_t i = 0; i < internal.utterances.size(); ++i) {
            const auto& utt = internal.utterances[i];
            result->utterances[i].start = utt.start;
            result->utterances[i].end = utt.end;
            result->utterances[i].text = strdup(utt.text.c_str());
            result->utterances[i].n_words = static_cast<int32_t>(utt.words.size());
            
            if (!result->utterances[i].text) {
                for (size_t k = 0; k < i; ++k) {
                    free(result->utterances[k].text);
                    if (result->utterances[k].words) {
                        for (size_t j = 0; j < result->utterances[k].n_words; ++j) {
                            free(result->utterances[k].words[j].word);
                        }
                        free(result->utterances[k].words);
                    }
                }
                free(result->utterances);
                result->utterances = nullptr;
                g_last_error = "Failed to allocate memory for utterance text";
                return false;
            }
            
            if (utt.words.size() > 0) {
                result->utterances[i].words = (qwen3aligned_word*)malloc(utt.words.size() * sizeof(qwen3aligned_word));
                if (!result->utterances[i].words) {
                    for (size_t k = 0; k <= i; ++k) {
                        free(result->utterances[k].text);
                        if (result->utterances[k].words) {
                            for (size_t j = 0; j < result->utterances[k].n_words; ++j) {
                                free(result->utterances[k].words[j].word);
                            }
                            free(result->utterances[k].words);
                        }
                    }
                    free(result->utterances);
                    result->utterances = nullptr;
                    g_last_error = "Failed to allocate memory for words";
                    return false;
                }
                
                for (size_t j = 0; j < utt.words.size(); ++j) {
                    const auto& w = utt.words[j];
                    result->utterances[i].words[j].word = strdup(w.word.c_str());
                    result->utterances[i].words[j].start = w.start;
                    result->utterances[i].words[j].end = w.end;
                    result->utterances[i].words[j].conf_word = w.conf_word;
                    result->utterances[i].words[j].conf_start_time = w.conf_start_time;
                    result->utterances[i].words[j].conf_end_time = w.conf_end_time;
                    
                    if (!result->utterances[i].words[j].word) {
                        for (size_t kk = 0; kk <= i; ++kk) {
                            free(result->utterances[kk].text);
                            if (result->utterances[kk].words) {
                                for (size_t jj = 0; jj < result->utterances[kk].n_words; ++jj) {
                                    free(result->utterances[kk].words[jj].word);
                                }
                                free(result->utterances[kk].words);
                            }
                        }
                        free(result->utterances);
                        result->utterances = nullptr;
                        g_last_error = "Failed to allocate memory for word text";
                        return false;
                    }
                }
            } else {
                result->utterances[i].words = nullptr;
            }
        }
    } else {
        result->utterances = nullptr;
    }
    return true;
}

int qwen3aligner_align_wav_file(
    qwen3aligner_handle handle,
    const char* wav_path,
    const char* text,
    const qwen3aligner_params* params,
    qwen3alignment_result* result
) {
    if (!handle || !handle->aligner) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!wav_path) {
        g_last_error = "WAV path is null";
        return -1;
    }
    
    if (!text) {
        g_last_error = "Text is null";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    qwen3_asr::align_params ap;
    if (params) {
        ap.print_timing = false;
    }
    
    std::string lang = params && params->language ? params->language : "";
    auto internal = handle->aligner->align(wav_path, text, lang, ap);
    
    if (!internal.success) {
        g_last_error = internal.error_msg;
        return -1;
    }
    
    if (!fill_alignment_result(internal, result)) {
        return -1;
    }
    return 0;
}

int qwen3aligner_align_pcm(
    qwen3aligner_handle handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const char* text,
    const qwen3aligner_params* params,
    qwen3alignment_result* result
) {
    if (!handle || !handle->aligner) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!pcm_samples) {
        g_last_error = "PCM samples pointer is null";
        return -1;
    }
    
    if (n_samples <= 0) {
        g_last_error = "Invalid number of samples";
        return -1;
    }
    
    if (!text) {
        g_last_error = "Text is null";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    std::vector<float> float_samples = pcm_to_float(pcm_samples, n_samples);
    
    qwen3_asr::align_params ap;
    if (params) {
        ap.print_timing = false;
    }
    
    std::string lang = params && params->language ? params->language : "";
    auto internal = handle->aligner->align(float_samples.data(), n_samples, text, lang, ap);
    
    if (!internal.success) {
        g_last_error = internal.error_msg;
        return -1;
    }
    
    if (!fill_alignment_result(internal, result)) {
        return -1;
    }
    return 0;
}

int qwen3asr_transcribe_and_align_wav_file(
    qwen3asr_handle asr_handle,
    qwen3aligner_handle aligner_handle,
    const char* wav_path,
    const qwen3asr_params* asr_params,
    const qwen3aligner_params* align_params,
    qwen3alignment_result* result
) {
    if (!asr_handle || !asr_handle->asr) {
        g_last_error = "Invalid ASR handle";
        return -1;
    }
    
    if (!aligner_handle || !aligner_handle->aligner) {
        g_last_error = "Invalid aligner handle";
        return -1;
    }
    
    if (!wav_path) {
        g_last_error = "WAV path is null";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    qwen3_asr::transcribe_params tp;
    if (asr_params) {
        tp.max_tokens = asr_params->max_tokens;
        tp.language = asr_params->language ? asr_params->language : "";
        tp.context = asr_params->context ? asr_params->context : "";
        tp.n_threads = asr_params->n_threads;
    }
    
    auto asr_result = asr_handle->asr->transcribe(wav_path, tp);
    if (!asr_result.success) {
        g_last_error = asr_result.error_msg;
        return -1;
    }
    
    std::string lang = align_params && align_params->language ? align_params->language : "";
    
    qwen3_asr::align_params ap;
    auto internal = aligner_handle->aligner->align_with_asr_tokens(
        wav_path,
        asr_result.text_content,
        asr_result.tokens,
        asr_result.token_confidences,
        asr_result.token_strings,
        lang,
        ap
    );
    
    if (!internal.success) {
        g_last_error = internal.error_msg;
        return -1;
    }
    
    if (!fill_alignment_result(internal, result)) {
        return -1;
    }
    return 0;
}

int qwen3asr_transcribe_and_align_pcm(
    qwen3asr_handle asr_handle,
    qwen3aligner_handle aligner_handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const qwen3asr_params* asr_params,
    const qwen3aligner_params* align_params,
    qwen3alignment_result* result
) {
    if (!asr_handle || !asr_handle->asr) {
        g_last_error = "Invalid ASR handle";
        return -1;
    }
    
    if (!aligner_handle || !aligner_handle->aligner) {
        g_last_error = "Invalid aligner handle";
        return -1;
    }
    
    if (!pcm_samples) {
        g_last_error = "PCM samples pointer is null";
        return -1;
    }
    
    if (n_samples <= 0) {
        g_last_error = "Invalid number of samples";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    std::vector<float> float_samples = pcm_to_float(pcm_samples, n_samples);
    
    qwen3_asr::transcribe_params tp;
    if (asr_params) {
        tp.max_tokens = asr_params->max_tokens;
        tp.language = asr_params->language ? asr_params->language : "";
        tp.context = asr_params->context ? asr_params->context : "";
        tp.n_threads = asr_params->n_threads;
    }
    
    auto asr_result = asr_handle->asr->transcribe(float_samples.data(), n_samples, tp);
    if (!asr_result.success) {
        g_last_error = asr_result.error_msg;
        return -1;
    }
    
    std::string lang = align_params && align_params->language ? align_params->language : "";
    
    qwen3_asr::align_params ap;
    auto internal = aligner_handle->aligner->align_with_asr_tokens(
        float_samples.data(),
        n_samples,
        asr_result.text_content,
        asr_result.tokens,
        asr_result.token_confidences,
        asr_result.token_strings,
        lang,
        ap
    );
    
    if (!internal.success) {
        g_last_error = internal.error_msg;
        return -1;
    }
    
    if (!fill_alignment_result(internal, result)) {
        return -1;
    }
    return 0;
}

int qwen3asr_transcribe_align_pcm_combined(
    qwen3asr_handle asr_handle,
    qwen3aligner_handle aligner_handle,
    const int16_t* pcm_samples,
    int32_t n_samples,
    const qwen3asr_params* asr_params,
    const qwen3aligner_params* align_params,
    qwen3combined_result* result
) {
    if (!asr_handle || !asr_handle->asr) {
        g_last_error = "Invalid ASR handle";
        return -1;
    }
    
    if (!aligner_handle || !aligner_handle->aligner) {
        g_last_error = "Invalid aligner handle";
        return -1;
    }
    
    if (!pcm_samples) {
        g_last_error = "PCM samples pointer is null";
        return -1;
    }
    
    if (n_samples <= 0) {
        g_last_error = "Invalid number of samples";
        return -1;
    }
    
    if (!result) {
        g_last_error = "Result pointer is null";
        return -1;
    }
    
    std::vector<float> float_samples = pcm_to_float(pcm_samples, n_samples);
    
    qwen3_asr::transcribe_params tp;
    if (asr_params) {
        tp.max_tokens = asr_params->max_tokens;
        tp.language = asr_params->language ? asr_params->language : "";
        tp.context = asr_params->context ? asr_params->context : "";
        tp.n_threads = asr_params->n_threads;
    }
    
    auto asr_result = asr_handle->asr->transcribe(float_samples.data(), n_samples, tp);
    if (!asr_result.success) {
        g_last_error = asr_result.error_msg;
        return -1;
    }
    
    result->transcription.text = strdup(asr_result.text.c_str());
    result->transcription.text_content = strdup(asr_result.text_content.c_str());
    
    if (!result->transcription.text || !result->transcription.text_content) {
        if (result->transcription.text) free(result->transcription.text);
        if (result->transcription.text_content) free(result->transcription.text_content);
        g_last_error = "Failed to allocate memory for transcription strings";
        return -1;
    }
    
    result->transcription.n_tokens = static_cast<int32_t>(asr_result.tokens.size());
    result->transcription.t_mel_ms = asr_result.t_mel_ms;
    result->transcription.t_encode_ms = asr_result.t_encode_ms;
    result->transcription.t_decode_ms = asr_result.t_decode_ms;
    result->transcription.t_total_ms = asr_result.t_total_ms;
    
    if (asr_result.tokens.size() > 0) {
        result->transcription.token_ids = (int32_t*)malloc(asr_result.tokens.size() * sizeof(int32_t));
        result->transcription.token_confs = (float*)malloc(asr_result.token_confidences.size() * sizeof(float));
        
        if (!result->transcription.token_ids || !result->transcription.token_confs) {
            free(result->transcription.text);
            free(result->transcription.text_content);
            if (result->transcription.token_ids) free(result->transcription.token_ids);
            if (result->transcription.token_confs) free(result->transcription.token_confs);
            g_last_error = "Failed to allocate memory for token arrays";
            return -1;
        }
        
        for (size_t i = 0; i < asr_result.tokens.size(); ++i) {
            result->transcription.token_ids[i] = asr_result.tokens[i];
            result->transcription.token_confs[i] = asr_result.token_confidences[i];
        }
    } else {
        result->transcription.token_ids = nullptr;
        result->transcription.token_confs = nullptr;
    }
    
    std::string lang = align_params && align_params->language ? align_params->language : "";
    
    qwen3_asr::align_params ap;
    auto align_internal = aligner_handle->aligner->align_with_asr_tokens(
        float_samples.data(),
        n_samples,
        asr_result.text_content,
        asr_result.tokens,
        asr_result.token_confidences,
        asr_result.token_strings,
        lang,
        ap
    );
    
    if (!align_internal.success) {
        g_last_error = align_internal.error_msg;
        qwen3asr_free_result(&result->transcription);
        return -1;
    }
    
    if (!fill_alignment_result(align_internal, &result->alignment)) {
        qwen3asr_free_result(&result->transcription);
        return -1;
    }
    
    return 0;
}

void qwen3asr_free_combined_result(qwen3combined_result* result) {
    if (result) {
        qwen3asr_free_result(&result->transcription);
        qwen3aligner_free_result(&result->alignment);
    }
}

int qwen3asr_transcribe_batch(
    qwen3asr_handle handle,
    const int16_t** pcm_samples,
    const int32_t* n_samples,
    int n_requests,
    const qwen3asr_params* params,
    qwen3asr_result* results
) {
    if (!handle || !handle->asr) {
        g_last_error = "Invalid handle";
        return -1;
    }
    
    if (!pcm_samples || !n_samples || !results) {
        g_last_error = "Null pointer argument";
        return -1;
    }
    
    if (n_requests <= 0) {
        g_last_error = "Invalid number of requests";
        return -1;
    }
    
    std::vector<std::vector<float>> float_buffers(n_requests);
    std::vector<const float*> float_ptrs(n_requests);
    std::vector<int> sample_counts(n_requests);
    
    for (int i = 0; i < n_requests; ++i) {
        if (!pcm_samples[i]) {
            g_last_error = "Null PCM pointer at index " + std::to_string(i);
            return -1;
        }
        if (n_samples[i] <= 0) {
            g_last_error = "Invalid sample count at index " + std::to_string(i);
            return -1;
        }
        float_buffers[i] = pcm_to_float(pcm_samples[i], n_samples[i]);
        float_ptrs[i] = float_buffers[i].data();
        sample_counts[i] = n_samples[i];
    }
    
    qwen3_asr::transcribe_params tp;
    if (params) {
        tp.max_tokens = params->max_tokens;
        tp.language = params->language ? params->language : "";
        tp.context = params->context ? params->context : "";
        tp.n_threads = params->n_threads;
    }
    
    auto batch_results = handle->asr->transcribe_batch(float_ptrs, sample_counts, tp);
    
    if (batch_results.size() != static_cast<size_t>(n_requests)) {
        g_last_error = "Batch transcription returned wrong number of results";
        return -1;
    }
    
    for (int i = 0; i < n_requests; ++i) {
        const auto& br = batch_results[i];
        
        if (!br.success) {
            results[i].text = nullptr;
            results[i].text_content = nullptr;
            results[i].token_ids = nullptr;
            results[i].token_confs = nullptr;
            results[i].n_tokens = 0;
            results[i].t_mel_ms = 0;
            results[i].t_encode_ms = 0;
            results[i].t_decode_ms = 0;
            results[i].t_total_ms = 0;
            continue;
        }
        
        results[i].text = strdup(br.text.c_str());
        results[i].text_content = strdup(br.text_content.c_str());
        
        if (!results[i].text || !results[i].text_content) {
            for (int j = 0; j <= i; ++j) {
                if (results[j].text) free(results[j].text);
                if (results[j].text_content) free(results[j].text_content);
                if (results[j].token_ids) free(results[j].token_ids);
                if (results[j].token_confs) free(results[j].token_confs);
            }
            g_last_error = "Failed to allocate memory for result strings";
            return -1;
        }
        
        results[i].n_tokens = static_cast<int32_t>(br.tokens.size());
        results[i].t_mel_ms = 0;
        results[i].t_encode_ms = 0;
        results[i].t_decode_ms = 0;
        results[i].t_total_ms = 0;
        
        if (br.tokens.size() > 0) {
            results[i].token_ids = (int32_t*)malloc(br.tokens.size() * sizeof(int32_t));
            results[i].token_confs = (float*)malloc(br.token_confs.size() * sizeof(float));
            
            if (!results[i].token_ids || !results[i].token_confs) {
                for (int j = 0; j <= i; ++j) {
                    if (results[j].text) free(results[j].text);
                    if (results[j].text_content) free(results[j].text_content);
                    if (results[j].token_ids) free(results[j].token_ids);
                    if (results[j].token_confs) free(results[j].token_confs);
                }
                g_last_error = "Failed to allocate memory for token arrays";
                return -1;
            }
            
            for (size_t j = 0; j < br.tokens.size(); ++j) {
                results[i].token_ids[j] = br.tokens[j];
                results[i].token_confs[j] = br.token_confs[j];
            }
        } else {
            results[i].token_ids = nullptr;
            results[i].token_confs = nullptr;
        }
    }
    
    return 0;
}

struct qwen3_batch_scheduler_t {
    qwen3_asr::BatchScheduler scheduler;
    std::unordered_map<int, std::future<std::string>> pending_futures;
    std::mutex futures_mutex;
    int next_result_id;
    
    qwen3_batch_scheduler_t() : next_result_id(0) {}
};

qwen3_batch_scheduler_handle qwen3_batch_scheduler_init(
    qwen3asr_handle asr,
    qwen3aligner_handle aligner,
    qwen3_batch_config config
) {
    if (!asr || !asr->asr) {
        g_last_error = "Invalid ASR handle";
        return nullptr;
    }
    
    auto* scheduler = new qwen3_batch_scheduler_t();
    scheduler->scheduler.set_asr(asr->asr);
    
    if (aligner && aligner->aligner) {
        scheduler->scheduler.set_aligner(aligner->aligner);
    }
    
    scheduler->scheduler.set_batch_size(config.max_batch_size > 0 ? config.max_batch_size : 2);
    scheduler->scheduler.set_timeout_ms(config.batch_timeout_ms > 0 ? config.batch_timeout_ms : 100);
    
    return scheduler;
}

void qwen3_batch_scheduler_free(qwen3_batch_scheduler_handle scheduler) {
    if (scheduler) {
        auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
        s->scheduler.stop();
        delete s;
    }
}

int qwen3_batch_scheduler_start(qwen3_batch_scheduler_handle scheduler) {
    if (!scheduler) {
        g_last_error = "Invalid scheduler handle";
        return -1;
    }
    
    auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
    s->scheduler.start();
    
    return 0;
}

void qwen3_batch_scheduler_stop(qwen3_batch_scheduler_handle scheduler) {
    if (scheduler) {
        auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
        s->scheduler.stop();
    }
}

int qwen3_batch_scheduler_submit(
    qwen3_batch_scheduler_handle scheduler,
    const int16_t* pcm,
    int32_t n_samples,
    const char* language,
    const char* context,
    int max_tokens,
    int* request_id
) {
    if (!scheduler) {
        g_last_error = "Invalid scheduler handle";
        return -1;
    }
    
    if (!pcm || n_samples <= 0) {
        g_last_error = "Invalid PCM data";
        return -1;
    }
    
    auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
    
    std::vector<int16_t> pcm_vec(pcm, pcm + n_samples);
    std::string lang = language ? language : "";
    std::string ctx = context ? context : "";
    
    int id = s->next_result_id++;
    
    auto future = s->scheduler.submit_request(pcm_vec, lang, ctx, max_tokens > 0 ? max_tokens : 1024);
    
    {
        std::lock_guard<std::mutex> lock(s->futures_mutex);
        s->pending_futures[id] = std::move(future);
    }
    
    if (request_id) {
        *request_id = id;
    }
    
    return 0;
}

int qwen3_batch_scheduler_get_result(
    qwen3_batch_scheduler_handle scheduler,
    int request_id,
    char** json_result
) {
    if (!scheduler) {
        g_last_error = "Invalid scheduler handle";
        return -1;
    }
    
    if (!json_result) {
        g_last_error = "Null result pointer";
        return -1;
    }
    
    auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
    
    std::future<std::string> future;
    {
        std::lock_guard<std::mutex> lock(s->futures_mutex);
        auto it = s->pending_futures.find(request_id);
        if (it == s->pending_futures.end()) {
            g_last_error = "Invalid request ID";
            return -1;
        }
        future = std::move(it->second);
        s->pending_futures.erase(it);
    }
    
    try {
        std::string result = future.get();
        *json_result = strdup(result.c_str());
        if (!*json_result) {
            g_last_error = "Failed to allocate memory for result";
            return -1;
        }
    } catch (const std::exception& e) {
        g_last_error = std::string("Exception: ") + e.what();
        return -1;
    }
    
    return 0;
}

void qwen3_batch_scheduler_free_result(char* json_result) {
    if (json_result) {
        free(json_result);
    }
}

int qwen3_batch_scheduler_get_pending_count(qwen3_batch_scheduler_handle scheduler) {
    if (!scheduler) {
        return 0;
    }
    
    auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
    return s->scheduler.get_pending_count();
}

int qwen3_batch_scheduler_is_running(qwen3_batch_scheduler_handle scheduler) {
    if (!scheduler) {
        return 0;
    }
    
    auto* s = static_cast<qwen3_batch_scheduler_t*>(scheduler);
    return s->scheduler.is_running() ? 1 : 0;
}

}