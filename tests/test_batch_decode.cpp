#include "asr/transcribe/decoder.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace asr::transcribe::decoder;

static bool validate_logits(const std::vector<float>& logits, const char* name) {
    bool has_nan = false;
    bool all_zero = true;
    float max_val = -1e10f;
    float min_val = 1e10f;
    
    for (float v : logits) {
        if (std::isnan(v)) has_nan = true;
        if (v != 0.0f) all_zero = false;
        max_val = std::max(max_val, v);
        min_val = std::min(min_val, v);
    }
    
    if (has_nan) {
        fprintf(stderr, "FAIL: %s logits contain NaN\n", name);
        return false;
    }
    if (all_zero) {
        fprintf(stderr, "FAIL: %s logits are all zeros\n", name);
        return false;
    }
    
    fprintf(stderr, "  %s logits range: [%.4f, %.4f]\n", name, min_val, max_val);
    return true;
}

bool test_batch_init_free(const char* model_path) {
    fprintf(stderr, "\n=== Testing Batch Init/Free ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    int max_batch_size = 4;
    int max_total_capacity = 8192;
    
    BatchDecodeState* state = init_batch(config, max_batch_size, max_total_capacity);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    fprintf(stderr, "Batch state initialized\n");
    fprintf(stderr, "Device: %s\n", batch_get_device_name(state));
    
    HyperParams hp = batch_get_hparams(state);
    fprintf(stderr, "Hyperparams:\n");
    fprintf(stderr, "  vocab_size: %d\n", hp.vocab_size);
    fprintf(stderr, "  hidden_size: %d\n", hp.hidden_size);
    fprintf(stderr, "  n_layers: %d\n", hp.n_layers);
    fprintf(stderr, "  n_heads: %d\n", hp.n_heads);
    fprintf(stderr, "  n_kv_heads: %d\n", hp.n_kv_heads);
    fprintf(stderr, "  head_dim: %d\n", hp.head_dim);
    
    if (batch_get_capacity(state) != max_total_capacity) {
        fprintf(stderr, "FAIL: Capacity mismatch: %d vs expected %d\n",
                batch_get_capacity(state), max_total_capacity);
        passed = false;
    }
    
    if (batch_get_n_active(state) != 0) {
        fprintf(stderr, "FAIL: Should have 0 active slots after init\n");
        passed = false;
    }
    
    free_batch(state);
    fprintf(stderr, "Batch state freed\n");
    
    fprintf(stderr, "=== Batch Init/Free Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_single_sequence_add_decode(const char* model_path) {
    fprintf(stderr, "\n=== Testing Single Sequence Add/Decode ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 8192);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens;
    tokens.push_back(hp.audio_start_token);
    tokens.push_back(hp.audio_end_token);
    tokens.push_back(100);
    
    ErrorInfo error;
    int slot_id = batch_add_sequence(state, tokens, nullptr, 0, hp.hidden_size, -1, 16, "", &error);
    
    if (slot_id < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    fprintf(stderr, "Sequence added to slot %d\n", slot_id);
    
    if (batch_get_n_active(state) != 1) {
        fprintf(stderr, "FAIL: Should have 1 active slot\n");
        passed = false;
    }
    
    if (batch_get_slot_used(state, slot_id) != static_cast<int>(tokens.size())) {
        fprintf(stderr, "FAIL: Slot used mismatch: %d vs expected %zu\n",
                batch_get_slot_used(state, slot_id), tokens.size());
        passed = false;
    }
    
    std::vector<int> gen_tokens = batch_get_tokens(state, slot_id);
    fprintf(stderr, "Generated tokens count: %zu\n", gen_tokens.size());
    
    if (gen_tokens.empty()) {
        fprintf(stderr, "Note: First token may be EOS, no generated tokens\n");
    }
    
    fprintf(stderr, "Performing batch decode step...\n");
    if (!batch_decode_step(state, &error)) {
        fprintf(stderr, "FAIL: Batch decode step failed: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    std::vector<int> active_slots = batch_get_active_slots(state);
    fprintf(stderr, "Active slots after decode: %zu\n", active_slots.size());
    
    std::vector<int> eos_slots = batch_get_eos_slots(state);
    fprintf(stderr, "EOS slots: %zu\n", eos_slots.size());
    
    batch_remove_sequence(state, slot_id);
    
    if (batch_get_n_active(state) != 0) {
        fprintf(stderr, "FAIL: Should have 0 active slots after removal\n");
        passed = false;
    }
    
    batch_clear(state);
    free_batch(state);
    
    fprintf(stderr, "=== Single Sequence Add/Decode Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_two_sequences_different_lengths(const char* model_path) {
    fprintf(stderr, "\n=== Testing Two Sequences (Different Audio Lengths) ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 8192);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens_a;
    tokens_a.push_back(hp.audio_start_token);
    tokens_a.push_back(hp.audio_pad_token);
    tokens_a.push_back(hp.audio_pad_token);
    tokens_a.push_back(hp.audio_end_token);
    
    std::vector<int> tokens_b;
    tokens_b.push_back(hp.audio_start_token);
    tokens_b.push_back(hp.audio_pad_token);
    tokens_b.push_back(hp.audio_pad_token);
    tokens_b.push_back(hp.audio_pad_token);
    tokens_b.push_back(hp.audio_pad_token);
    tokens_b.push_back(hp.audio_end_token);
    
    std::vector<float> audio_a(tokens_a.size() * hp.hidden_size, 0.5f);
    std::vector<float> audio_b(tokens_b.size() * hp.hidden_size, 0.3f);
    
    ErrorInfo error;
    
    int audio_start_pos_a = 1;
    int slot_a = batch_add_sequence(state, tokens_a, audio_a.data(), 3, hp.hidden_size,
                                    audio_start_pos_a, 16, "chinese", &error);
    if (slot_a < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence A: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    int audio_start_pos_b = 1;
    int slot_b = batch_add_sequence(state, tokens_b, audio_b.data(), 5, hp.hidden_size,
                                    audio_start_pos_b, 16, "korean", &error);
    if (slot_b < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence B: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    fprintf(stderr, "Added sequences to slots %d and %d\n", slot_a, slot_b);
    
    if (batch_get_n_active(state) != 2) {
        fprintf(stderr, "FAIL: Should have 2 active slots\n");
        passed = false;
    }
    
    fprintf(stderr, "Slot A capacity: %d, used: %d\n",
            batch_get_slot_capacity(state, slot_a), batch_get_slot_used(state, slot_a));
    fprintf(stderr, "Slot B capacity: %d, used: %d\n",
            batch_get_slot_capacity(state, slot_b), batch_get_slot_used(state, slot_b));
    
    fprintf(stderr, "Performing batch decode steps...\n");
    for (int step = 0; step < 5; ++step) {
        if (!batch_decode_step(state, &error)) {
            fprintf(stderr, "FAIL: Batch decode step %d failed: %s\n", step, error.message.c_str());
            free_batch(state);
            return false;
        }
        
        fprintf(stderr, "Step %d: active=%d, slot_a_tokens=%zu, slot_b_tokens=%zu\n",
                step, batch_get_n_active(state),
                batch_get_tokens(state, slot_a).size(),
                batch_get_tokens(state, slot_b).size());
    }
    
    batch_clear(state);
    free_batch(state);
    
    fprintf(stderr, "=== Two Sequences Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_sequence_finish_before_others(const char* model_path) {
    fprintf(stderr, "\n=== Testing Sequence A Finishes Before B ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 8192);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens_a;
    tokens_a.push_back(hp.audio_start_token);
    tokens_a.push_back(hp.audio_end_token);
    
    std::vector<int> tokens_b;
    tokens_b.push_back(hp.audio_start_token);
    tokens_b.push_back(hp.audio_end_token);
    
    ErrorInfo error;
    
    int slot_a = batch_add_sequence(state, tokens_a, nullptr, 0, hp.hidden_size, -1, 8, "", &error);
    if (slot_a < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence A: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    int slot_b = batch_add_sequence(state, tokens_b, nullptr, 0, hp.hidden_size, -1, 32, "", &error);
    if (slot_b < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence B: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    fprintf(stderr, "Added sequences: slot_a=%d (max=8), slot_b=%d (max=32)\n", slot_a, slot_b);
    
    int max_steps = 40;
    for (int step = 0; step < max_steps && batch_get_n_active(state) > 0; ++step) {
        if (!batch_decode_step(state, &error)) {
            fprintf(stderr, "FAIL: Batch decode step %d failed: %s\n", step, error.message.c_str());
            free_batch(state);
            return false;
        }
        
        std::vector<int> eos_slots = batch_get_eos_slots(state);
        if (!eos_slots.empty()) {
            fprintf(stderr, "Step %d: EOS detected in slots: ", step);
            for (int s : eos_slots) {
                fprintf(stderr, "%d ", s);
            }
            fprintf(stderr, "\n");
            
            for (int s : eos_slots) {
                batch_remove_sequence(state, s);
            }
        }
        
        if (batch_get_n_active(state) == 0) {
            fprintf(stderr, "All sequences finished at step %d\n", step);
            break;
        }
        
        if (step == max_steps - 1) {
            fprintf(stderr, "Reached max steps, forcibly stopping\n");
        }
    }
    
    std::vector<int> tokens_a_result = batch_get_tokens(state, slot_a);
    std::vector<int> tokens_b_result = batch_get_tokens(state, slot_b);
    
    fprintf(stderr, "Slot A generated %zu tokens\n", tokens_a_result.size());
    fprintf(stderr, "Slot B generated %zu tokens\n", tokens_b_result.size());
    
    batch_clear(state);
    free_batch(state);
    
    fprintf(stderr, "=== Sequence Finish Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_add_new_sequence_while_active(const char* model_path) {
    fprintf(stderr, "\n=== Testing Add New Sequence While Others Active ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 16384);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens1;
    tokens1.push_back(hp.audio_start_token);
    tokens1.push_back(hp.audio_end_token);
    
    ErrorInfo error;
    
    int slot1 = batch_add_sequence(state, tokens1, nullptr, 0, hp.hidden_size, -1, 32, "", &error);
    if (slot1 < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence 1: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    fprintf(stderr, "Added sequence 1 to slot %d\n", slot1);
    
    for (int step = 0; step < 3; ++step) {
        if (!batch_decode_step(state, &error)) {
            fprintf(stderr, "FAIL: Decode step %d failed: %s\n", step, error.message.c_str());
            free_batch(state);
            return false;
        }
    }
    
    fprintf(stderr, "After 3 steps, adding new sequence...\n");
    
    int slot2 = batch_add_sequence(state, tokens1, nullptr, 0, hp.hidden_size, -1, 32, "english", &error);
    if (slot2 < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence 2: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    fprintf(stderr, "Added sequence 2 to slot %d\n", slot2);
    
    if (batch_get_n_active(state) != 2) {
        fprintf(stderr, "FAIL: Should have 2 active slots after adding second sequence\n");
        passed = false;
    }
    
    for (int step = 0; step < 5; ++step) {
        if (!batch_decode_step(state, &error)) {
            fprintf(stderr, "FAIL: Decode step %d failed: %s\n", step, error.message.c_str());
            free_batch(state);
            return false;
        }
        
        fprintf(stderr, "Step %d: active=%d, slot1_tokens=%zu, slot2_tokens=%zu\n",
                step, batch_get_n_active(state),
                batch_get_tokens(state, slot1).size(),
                batch_get_tokens(state, slot2).size());
        
        std::vector<int> eos_slots = batch_get_eos_slots(state);
        for (int s : eos_slots) {
            batch_remove_sequence(state, s);
        }
        
        if (batch_get_n_active(state) == 0) {
            break;
        }
    }
    
    fprintf(stderr, "Slot 1 seq_id: %d, tokens: %zu\n",
            batch_get_seq_id(state, slot1), batch_get_tokens(state, slot1).size());
    fprintf(stderr, "Slot 2 seq_id: %d, tokens: %zu\n",
            batch_get_seq_id(state, slot2), batch_get_tokens(state, slot2).size());
    
    batch_clear(state);
    free_batch(state);
    
    fprintf(stderr, "=== Add New Sequence Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_slot_text_extraction(const char* model_path) {
    fprintf(stderr, "\n=== Testing Slot Text Extraction ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 8192);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens;
    tokens.push_back(hp.audio_start_token);
    tokens.push_back(hp.audio_end_token);
    
    ErrorInfo error;
    
    int slot = batch_add_sequence(state, tokens, nullptr, 0, hp.hidden_size, -1, 16, "chinese", &error);
    if (slot < 0) {
        fprintf(stderr, "FAIL: Failed to add sequence: %s\n", error.message.c_str());
        free_batch(state);
        return false;
    }
    
    for (int step = 0; step < 5 && batch_get_n_active(state) > 0; ++step) {
        if (!batch_decode_step(state, &error)) {
            fprintf(stderr, "FAIL: Decode step %d failed: %s\n", step, error.message.c_str());
            free_batch(state);
            return false;
        }
        
        std::vector<int> eos_slots = batch_get_eos_slots(state);
        for (int s : eos_slots) {
            batch_remove_sequence(state, s);
        }
    }
    
    std::vector<int> gen_tokens = batch_get_tokens(state, slot);
    std::string text = batch_get_text(state, slot);
    
    fprintf(stderr, "Generated %zu tokens\n", gen_tokens.size());
    fprintf(stderr, "Text: \"%s\"\n", text.c_str());
    
    batch_clear(state);
    free_batch(state);
    
    fprintf(stderr, "=== Text Extraction Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_build_token_sequence(const char* model_path) {
    fprintf(stderr, "\n=== Testing Build Token Sequence ===\n");
    bool passed = true;
    
    Config config;
    config.model_path = model_path;
    config.n_threads = 4;
    
    BatchDecodeState* state = init_batch(config, 4, 8192);
    if (!state) {
        fprintf(stderr, "FAIL: Failed to init batch state\n");
        return false;
    }
    
    HyperParams hp = batch_get_hparams(state);
    
    std::vector<int> tokens = batch_build_token_sequence(state, 10, "chinese", "", "", "");
    
    fprintf(stderr, "Built token sequence with %zu tokens\n", tokens.size());
    
    if (tokens.empty()) {
        fprintf(stderr, "FAIL: Empty token sequence\n");
        passed = false;
    } else {
        fprintf(stderr, "First 10 tokens: ");
        for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); ++i) {
            fprintf(stderr, "%d ", tokens[i]);
        }
        fprintf(stderr, "\n");
        
        bool found_audio_start = false;
        bool found_audio_end = false;
        int audio_pad_count = 0;
        
        for (int t : tokens) {
            if (t == hp.audio_start_token) found_audio_start = true;
            if (t == hp.audio_end_token) found_audio_end = true;
            if (t == hp.audio_pad_token) audio_pad_count++;
        }
        
        if (!found_audio_start) {
            fprintf(stderr, "FAIL: Missing audio_start_token\n");
            passed = false;
        }
        if (!found_audio_end) {
            fprintf(stderr, "FAIL: Missing audio_end_token\n");
            passed = false;
        }
        if (audio_pad_count != 10) {
            fprintf(stderr, "FAIL: Expected 10 audio_pad tokens, got %d\n", audio_pad_count);
            passed = false;
        }
    }
    
    free_batch(state);
    
    fprintf(stderr, "=== Build Token Sequence Tests %s ===\n", passed ? "PASSED" : "FAILED");
    return passed;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    
    fprintf(stderr, "Running batch decode tests with model: %s\n", model_path);
    
    bool all_passed = true;
    
    all_passed &= test_batch_init_free(model_path);
    
    all_passed &= test_single_sequence_add_decode(model_path);
    
    all_passed &= test_two_sequences_different_lengths(model_path);
    
    all_passed &= test_sequence_finish_before_others(model_path);
    
    all_passed &= test_add_new_sequence_while_active(model_path);
    
    all_passed &= test_slot_text_extraction(model_path);
    
    all_passed &= test_build_token_sequence(model_path);
    
    fprintf(stderr, "\n=== All Batch Decode Tests %s ===\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}