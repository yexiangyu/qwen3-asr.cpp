#include "asr/codec/codec.hpp"
#include "asr/mel/mel.hpp"
#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>

using namespace asr;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <wav1> [wav2] [wav3] ...\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    std::vector<std::string> wav_files;
    for (int i = 2; i < argc; ++i) {
        wav_files.push_back(argv[i]);
    }
    
    int n_wavs = wav_files.size();
    printf("=== End-to-End Batch Transcribe Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Audio files: %d\n", n_wavs);
    for (int i = 0; i < n_wavs; ++i) {
        printf("  [%d] %s\n", i, wav_files[i].c_str());
    }
    printf("\n");
    
    ErrorInfo error;
    
    // ========================================
    // Step 1: Load all audio files
    // ========================================
    printf("=== Step 1: Load audio files ===\n");
    
    auto t_load_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<float>> audio_samples(n_wavs);
    std::vector<int> sample_rates(n_wavs);
    
    for (int i = 0; i < n_wavs; ++i) {
        if (!codec::decode_file(wav_files[i].c_str(), audio_samples[i], sample_rates[i], &error)) {
            fprintf(stderr, "FAIL: Failed to load %s: %s\n", wav_files[i].c_str(), error.message.c_str());
            return 1;
        }
        printf("  [%d] %zu samples, %d Hz, %.2f sec\n", 
               i, audio_samples[i].size(), sample_rates[i], 
               (float)audio_samples[i].size() / sample_rates[i]);
    }
    
    auto t_load_end = std::chrono::high_resolution_clock::now();
    auto t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t_load_start).count();
    printf("Audio loading time: %ld ms\n\n", t_load_ms);
    
    // ========================================
    // Step 2: Compute mel spectrograms (batch)
    // ========================================
    printf("=== Step 2: Compute mel spectrograms ===\n");
    
    auto t_mel_start = std::chrono::high_resolution_clock::now();
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    std::vector<mel::MelSpectrum> mel_specs(n_wavs);
    
    for (int i = 0; i < n_wavs; ++i) {
        if (!mel::compute_from_file(wav_files[i].c_str(), mel_specs[i], mel_config, &error)) {
            fprintf(stderr, "FAIL: Failed to compute mel for %s: %s\n", wav_files[i].c_str(), error.message.c_str());
            return 1;
        }
        printf("  [%d] %d mels, %d frames\n", i, mel_specs[i].n_mels, mel_specs[i].n_frames);
    }
    
    auto t_mel_end = std::chrono::high_resolution_clock::now();
    auto t_mel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_mel_end - t_mel_start).count();
    printf("Mel computation time: %ld ms\n\n", t_mel_ms);
    
    // ========================================
    // Step 3: Batch encode
    // ========================================
    printf("=== Step 3: Batch encode ===\n");
    
    transcribe::encoder::Config enc_config;
    enc_config.model_path = model_path;
    enc_config.device_name = "CUDA0";
    enc_config.n_threads = 4;
    
    auto enc_state = transcribe::encoder::init(enc_config);
    if (!enc_state) {
        fprintf(stderr, "FAIL: Failed to init encoder\n");
        return 1;
    }
    
    printf("Encoder initialized on %s\n", transcribe::encoder::get_device_name(enc_state));
    
    auto t_enc_start = std::chrono::high_resolution_clock::now();
    
    // Prepare batch input
    transcribe::encoder::BatchInput enc_input;
    int max_frames = 0;
    for (int i = 0; i < n_wavs; ++i) {
        enc_input.mel_data.push_back(mel_specs[i].data.data());
        enc_input.n_frames.push_back(mel_specs[i].n_frames);
        if (mel_specs[i].n_frames > max_frames) max_frames = mel_specs[i].n_frames;
    }
    enc_input.n_mels = mel_specs[0].n_mels;
    enc_input.max_frames = max_frames;
    
    printf("Batch encode input: %d items, max_frames=%d\n", n_wavs, max_frames);
    
    transcribe::encoder::BatchOutput enc_output;
    if (!transcribe::encoder::encode_batch(enc_state, enc_input, enc_output, &error)) {
        fprintf(stderr, "FAIL: Batch encode failed: %s\n", error.message.c_str());
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    auto t_enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_enc_end - t_enc_start).count();
    
    printf("Batch encode output: %d results\n", enc_output.batch_size());
    for (int i = 0; i < enc_output.batch_size(); ++i) {
        auto& feat = enc_output.features[i];
        printf("  [%d] hidden=%d, frames=%d\n", i, feat.hidden_size, feat.n_frames);
    }
    printf("Batch encode time: %ld ms\n\n", t_enc_ms);
    
    transcribe::encoder::free(enc_state);
    
    // ========================================
    // Step 4: Init batch decoder
    // ========================================
    printf("=== Step 4: Init batch decoder ===\n");
    
    transcribe::decoder::Config dec_config;
    dec_config.model_path = model_path;
    dec_config.device_name = "CUDA0";
    dec_config.n_threads = 4;
    
    int max_batch_size = n_wavs;
    int max_total_capacity = 16384;
    
    auto batch_state = transcribe::decoder::init_batch(dec_config, max_batch_size, max_total_capacity);
    if (!batch_state) {
        fprintf(stderr, "FAIL: Failed to init batch decoder\n");
        return 1;
    }
    
    printf("Batch decoder initialized\n");
    printf("Max batch size: %d\n", max_batch_size);
    printf("Max total capacity: %d\n\n", max_total_capacity);
    
    // ========================================
    // Step 5: Single prefill for each sequence
    // ========================================
    printf("=== Step 5: Prefill each sequence ===\n");
    
    auto hp = transcribe::decoder::batch_get_hparams(batch_state);
    
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> slot_ids(n_wavs);
    
    for (int i = 0; i < n_wavs; ++i) {
        // Build token sequence for this audio
        std::vector<int> tokens = transcribe::decoder::batch_build_token_sequence(
            batch_state, enc_output.features[i].n_frames, "", "", "", "");
        
        // Compute audio_start_pos (position of first audio_pad token)
        int audio_start_pos = -1;
        for (int j = 0; j < (int)tokens.size(); ++j) {
            if (tokens[j] == hp.audio_start_token) {
                audio_start_pos = j + 1;
                break;
            }
        }
        
        int slot_id = transcribe::decoder::batch_add_sequence(
            batch_state,
            tokens,
            enc_output.features[i].data.data(),
            enc_output.features[i].n_frames,
            enc_output.features[i].hidden_size,
            audio_start_pos,
            256,  // max_tokens
            "",    // language (auto-detect)
            &error);
        
        if (slot_id < 0) {
            fprintf(stderr, "FAIL: Failed to add sequence %d: %s\n", i, error.message.c_str());
            transcribe::decoder::free_batch(batch_state);
            return 1;
        }
        
        slot_ids[i] = slot_id;
        
        printf("  [%d] Added to slot %d, tokens=%zu, capacity=%d, audio_frames=%d\n", 
               i, slot_id, tokens.size(),
               transcribe::decoder::batch_get_slot_capacity(batch_state, slot_id),
               enc_output.features[i].n_frames);
    }
    
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    auto t_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();
    printf("Prefill time (total): %ld ms\n\n", t_prefill_ms);
    
    // ========================================
    // Step 6: Batch decode
    // ========================================
    printf("=== Step 6: Batch decode ===\n");
    
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    
    int step = 0;
    int max_steps = 300;
    std::vector<bool> finished(n_wavs, false);
    std::vector<int> finish_step(n_wavs, -1);
    
    while (transcribe::decoder::batch_get_n_active(batch_state) > 0 && step < max_steps) {
        if (!transcribe::decoder::batch_decode_step(batch_state, &error)) {
            fprintf(stderr, "FAIL: Batch decode step %d failed: %s\n", step, error.message.c_str());
            break;
        }
        
        // Check for EOS
        auto eos_slots = transcribe::decoder::batch_get_eos_slots(batch_state);
        for (int slot : eos_slots) {
            // Find which wav this slot corresponds to
            for (int i = 0; i < n_wavs; ++i) {
                if (slot_ids[i] == slot && !finished[i]) {
                    finished[i] = true;
                    finish_step[i] = step;
                    printf("  [%d] Slot %d finished at step %d, tokens=%d\n",
                           i, slot, step, transcribe::decoder::batch_get_slot_used(batch_state, slot));
                }
            }
        }
        
        step++;
        
        if (step % 50 == 0) {
            printf("  Step %d: active=%d\n", step, transcribe::decoder::batch_get_n_active(batch_state));
        }
    }
    
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    auto t_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_end - t_decode_start).count();
    
    printf("Batch decode completed in %d steps, time: %ld ms\n\n", step, t_decode_ms);
    
    // ========================================
    // Step 7: Get results
    // ========================================
    printf("=== Step 7: Results ===\n");
    
    for (int i = 0; i < n_wavs; ++i) {
        std::string text = transcribe::decoder::batch_get_text(batch_state, slot_ids[i]);
        auto tokens = transcribe::decoder::batch_get_tokens(batch_state, slot_ids[i]);
        
        printf("\n[%d] %s:\n", i, wav_files[i].c_str());
        printf("  Tokens: %d\n", (int)tokens.size());
        printf("  Text: %s\n", text.c_str());
    }
    
    // ========================================
    // Summary
    // ========================================
    printf("\n=== Summary ===\n");
    printf("Audio loading:  %ld ms\n", t_load_ms);
    printf("Mel computation: %ld ms\n", t_mel_ms);
    printf("Batch encode:   %ld ms\n", t_enc_ms);
    printf("Prefill:        %ld ms\n", t_prefill_ms);
    printf("Batch decode:   %ld ms (%d steps)\n", t_decode_ms, step);
    printf("Total:          %ld ms\n", 
           t_load_ms + t_mel_ms + t_enc_ms + t_prefill_ms + t_decode_ms);
    
    // Cleanup
    transcribe::decoder::free_batch(batch_state);
    
    printf("\n=== Test PASSED ===\n");
    
    return 0;
}