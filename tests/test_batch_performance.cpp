#include "asr/codec/codec.hpp"
#include "asr/mel/mel.hpp"
#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

using namespace asr;

// Split audio into two halves
void split_audio(const std::vector<float>& full_audio, int sample_rate,
                 std::vector<float>& audio_a, std::vector<float>& audio_b) {
    int half = full_audio.size() / 2;
    audio_a.assign(full_audio.begin(), full_audio.begin() + half);
    audio_b.assign(full_audio.begin() + half, full_audio.end());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [audio.wav]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* audio_path = (argc >= 3) ? argv[2] : "tests/data/test_audio.wav";
    
    printf("=== Batch Decode Performance Comparison ===\n");
    printf("Model: %s\n", model_path);
    printf("Audio: %s\n\n", audio_path);
    
    ErrorInfo error;
    
    // Load full audio
    std::vector<float> full_audio;
    int sample_rate;
    if (!codec::decode_file(audio_path, full_audio, sample_rate, &error)) {
        fprintf(stderr, "FAIL: Failed to load audio: %s\n", error.message.c_str());
        return 1;
    }
    printf("Full audio: %zu samples, %d Hz, %.2f sec\n", 
           full_audio.size(), sample_rate, (float)full_audio.size() / sample_rate);
    
    // Split into two halves
    std::vector<float> audio_a, audio_b;
    split_audio(full_audio, sample_rate, audio_a, audio_b);
    printf("Audio A: %zu samples, %.2f sec\n", audio_a.size(), (float)audio_a.size() / sample_rate);
    printf("Audio B: %zu samples, %.2f sec\n\n", audio_b.size(), (float)audio_b.size() / sample_rate);
    
    // Save split audio to temp files
    std::string temp_a = "/tmp/audio_a.raw";
    std::string temp_b = "/tmp/audio_b.raw";
    
    // Write raw PCM (16-bit signed)
    {
        std::ofstream f(temp_a, std::ios::binary);
        for (float v : audio_a) {
            int16_t s = (int16_t)(v * 32767);
            f.write((const char*)&s, 2);
        }
    }
    {
        std::ofstream f(temp_b, std::ios::binary);
        for (float v : audio_b) {
            int16_t s = (int16_t)(v * 32767);
            f.write((const char*)&s, 2);
        }
    }
    
    // Compute mel for both parts
    printf("=== Computing mel spectrograms ===\n");
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    // Use codec::decode_file for mel (need WAV, so use original file with offsets)
    // Instead, compute mel directly from samples
    mel::MelSpectrum mel_a, mel_b;
    
    // Compute mel from samples (we need to add this function)
    // For now, use WAV file segments
    // Let's create WAV headers for the temp files
    // Actually, simpler: use the codec::load_wav and then compute
    
    // Simpler approach: use codec::decode_file with original file, then manually compute mel
    // We already have audio_a and audio_b samples
    
    // We need to compute mel from raw samples - use the mel module directly
    // Check if mel::compute_from_samples exists
    
    // For simplicity, let's use the original file twice
    // Actually, let's just use two different segments of the same file by computing mel for full file
    // and then slicing
    
    mel::MelSpectrum mel_full;
    if (!mel::compute_from_file(audio_path, mel_full, mel_config, &error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", error.message.c_str());
        return 1;
    }
    printf("Full mel: %d mels, %d frames\n\n", mel_full.n_mels, mel_full.n_frames);
    
    // Split mel into two halves
    int mel_half = mel_full.n_frames / 2;
    
    mel_a.n_mels = mel_full.n_mels;
    mel_a.n_frames = mel_half;
    mel_a.data.resize(mel_full.n_mels * mel_half);
    for (int m = 0; m < mel_full.n_mels; ++m) {
        for (int f = 0; f < mel_half; ++f) {
            mel_a.data[m * mel_half + f] = mel_full.data[m * mel_full.n_frames + f];
        }
    }
    
    mel_b.n_mels = mel_full.n_mels;
    mel_b.n_frames = mel_full.n_frames - mel_half;
    mel_b.data.resize(mel_full.n_mels * mel_b.n_frames);
    for (int m = 0; m < mel_full.n_mels; ++m) {
        for (int f = 0; f < mel_b.n_frames; ++f) {
            mel_b.data[m * mel_b.n_frames + f] = mel_full.data[m * mel_full.n_frames + mel_half + f];
        }
    }
    
    printf("Mel A: %d mels, %d frames\n", mel_a.n_mels, mel_a.n_frames);
    printf("Mel B: %d mels, %d frames\n\n", mel_b.n_mels, mel_b.n_frames);
    
    // ========================================
    // Test 1: Non-batch (process separately)
    // ========================================
    printf("=== Test 1: Non-batch (separate processing) ===\n");
    
    transcribe::encoder::Config enc_config;
    enc_config.model_path = model_path;
    enc_config.device_name = "CUDA0";
    enc_config.n_threads = 4;
    
    transcribe::decoder::Config dec_config;
    dec_config.model_path = model_path;
    dec_config.device_name = "CUDA0";
    dec_config.n_threads = 4;
    
    auto t_nonbatch_start = std::chrono::high_resolution_clock::now();
    
    // Process A
    auto enc_state_a = transcribe::encoder::init(enc_config);
    auto dec_state_a = transcribe::decoder::init(dec_config);
    
    transcribe::encoder::BatchInput enc_in_a;
    enc_in_a.mel_data.push_back(mel_a.data.data());
    enc_in_a.n_frames.push_back(mel_a.n_frames);
    enc_in_a.n_mels = mel_a.n_mels;
    enc_in_a.max_frames = mel_a.n_frames;
    
    transcribe::encoder::BatchOutput enc_out_a;
    transcribe::encoder::encode_batch(enc_state_a, enc_in_a, enc_out_a, &error);
    
    transcribe::decoder::TranscribeInput trans_in_a;
    trans_in_a.audio_features = enc_out_a.features[0].data.data();
    trans_in_a.n_audio_frames = enc_out_a.features[0].n_frames;
    trans_in_a.audio_feature_dim = enc_out_a.features[0].hidden_size;
    trans_in_a.max_tokens = 256;
    
    transcribe::decoder::TranscribeOutput trans_out_a;
    transcribe::decoder::transcribe(dec_state_a, trans_in_a, trans_out_a, &error);
    
    transcribe::encoder::free(enc_state_a);
    transcribe::decoder::free(dec_state_a);
    
    // Process B
    auto enc_state_b = transcribe::encoder::init(enc_config);
    auto dec_state_b = transcribe::decoder::init(dec_config);
    
    transcribe::encoder::BatchInput enc_in_b;
    enc_in_b.mel_data.push_back(mel_b.data.data());
    enc_in_b.n_frames.push_back(mel_b.n_frames);
    enc_in_b.n_mels = mel_b.n_mels;
    enc_in_b.max_frames = mel_b.n_frames;
    
    transcribe::encoder::BatchOutput enc_out_b;
    transcribe::encoder::encode_batch(enc_state_b, enc_in_b, enc_out_b, &error);
    
    transcribe::decoder::TranscribeInput trans_in_b;
    trans_in_b.audio_features = enc_out_b.features[0].data.data();
    trans_in_b.n_audio_frames = enc_out_b.features[0].n_frames;
    trans_in_b.audio_feature_dim = enc_out_b.features[0].hidden_size;
    trans_in_b.max_tokens = 256;
    
    transcribe::decoder::TranscribeOutput trans_out_b;
    transcribe::decoder::transcribe(dec_state_b, trans_in_b, trans_out_b, &error);
    
    transcribe::encoder::free(enc_state_b);
    transcribe::decoder::free(dec_state_b);
    
    auto t_nonbatch_end = std::chrono::high_resolution_clock::now();
    auto t_nonbatch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_nonbatch_end - t_nonbatch_start).count();
    
    printf("Audio A: %d tokens, text: %s\n", trans_out_a.n_tokens, trans_out_a.text.substr(0, 50).c_str());
    printf("Audio B: %d tokens, text: %s\n", trans_out_b.n_tokens, trans_out_b.text.substr(0, 50).c_str());
    printf("Non-batch total time: %ld ms\n\n", t_nonbatch_ms);
    
    // ========================================
    // Test 2: Batch (batch_size=2)
    // ========================================
    printf("=== Test 2: Batch (batch_size=2) ===\n");
    
    auto t_batch_start = std::chrono::high_resolution_clock::now();
    
    // Batch encode
    auto enc_state = transcribe::encoder::init(enc_config);
    
    transcribe::encoder::BatchInput enc_input;
    enc_input.mel_data.push_back(mel_a.data.data());
    enc_input.mel_data.push_back(mel_b.data.data());
    enc_input.n_frames.push_back(mel_a.n_frames);
    enc_input.n_frames.push_back(mel_b.n_frames);
    enc_input.n_mels = mel_a.n_mels;
    enc_input.max_frames = std::max(mel_a.n_frames, mel_b.n_frames);
    
    transcribe::encoder::BatchOutput enc_output;
    transcribe::encoder::encode_batch(enc_state, enc_input, enc_output, &error);
    
    transcribe::encoder::free(enc_state);
    
    // Batch decode
    auto batch_state = transcribe::decoder::init_batch(dec_config, 2, 16384);
    
    auto hp = transcribe::decoder::batch_get_hparams(batch_state);
    
    // Add sequence A
    std::vector<int> tokens_a = transcribe::decoder::batch_build_token_sequence(
        batch_state, enc_output.features[0].n_frames, "", "", "", "");
    int audio_start_pos_a = -1;
    for (int j = 0; j < (int)tokens_a.size(); ++j) {
        if (tokens_a[j] == hp.audio_start_token) {
            audio_start_pos_a = j + 1;
            break;
        }
    }
    
    int slot_a = transcribe::decoder::batch_add_sequence(
        batch_state, tokens_a,
        enc_output.features[0].data.data(),
        enc_output.features[0].n_frames,
        enc_output.features[0].hidden_size,
        audio_start_pos_a, 256, "", &error);
    
    // Add sequence B
    std::vector<int> tokens_b = transcribe::decoder::batch_build_token_sequence(
        batch_state, enc_output.features[1].n_frames, "", "", "", "");
    int audio_start_pos_b = -1;
    for (int j = 0; j < (int)tokens_b.size(); ++j) {
        if (tokens_b[j] == hp.audio_start_token) {
            audio_start_pos_b = j + 1;
            break;
        }
    }
    
    int slot_b = transcribe::decoder::batch_add_sequence(
        batch_state, tokens_b,
        enc_output.features[1].data.data(),
        enc_output.features[1].n_frames,
        enc_output.features[1].hidden_size,
        audio_start_pos_b, 256, "", &error);
    
    // Batch decode loop
    int step = 0;
    while (transcribe::decoder::batch_get_n_active(batch_state) > 0 && step < 300) {
        transcribe::decoder::batch_decode_step(batch_state, &error);
        
        auto eos_slots = transcribe::decoder::batch_get_eos_slots(batch_state);
        (void)eos_slots; // Don't remove immediately, just note
        step++;
    }
    
    auto t_batch_end = std::chrono::high_resolution_clock::now();
    auto t_batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_batch_end - t_batch_start).count();
    
    std::string text_a = transcribe::decoder::batch_get_text(batch_state, slot_a);
    std::string text_b = transcribe::decoder::batch_get_text(batch_state, slot_b);
    auto gen_tokens_a = transcribe::decoder::batch_get_tokens(batch_state, slot_a);
    auto gen_tokens_b = transcribe::decoder::batch_get_tokens(batch_state, slot_b);
    
    printf("Audio A: %zu tokens, text: %s\n", gen_tokens_a.size(), text_a.substr(0, 50).c_str());
    printf("Audio B: %zu tokens, text: %s\n", gen_tokens_b.size(), text_b.substr(0, 50).c_str());
    printf("Batch total time: %ld ms (%d steps)\n\n", t_batch_ms, step);
    
    transcribe::decoder::free_batch(batch_state);
    
    // ========================================
    // Summary
    // ========================================
    printf("=== Summary ===\n");
    printf("Audio A duration: %.2f sec\n", (float)audio_a.size() / sample_rate);
    printf("Audio B duration: %.2f sec\n", (float)audio_b.size() / sample_rate);
    printf("Total audio: %.2f sec\n\n", (float)full_audio.size() / sample_rate);
    
    printf("Non-batch time: %ld ms\n", t_nonbatch_ms);
    printf("Batch time:     %ld ms\n", t_batch_ms);
    printf("Speedup:        %.2fx\n\n", (float)t_nonbatch_ms / t_batch_ms);
    
    printf("Time per second of audio:\n");
    printf("  Non-batch: %.1f ms/sec\n", (float)t_nonbatch_ms / ((float)full_audio.size() / sample_rate));
    printf("  Batch:     %.1f ms/sec\n", (float)t_batch_ms / ((float)full_audio.size() / sample_rate));
    
    // Verify outputs match
    bool match_a = (trans_out_a.text == text_a);
    bool match_b = (trans_out_b.text == text_b);
    printf("\nOutput verification:\n");
    printf("  Audio A: %s\n", match_a ? "MATCH" : "DIFFER");
    printf("  Audio B: %s\n", match_b ? "MATCH" : "DIFFER");
    
    return 0;
}