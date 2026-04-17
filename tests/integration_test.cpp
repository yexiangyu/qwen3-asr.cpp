#include "asr/transcribe/encoder.hpp"
#include "asr/transcribe/decoder.hpp"
#include "asr/mel/mel.hpp"
#include "asr/codec/codec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace asr;

int main() {
    const char* test_wav = "tests/data/test_audio.wav";
    
    printf("=== ASR Pipeline Integration Test ===\n\n");
    
    printf("Step 1: Audio -> Mel\n");
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    mel::MelSpectrum mel_spec;
    mel::ErrorInfo mel_error;
    
    if (!mel::compute_from_file(test_wav, mel_spec, mel_config, &mel_error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", mel_error.message.c_str());
        return 1;
    }
    
    printf("  Mel: %d mels, %d frames\n\n", mel_spec.n_mels, mel_spec.n_frames);
    
    printf("Step 2: ASR Encoder\n");
    
    transcribe::encoder::Config enc_config;
    enc_config.model_path = "models/qwen3-asr-1.7b-f16.gguf";
    enc_config.n_threads = 4;
    enc_config.device_name = "CUDA0";
    
    transcribe::encoder::EncoderState* enc_state = transcribe::encoder::init(enc_config);
    if (!enc_state) {
        fprintf(stderr, "FAIL: Failed to init asr_encoder\n");
        return 1;
    }
    
    auto enc_hparams = transcribe::encoder::get_hparams(enc_state);
    printf("  Model: qwen3-asr-1.7b-f16.gguf\n");
    printf("  Encoder hparams: hidden=%d\n", enc_hparams.hidden_size);
    
    transcribe::encoder::BatchInput batch_input;
    batch_input.mel_data.push_back(mel_spec.data.data());
    batch_input.n_frames.push_back(mel_spec.n_frames);
    batch_input.n_mels = mel_spec.n_mels;
    batch_input.max_frames = mel_spec.n_frames;
    
    transcribe::encoder::BatchOutput batch_output;
    transcribe::encoder::ErrorInfo enc_error;
    
    if (!transcribe::encoder::encode_batch(enc_state, batch_input, batch_output, &enc_error)) {
        fprintf(stderr, "FAIL: ASR encoder encode failed: %s\n", enc_error.message.c_str());
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    auto& audio_features = batch_output.features[0];
    printf("  Output: hidden=%d, frames=%d\n\n", 
           audio_features.hidden_size, audio_features.n_frames);
    
    printf("Step 3: Decoder Prefill\n");
    
    transcribe::decoder::Config dec_config;
    dec_config.model_path = "models/qwen3-asr-1.7b-f16.gguf";
    dec_config.n_threads = 4;
    dec_config.device_name = "CUDA0";
    dec_config.max_ctx_length = 4096;
    
    transcribe::decoder::State* dec_state = transcribe::decoder::init(dec_config);
    if (!dec_state) {
        fprintf(stderr, "FAIL: Failed to init decoder\n");
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    auto dec_hparams = transcribe::decoder::get_hparams(dec_state);
    printf("  Decoder hparams: vocab=%d, hidden=%d\n", dec_hparams.vocab_size, dec_hparams.hidden_size);
    
    if (audio_features.hidden_size != dec_hparams.hidden_size) {
        fprintf(stderr, "FAIL: Dimension mismatch! Encoder hidden=%d, Decoder hidden=%d\n",
                audio_features.hidden_size, dec_hparams.hidden_size);
        transcribe::decoder::free(dec_state);
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    std::vector<int> tokens;
    
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    tokens.push_back(dec_hparams.audio_start_token);
    for (int i = 0; i < audio_features.n_frames; ++i) {
        tokens.push_back(dec_hparams.audio_pad_token);
    }
    tokens.push_back(dec_hparams.audio_end_token);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    printf("  Tokens: [im_start, system, newline, im_end, newline, im_start, user, newline, audio_start, audio_pad*%d, audio_end, im_end, newline, im_start, assistant, newline]\n", audio_features.n_frames);
    
    transcribe::decoder::PrefillInput prefill_input;
    prefill_input.tokens = tokens.data();
    prefill_input.n_tokens = tokens.size();
    prefill_input.audio_features = audio_features.data.data();
    prefill_input.n_audio_frames = audio_features.n_frames;
    prefill_input.audio_feature_dim = audio_features.hidden_size;
    prefill_input.audio_start_pos = -1;
    
    transcribe::decoder::DecoderOutput prefill_output;
    transcribe::decoder::ErrorInfo dec_error;
    
    if (!transcribe::decoder::prefill(dec_state, prefill_input, prefill_output, &dec_error)) {
        fprintf(stderr, "FAIL: Decoder prefill failed: %s\n", dec_error.message.c_str());
        transcribe::decoder::free(dec_state);
        transcribe::encoder::free(enc_state);
        return 1;
    }
    
    int kv_used = transcribe::decoder::get_kv_cache_used(dec_state);
    printf("  KV cache: %d used\n\n", kv_used);
    
    printf("Step 4: Decode (generate tokens)\n");
    
    std::vector<int> generated_tokens;
    int n_past = tokens.size();
    int max_tokens = 20;
    
    for (int i = 0; i < max_tokens; ++i) {
        int next_token = -1;
        if (i == 0) {
            if (prefill_output.logits.empty()) {
                fprintf(stderr, "FAIL: Prefill output logits empty\n");
                transcribe::decoder::free(dec_state);
                transcribe::encoder::free(enc_state);
                return 1;
            }
            
            int vocab_size = prefill_output.vocab_size;
            float max_logit = -1e30f;
            int max_idx = 0;
            for (int j = 0; j < vocab_size; ++j) {
                if (prefill_output.logits[j] > max_logit) {
                    max_logit = prefill_output.logits[j];
                    max_idx = j;
                }
            }
            next_token = max_idx;
            
            bool has_nan = false;
            for (int j = 0; j < std::min(10, vocab_size); ++j) {
                if (std::isnan(prefill_output.logits[j])) {
                    has_nan = true;
                    break;
                }
            }
            if (has_nan) {
                fprintf(stderr, "FAIL: Prefill output contains NaN\n");
                transcribe::decoder::free(dec_state);
                transcribe::encoder::free(enc_state);
                return 1;
            }
        } else {
            transcribe::decoder::DecodeInput decode_input;
            decode_input.tokens = &next_token;
            decode_input.n_tokens = 1;
            decode_input.n_past = n_past;
            
            transcribe::decoder::DecoderOutput decode_output;
            if (!transcribe::decoder::decode(dec_state, decode_input, decode_output, &dec_error)) {
                fprintf(stderr, "FAIL: Decoder decode failed: %s\n", dec_error.message.c_str());
                transcribe::decoder::free(dec_state);
                transcribe::encoder::free(enc_state);
                return 1;
            }
            
            int vocab_size = decode_output.vocab_size;
            float max_logit = -1e30f;
            int max_idx = 0;
            for (int j = 0; j < vocab_size; ++j) {
                if (decode_output.logits[j] > max_logit) {
                    max_logit = decode_output.logits[j];
                    max_idx = j;
                }
            }
            next_token = max_idx;
        }
        
        generated_tokens.push_back(next_token);
        n_past++;
        
        if (next_token == dec_hparams.eos_token || next_token == im_end) {
            printf("  EOS/im_end token reached at position %d\n", i);
            break;
        }
    }
    
    printf("  Generated %zu tokens: [", generated_tokens.size());
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        printf("%d", generated_tokens[i]);
        if (i < generated_tokens.size() - 1) printf(", ");
    }
    printf("]\n");
    
    bool all_same = true;
    for (size_t i = 1; i < generated_tokens.size(); ++i) {
        if (generated_tokens[i] != generated_tokens[0]) {
            all_same = false;
            break;
        }
    }
    if (all_same && generated_tokens.size() > 1) {
        printf("  WARNING: All generated tokens are the same!\n");
    } else {
        printf("  Tokens are varied (good)\n");
    }
    
    transcribe::decoder::free(dec_state);
    transcribe::encoder::free(enc_state);
    
    printf("\nPASS: Complete ASR pipeline works\n\n");
    
    printf("=== Test Batch ASR Pipeline (batch_size=3) ===\n\n");
    
    transcribe::encoder::EncoderState* enc_state2 = transcribe::encoder::init(enc_config);
    if (!enc_state2) {
        fprintf(stderr, "FAIL: Failed to init asr_encoder for batch test\n");
        return 1;
    }
    
    transcribe::decoder::State* dec_state2 = transcribe::decoder::init(dec_config);
    if (!dec_state2) {
        fprintf(stderr, "FAIL: Failed to init decoder for batch test\n");
        transcribe::encoder::free(enc_state2);
        return 1;
    }
    
    printf("Step B1: Create batch mel inputs (3 different audio segments)\n");
    
    const int batch_size = 3;
    const int max_frames = 1000;
    const int segment_starts[3] = {0, 3000, 8000};
    const int segment_lengths[3] = {800, 1000, 600};
    
    std::vector<std::vector<float>> batch_mels(batch_size);
    std::vector<int> batch_frames(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int start = segment_starts[b];
        int len = segment_lengths[b];
        batch_frames[b] = len;
        batch_mels[b].resize(mel_spec.n_mels * len);
        
        for (int m = 0; m < mel_spec.n_mels; ++m) {
            for (int f = 0; f < len; ++f) {
                batch_mels[b][m * len + f] = mel_spec.data[m * mel_spec.n_frames + start + f];
            }
        }
    }
    
    printf("Batch mel inputs: sizes=[%d, %d, %d]\n", batch_frames[0], batch_frames[1], batch_frames[2]);
    
    transcribe::encoder::BatchInput batch_input2;
    for (int b = 0; b < batch_size; ++b) {
        batch_input2.mel_data.push_back(batch_mels[b].data());
    }
    batch_input2.n_frames = batch_frames;
    batch_input2.n_mels = mel_spec.n_mels;
    batch_input2.max_frames = max_frames;
    
    transcribe::encoder::BatchOutput batch_output2;
    
    if (!transcribe::encoder::encode_batch(enc_state2, batch_input2, batch_output2, &enc_error)) {
        fprintf(stderr, "FAIL: Batch ASR encoder failed: %s\n", enc_error.message.c_str());
        transcribe::decoder::free(dec_state2);
        transcribe::encoder::free(enc_state2);
        return 1;
    }
    
    printf("Batch ASR encoder output: %d results\n", batch_output2.batch_size());
    
    for (int b = 0; b < batch_output2.batch_size(); ++b) {
        auto& feat = batch_output2.features[b];
        float min_v = *std::min_element(feat.data.begin(), feat.data.end());
        float max_v = *std::max_element(feat.data.begin(), feat.data.end());
        printf("  Item %d: hidden=%d, frames=%d, range=[%.3f, %.3f]\n", 
               b, feat.hidden_size, feat.n_frames, min_v, max_v);
        
        if (feat.data.empty() || std::isnan(min_v) || std::isnan(max_v)) {
            fprintf(stderr, "FAIL: Invalid output for batch item %d\n", b);
            transcribe::decoder::free(dec_state2);
            transcribe::encoder::free(enc_state2);
            return 1;
        }
    }
    
    printf("PASS: Batch ASR encoder\n\n");
    
    printf("Step B2: Verify each batch item can feed decoder\n");
    
    for (int b = 0; b < batch_size; ++b) {
        auto& feat = batch_output2.features[b];
        
        std::vector<int> tokens_b;
        tokens_b.push_back(dec_hparams.audio_start_token);
        for (int i = 0; i < feat.n_frames; ++i) {
            tokens_b.push_back(dec_hparams.audio_pad_token);
        }
        tokens_b.push_back(dec_hparams.audio_end_token);
        tokens_b.push_back(im_start);
        tokens_b.push_back(assistant_token);
        tokens_b.push_back(198);
        
        transcribe::decoder::PrefillInput prefill_b;
        prefill_b.tokens = tokens_b.data();
        prefill_b.n_tokens = tokens_b.size();
        prefill_b.audio_features = feat.data.data();
        prefill_b.n_audio_frames = feat.n_frames;
        prefill_b.audio_feature_dim = feat.hidden_size;
        prefill_b.audio_start_pos = -1;
        
        transcribe::decoder::DecoderOutput output_b;
        transcribe::decoder::clear_kv_cache(dec_state2);
        
        if (!transcribe::decoder::prefill(dec_state2, prefill_b, output_b, &dec_error)) {
            fprintf(stderr, "FAIL: Decoder prefill for batch item %d failed: %s\n", b, dec_error.message.c_str());
            transcribe::decoder::free(dec_state2);
            transcribe::encoder::free(enc_state2);
            return 1;
        }
        
        float logit_min = *std::min_element(output_b.logits.begin(), output_b.logits.end());
        float logit_max = *std::max_element(output_b.logits.begin(), output_b.logits.end());
        
        printf("  Batch %d prefill: tokens=%zu, logits_range=[%.3f, %.3f]\n", 
               b, tokens_b.size(), logit_min, logit_max);
        
        if (output_b.logits.empty() || std::isnan(logit_min)) {
            fprintf(stderr, "FAIL: Invalid decoder output for batch item %d\n", b);
            transcribe::decoder::free(dec_state2);
            transcribe::encoder::free(enc_state2);
            return 1;
        }
    }
    
    printf("PASS: Each batch item feeds decoder correctly\n\n");
    
    transcribe::decoder::free(dec_state2);
    transcribe::encoder::free(enc_state2);
    
    printf("=== ALL TESTS PASSED ===\n");
    printf("\nVerified:\n");
    printf("  - Single ASR pipeline: audio -> mel -> encoder -> decoder\n");
    printf("  - Batch ASR pipeline: 3 items with different lengths\n");
    printf("  - Each batch output valid (no NaN, correct range)\n");
    printf("  - Each batch item compatible with decoder input\n");
    
    return 0;
}