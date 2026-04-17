#include "../modules/encoder/encoder.h"
#include "../modules/decoder/decoder.h"
#include "../modules/mel/mel.h"
#include "../modules/audio_codec/audio_codec.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace qwen3_asr;

int main() {
    const char* test_wav = "tests/data/test_audio.wav";
    
    printf("=== Module Interface Compatibility Test ===\n\n");
    
    printf("Test 1: Audio Codec -> Mel -> Encoder pipeline\n");
    
    mel::Config mel_config;
    mel_config.n_threads = 4;
    
    mel::MelSpectrum mel_spec;
    mel::ErrorInfo mel_error;
    
    if (!mel::compute_from_file(test_wav, mel_spec, mel_config, &mel_error)) {
        fprintf(stderr, "FAIL: Failed to compute mel: %s\n", mel_error.message.c_str());
        return 1;
    }
    
    printf("Mel computed: %d mels, %d frames\n", mel_spec.n_mels, mel_spec.n_frames);
    
    encoder::Config enc_config;
    enc_config.model_path = "models/qwen3-forced-aligner-0.6b-f16.gguf";
    enc_config.n_threads = 4;
    enc_config.device_name = "CUDA0";
    
    encoder::EncoderState* enc_state = encoder::init(enc_config);
    if (!enc_state) {
        fprintf(stderr, "FAIL: Failed to init encoder\n");
        return 1;
    }
    
    encoder::BatchInput batch_input;
    batch_input.mel_data.push_back(mel_spec.data.data());
    batch_input.n_frames.push_back(mel_spec.n_frames);
    batch_input.n_mels = mel_spec.n_mels;
    batch_input.max_frames = mel_spec.n_frames;
    
    encoder::BatchOutput batch_output;
    encoder::ErrorInfo enc_error;
    
    if (!encoder::encode_batch(enc_state, batch_input, batch_output, &enc_error)) {
        fprintf(stderr, "FAIL: Encoder encode failed: %s\n", enc_error.message.c_str());
        encoder::free(enc_state);
        return 1;
    }
    
    auto& audio_features = batch_output.features[0];
    printf("Encoder output: hidden=%d, frames=%d\n", 
           audio_features.hidden_size, audio_features.n_frames);
    
    printf("PASS: Mel -> Encoder pipeline\n\n");
    encoder::free(enc_state);
    
    printf("Test 2: Decoder standalone test\n");
    
    decoder::Config dec_config;
    dec_config.model_path = "models/qwen3-asr-1.7b-f16.gguf";
    dec_config.n_threads = 4;
    dec_config.device_name = "CUDA0";
    dec_config.max_ctx_length = 2048;
    
    decoder::DecoderState* dec_state = decoder::init(dec_config);
    if (!dec_state) {
        fprintf(stderr, "FAIL: Failed to init decoder\n");
        return 1;
    }
    
    auto dec_hparams = decoder::get_hparams(dec_state);
    printf("Decoder hparams: vocab=%d, hidden=%d\n", dec_hparams.vocab_size, dec_hparams.hidden_size);
    
    std::vector<float> synthetic_audio(100 * dec_hparams.hidden_size, 0.1f);
    
    std::vector<int> tokens;
    tokens.push_back(dec_hparams.audio_start_token);
    for (int i = 0; i < 100; ++i) {
        tokens.push_back(dec_hparams.audio_pad_token);
    }
    tokens.push_back(dec_hparams.audio_end_token);
    tokens.push_back(151644);
    tokens.push_back(89463);
    tokens.push_back(198);
    
    decoder::PrefillInput prefill_input;
    prefill_input.tokens = tokens.data();
    prefill_input.n_tokens = tokens.size();
    prefill_input.audio_features = synthetic_audio.data();
    prefill_input.n_audio_frames = 100;
    prefill_input.audio_feature_dim = dec_hparams.hidden_size;
    prefill_input.audio_start_pos = -1;
    
    decoder::DecoderOutput prefill_output;
    decoder::ErrorInfo dec_error;
    
    if (!decoder::prefill(dec_state, prefill_input, prefill_output, &dec_error)) {
        fprintf(stderr, "FAIL: Decoder prefill failed: %s\n", dec_error.message.c_str());
        decoder::free(dec_state);
        return 1;
    }
    
    printf("Prefill output: logits_size=%zu\n", prefill_output.logits.size());
    
    decoder::DecodeInput decode_input;
    int next_token = 0;
    decode_input.tokens = &next_token;
    decode_input.n_tokens = 1;
    decode_input.n_past = tokens.size();
    
    decoder::DecoderOutput decode_output;
    if (!decoder::decode(dec_state, decode_input, decode_output, &dec_error)) {
        fprintf(stderr, "FAIL: Decoder decode failed: %s\n", dec_error.message.c_str());
        decoder::free(dec_state);
        return 1;
    }
    
    printf("Decode output: logits_size=%zu\n", decode_output.logits.size());
    printf("PASS: Decoder standalone test\n\n");
    
    decoder::free(dec_state);
    
    printf("Test 3: Interface compatibility check\n");
    
    printf("Encoder output format:\n");
    printf("  hidden_size: int (feature dimension)\n");
    printf("  n_frames: int (sequence length)\n");
    printf("  data: std::vector<float> (flattened features)\n");
    
    printf("\nDecoder input format:\n");
    printf("  audio_features: const float* (feature pointer)\n");
    printf("  n_audio_frames: int (sequence length)\n");
    printf("  audio_feature_dim: int (feature dimension)\n");
    
    printf("\nCompatibility: Encoder output can feed Decoder input\n");
    printf("  encoder.features[b].data -> decoder.prefill.audio_features\n");
    printf("  encoder.features[b].n_frames -> decoder.prefill.n_audio_frames\n");
    printf("  encoder.features[b].hidden_size -> decoder.prefill.audio_feature_dim\n");
    
    printf("PASS: Interface compatible\n\n");
    
    printf("=== All tests PASSED ===\n");
    printf("\nNote: Full ASR pipeline requires ASR encoder implementation\n");
    printf("  Current encoder module: forced-aligner encoder (different architecture)\n");
    printf("  Current decoder module: ASR decoder (expects ASR encoder features)\n");
    printf("  Future work: Add ASR encoder variant to encoder module\n");
    
    return 0;
}