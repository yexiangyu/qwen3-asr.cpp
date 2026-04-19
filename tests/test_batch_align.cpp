#include "asr/codec/codec.hpp"
#include "asr/mel/mel.hpp"
#include "asr/aligner/encoder.hpp"
#include "asr/aligner/decoder.hpp"

#include <cstdio>
#include <chrono>
#include <vector>

using namespace asr;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <aligner.gguf> <wav1> <wav2> [device]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* wav1 = argv[2];
    const char* wav2 = argv[3];
    const char* device = argc > 4 ? argv[4] : "CUDA0";
    const int n_wavs = 2;

    ErrorInfo error;

    printf("=== Batch Align Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Wav1: %s\n", wav1);
    printf("Wav2: %s\n", wav2);
    printf("Device: %s\n\n", device);

    // Step 1: Load audio
    printf("--- Step 1: Load audio ---\n");
    std::vector<std::vector<float>> audio(n_wavs);
    std::vector<int> sr(n_wavs);
    const char* wav_paths[] = { wav1, wav2 };

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_wavs; ++i) {
        if (!codec::decode_file(wav_paths[i], audio[i], sr[i], &error)) {
            fprintf(stderr, "FAIL: load %s: %s\n", wav_paths[i], error.message.c_str());
            return 1;
        }
        printf("  [%d] %zu samples, %d Hz, %.2f sec\n", i, audio[i].size(), sr[i], (float)audio[i].size() / sr[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Load time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    // Step 2: Compute mel spectrograms
    printf("--- Step 2: Compute mel ---\n");
    mel::Config mel_cfg;
    mel_cfg.n_threads = 4;
    std::vector<mel::MelSpectrum> mel_specs(n_wavs);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_wavs; ++i) {
        if (!mel::compute_from_file(wav_paths[i], mel_specs[i], mel_cfg, &error)) {
            fprintf(stderr, "FAIL: mel %s: %s\n", wav_paths[i], error.message.c_str());
            return 1;
        }
        printf("  [%d] %d mels, %d frames\n", i, mel_specs[i].n_mels, mel_specs[i].n_frames);
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("Mel time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    // Step 3: Init aligner encoder
    printf("--- Step 3: Init aligner encoder ---\n");
    aligner::encoder::Config enc_cfg;
    enc_cfg.model_path = model_path;
    enc_cfg.device_name = device;
    enc_cfg.n_threads = 4;

    auto enc_state = aligner::encoder::init(enc_cfg);
    if (!enc_state) { fprintf(stderr, "FAIL: encoder init\n"); return 1; }
    printf("Encoder on %s\n\n", aligner::encoder::get_device_name(enc_state));

    // Step 4: Batch encode (2 wavs simultaneously)
    printf("--- Step 4: Batch encode (2 wavs) ---\n");
    aligner::encoder::BatchInput enc_in;
    int max_frames = 0;
    for (int i = 0; i < n_wavs; ++i) {
        enc_in.mel_data.push_back(mel_specs[i].data.data());
        enc_in.n_frames.push_back(mel_specs[i].n_frames);
        if (mel_specs[i].n_frames > max_frames) max_frames = mel_specs[i].n_frames;
    }
    enc_in.n_mels = mel_specs[0].n_mels;
    enc_in.max_frames = max_frames;

    aligner::encoder::BatchOutput enc_out;

    t0 = std::chrono::high_resolution_clock::now();
    if (!aligner::encoder::encode_batch(enc_state, enc_in, enc_out, &error)) {
        fprintf(stderr, "FAIL: batch encode: %s\n", error.message.c_str());
        aligner::encoder::free(enc_state);
        return 1;
    }
    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < enc_out.batch_size(); ++i) {
        printf("  [%d] hidden=%d, frames=%d\n", i, enc_out.features[i].hidden_size, enc_out.features[i].n_frames);
    }
    printf("Batch encode time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    aligner::encoder::free(enc_state);

    // Step 5: Init aligner decoder
    printf("--- Step 5: Init aligner decoder ---\n");
    aligner::decoder::Config dec_cfg;
    dec_cfg.model_path = model_path;
    dec_cfg.device_name = device;
    dec_cfg.n_threads = 4;

    auto dec_state = aligner::decoder::init(dec_cfg);
    if (!dec_state) { fprintf(stderr, "FAIL: decoder init\n"); return 1; }
    printf("Decoder initialized\n\n");

    // Step 6: Align each wav (sequentially for now, decoder doesn't support batch)
    printf("--- Step 6: Align each wav ---\n");

    const char* texts[] = { "hello world this is a test", "good morning everyone" };
    const char* langs[] = { "english", "english" };

    std::vector<aligner::decoder::AlignOutput> align_outs(n_wavs);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_wavs; ++i) {
        aligner::decoder::AlignInput align_in;
        align_in.audio_features = enc_out.features[i].data.data();
        align_in.n_audio_frames = enc_out.features[i].n_frames;
        align_in.audio_feature_dim = enc_out.features[i].hidden_size;
        align_in.n_mel_frames = mel_specs[i].n_frames;
        align_in.text = texts[i];
        align_in.language = langs[i];

        if (!aligner::decoder::align(dec_state, align_in, align_outs[i], &error)) {
            fprintf(stderr, "FAIL: align [%d]: %s\n", i, error.message.c_str());
            aligner::decoder::free(dec_state);
            return 1;
        }
        printf("  [%d] %zu words, duration %.2f sec\n", i, align_outs[i].words.size(), align_outs[i].audio_duration);
        for (const auto& w : align_outs[i].words) {
            printf("    %.3f - %.3f: %s\n", w.start, w.end, w.word.c_str());
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("Align time (2 wavs sequential): %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    aligner::decoder::free(dec_state);

    printf("=== Summary ===\n");
    printf("Audio load:   %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    printf("Batch encode: done\n");
    printf("Sequential align: done\n");
    printf("=== Test PASSED ===\n");

    return 0;
}