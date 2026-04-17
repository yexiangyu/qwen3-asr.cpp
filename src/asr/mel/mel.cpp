#include "asr/mel/mel.hpp"
#include "asr/codec/codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
#include <mutex>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asr { namespace mel {

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

bool create_filter_bank(FilterBank& filters, const Config& config) {
    int n_fft_bins = 1 + config.n_fft / 2;
    float fmax = config.fmax > 0 ? config.fmax : config.sample_rate / 2.0f;
    
    filters.n_mels = config.n_mels;
    filters.n_fft_bins = n_fft_bins;
    filters.data.resize(config.n_mels * n_fft_bins, 0.0f);
    
    float mel_min = hz_to_mel(config.fmin);
    float mel_max = hz_to_mel(fmax);
    
    std::vector<float> mel_points(config.n_mels + 2);
    for (int i = 0; i < config.n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (config.n_mels + 1);
    }
    
    std::vector<float> hz_points(config.n_mels + 2);
    for (int i = 0; i < config.n_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    std::vector<float> bin_points(config.n_mels + 2);
    for (int i = 0; i < config.n_mels + 2; i++) {
        bin_points[i] = (config.n_fft + 1) * hz_points[i] / config.sample_rate;
    }
    
    for (int m = 0; m < config.n_mels; m++) {
        float left = bin_points[m];
        float center = bin_points[m + 1];
        float right = bin_points[m + 2];
        
        for (int k = 0; k < n_fft_bins; k++) {
            float weight = 0.0f;
            if (k >= left && k <= center && center > left) {
                weight = (k - left) / (center - left);
            } else if (k >= center && k <= right && right > center) {
                weight = (right - k) / (right - center);
            }
            filters.data[m * n_fft_bins + k] = weight;
        }
    }
    
    for (int m = 0; m < config.n_mels; m++) {
        float enorm = 2.0f / (hz_points[m + 2] - hz_points[m]);
        for (int k = 0; k < n_fft_bins; k++) {
            filters.data[m * n_fft_bins + k] *= enorm;
        }
    }
    
    return true;
}

bool create_hann_window(Window& window, int length, bool periodic) {
    window.length = length;
    window.data.resize(length);
    
    int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) {
        window.data[i] = 0.5 * (1.0 - cos((2.0 * M_PI * i) / (length + offset)));
    }
    
    return true;
}

static void dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            float angle = 2.0f * M_PI * k * n / N;
            re += in[n] * cosf(angle);
            im -= in[n] * sinf(angle);
        }
        out[2 * k + 0] = re;
        out[2 * k + 1] = im;
    }
}

static void fft_recursive(const float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    
    if (N % 2 != 0) {
        dft(in, N, out);
        return;
    }
    
    int half_N = N / 2;
    std::vector<float> even(half_N);
    std::vector<float> odd(half_N);
    
    for (int i = 0; i < half_N; i++) {
        even[i] = in[2 * i];
        odd[i] = in[2 * i + 1];
    }
    
    std::vector<float> even_fft(N * 2);
    std::vector<float> odd_fft(N * 2);
    
    fft_recursive(even.data(), half_N, even_fft.data());
    fft_recursive(odd.data(), half_N, odd_fft.data());
    
    for (int k = 0; k < half_N; k++) {
        float angle = 2.0f * M_PI * k / N;
        float re = cosf(angle);
        float im = -sinf(angle);
        
        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];
        
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        
        out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

bool compute(const Input& input, MelSpectrum& output, const Config& config, ErrorInfo* error) {
    int frame_size = config.n_fft;
    int frame_step = config.hop_length;
    int n_fft_bins = 1 + config.n_fft / 2;
    
    int pad_amount = frame_size / 2;
    
    std::vector<float> samples_padded;
    samples_padded.resize(input.n_samples + 2 * pad_amount);
    
    std::copy(input.samples, input.samples + input.n_samples, samples_padded.begin() + pad_amount);
    
    for (int i = 0; i < pad_amount; i++) {
        int src_idx = pad_amount - i;
        if (src_idx < input.n_samples) {
            samples_padded[i] = input.samples[src_idx];
        } else {
            samples_padded[i] = 0.0f;
        }
    }
    
    for (int i = 0; i < pad_amount; i++) {
        int src_idx = input.n_samples - 2 - i;
        if (src_idx >= 0) {
            samples_padded[input.n_samples + pad_amount + i] = input.samples[src_idx];
        } else {
            samples_padded[input.n_samples + pad_amount + i] = 0.0f;
        }
    }
    
    int total_frames = (static_cast<int>(samples_padded.size()) - frame_size) / frame_step + 1;
    int n_frames = total_frames - 1;
    
    Window hann;
    create_hann_window(hann, frame_size, true);
    
    FilterBank filters;
    create_filter_bank(filters, config);
    
    output.n_mels = config.n_mels;
    output.n_frames = n_frames;
    output.data.resize(config.n_mels * n_frames, 0.0f);
    
    std::vector<float> power_all(n_fft_bins * n_frames, 0.0f);
    
    auto process_frame = [&](int frame_idx) {
        int offset = frame_idx * frame_step;
        
        std::vector<float> fft_in(frame_size, 0.0f);
        std::vector<float> fft_out(frame_size * 2);
        
        int valid_len = std::min(frame_size, static_cast<int>(samples_padded.size()) - offset);
        for (int j = 0; j < valid_len; j++) {
            fft_in[j] = static_cast<float>(hann.data[j]) * samples_padded[offset + j];
        }
        
        fft_recursive(fft_in.data(), frame_size, fft_out.data());
        
        for (int j = 0; j < n_fft_bins; j++) {
            power_all[j * n_frames + frame_idx] = 
                fft_out[2 * j + 0] * fft_out[2 * j + 0] + 
                fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
    };
    
    std::vector<std::thread> threads;
    int frames_per_thread = (n_frames + config.n_threads - 1) / config.n_threads;
    
    for (int t = 0; t < config.n_threads; t++) {
        int start = t * frames_per_thread;
        int end = std::min(start + frames_per_thread, n_frames);
        if (start < n_frames) {
            threads.emplace_back([&, start, end]() {
                for (int i = start; i < end; i++) {
                    process_frame(i);
                }
            });
        }
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    for (int m = 0; m < config.n_mels; m++) {
        for (int f = 0; f < n_frames; f++) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; k++) {
                sum += filters.data[m * n_fft_bins + k] * power_all[k * n_frames + f];
            }
            output.data[m * n_frames + f] = logf(sum + 1e-10f);
        }
    }
    
    return true;
}

bool compute_from_file(const char* wav_path, MelSpectrum& output, const Config& config, ErrorInfo* error) {
    std::vector<float> samples;
    int sample_rate;
    
    if (!codec::decode_file(wav_path, samples, sample_rate, error)) {
        return false;
    }
    
    if (sample_rate != config.sample_rate) {
        if (error) error->message = "Sample rate mismatch: expected " + std::to_string(config.sample_rate) + 
            ", got " + std::to_string(sample_rate);
        return false;
    }
    
    Input input;
    input.samples = samples.data();
    input.n_samples = static_cast<int>(samples.size());
    
    return compute(input, output, config, error);
}

bool load_ref_data(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return true;
}

bool save_ref_data(const char* path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

bool compare_float_arrays(const std::vector<float>& a, const std::vector<float>& b, float tolerance, bool verbose) {
    if (a.size() != b.size()) {
        if (verbose) fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size(), b.size());
        return false;
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    if (max_diff > tolerance) {
        if (verbose) fprintf(stderr, "Max diff exceeds tolerance: %.6f > %.6f\n", max_diff, tolerance);
        return false;
    }
    
    return true;
}

} // namespace mel
} // namespace asr
