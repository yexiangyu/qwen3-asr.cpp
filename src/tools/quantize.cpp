#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static const char * ggml_type_name_ext(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:     return "f32";
        case GGML_TYPE_F16:     return "f16";
        case GGML_TYPE_Q8_0:    return "q8_0";
        case GGML_TYPE_NVFP4:   return "nvfp4";
        default:                return ggml_type_name(t);
    }
}

static bool should_quantize_tensor(const char * name, ggml_type type, int64_t nrows) {
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32) return false;
    if (nrows <= 1) return false;
    if (strstr(name, "norm.weight") || strstr(name, "norm.bias")) return false;
    if (strstr(name, "_norm.weight")) return false;
    if (strstr(name, ".bias")) return false;
    if (strstr(name, "embed_tokens") || strstr(name, "token_embd")) return false;
    if (strstr(name, "lm_head") && strstr(name, ".weight")) return false;
    if (strstr(name, "output.weight")) return false;
    if (strstr(name, "classify_head")) return false;
    if (strstr(name, "ln_post.weight") || strstr(name, "ln_post.bias")) return false;
    if (strstr(name, "proj1.bias") || strstr(name, "proj2.bias")) return false;
    if (strstr(name, "conv1.bias") || strstr(name, "conv2.bias") || strstr(name, "conv3.bias")) return false;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <quant_type>\n", argv[0]);
        fprintf(stderr, "  quant_type: nvfp4, q8_0, q4_0\n");
        fprintf(stderr, "Example: %s models/qwen3-asr-1.7b-f16.gguf models/qwen3-asr-1.7b-nvfp4.gguf nvfp4\n", argv[0]);
        return 1;
    }

    const char * input_path = argv[1];
    const char * output_path = argv[2];
    const char * quant_type_str = argv[3];

    ggml_type quant_type;
    ggml_ftype ftype;
    if (strcmp(quant_type_str, "nvfp4") == 0) {
        quant_type = GGML_TYPE_NVFP4;
        ftype = GGML_FTYPE_MOSTLY_NVFP4;
    } else if (strcmp(quant_type_str, "q8_0") == 0) {
        quant_type = GGML_TYPE_Q8_0;
        ftype = GGML_FTYPE_MOSTLY_Q8_0;
    } else if (strcmp(quant_type_str, "q4_0") == 0) {
        quant_type = GGML_TYPE_Q4_0;
        ftype = GGML_FTYPE_MOSTLY_Q4_0;
    } else {
        fprintf(stderr, "Unknown quant type: %s (supported: nvfp4, q8_0, q4_0)\n", quant_type_str);
        return 1;
    }

    fprintf(stderr, "Loading %s ...\n", input_path);

    struct ggml_context * ggml_ctx_in = nullptr;
    struct gguf_init_params params = { true, &ggml_ctx_in };
    struct gguf_context * ctx_in = gguf_init_from_file(input_path, params);
    if (!ctx_in) {
        fprintf(stderr, "Failed to open input: %s\n", input_path);
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx_in);
    fprintf(stderr, "Input: %lld KV pairs, %lld tensors\n", (long long)gguf_get_n_kv(ctx_in), (long long)n_tensors);

    size_t data_offset = gguf_get_data_offset(ctx_in);
    FILE * fin = fopen(input_path, "rb");
    if (!fin) { fprintf(stderr, "Cannot re-open input\n"); gguf_free(ctx_in); return 1; }

    std::map<std::string, size_t> tensor_offsets;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        size_t offset = gguf_get_tensor_offset(ctx_in, i);
        tensor_offsets[name] = offset;
    }

    struct gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    int64_t ftype_idx = gguf_find_key(ctx_in, "general.file_type");
    if (ftype_idx >= 0) {
        gguf_set_val_u32(ctx_out, "general.file_type", (uint32_t)ftype);
    }

    ggml_quantize_init(quant_type);

    // Create output ggml context for adding tensors
    size_t ctx_size = 0;
    // Calculate context size needed for all tensor definitions
    for (ggml_tensor * t = ggml_get_first_tensor(ggml_ctx_in); t; t = ggml_get_next_tensor(ggml_ctx_in, t)) {
        ctx_size += ggml_tensor_overhead();
    }
    ctx_size += ggml_graph_overhead();

    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
    struct ggml_context * ggml_ctx_out = ggml_init(ctx_params);

    int64_t n_quantized = 0;
    int64_t n_skipped = 0;
    size_t total_orig_size = 0;
    size_t total_new_size = 0;

    std::vector<std::pair<std::string, std::vector<uint8_t>>> tensor_data_list;

    int tensor_idx = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ggml_ctx_in); t; t = ggml_get_next_tensor(ggml_ctx_in, t), tensor_idx++) {
        const char * name = ggml_get_name(t);
        ggml_type orig_type = t->type;
        size_t orig_offset = tensor_offsets[name];
        int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };

        int64_t n_elem = ne[0] * ne[1] * ne[2] * ne[3];
        int64_t nrows = ne[1] * ne[2] * ne[3];
        int64_t n_per_row = ne[0];

        size_t orig_size = ggml_nbytes(t);
        total_orig_size += orig_size;

        int64_t blck_size = ggml_get_type_traits(quant_type)->blck_size;
        bool do_quantize = should_quantize_tensor(name, orig_type, nrows) && (n_per_row % blck_size == 0);
        ggml_type new_type = do_quantize ? quant_type : orig_type;

        if (nrows <= 1) do_quantize = false;

        if (do_quantize && n_per_row % blck_size != 0) {
            fprintf(stderr, "  NOTE: skipping %s (n_per_row=%lld not multiple of block_size=%lld)\n",
                    name, (long long)n_per_row, (long long)blck_size);
            do_quantize = false;
            new_type = orig_type;
        }

        fseek(fin, data_offset + orig_offset, SEEK_SET);
        std::vector<uint8_t> orig_data(orig_size);
        if (fread(orig_data.data(), 1, orig_size, fin) != orig_size) {
            fprintf(stderr, "Failed to read tensor %s\n", name);
            fclose(fin); gguf_free(ctx_in); gguf_free(ctx_out); return 1;
        }

        std::vector<float> f32_data(n_elem);
        if (orig_type == GGML_TYPE_F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->to_float(orig_data.data(), f32_data.data(), n_elem);
        } else if (orig_type == GGML_TYPE_F32) {
            memcpy(f32_data.data(), orig_data.data(), n_elem * sizeof(float));
        } else {
            auto * traits = ggml_get_type_traits(orig_type);
            if (traits->to_float) {
                traits->to_float(orig_data.data(), f32_data.data(), n_elem);
            } else {
                new_type = orig_type;
                do_quantize = false;
                f32_data.clear();
            }
        }

        std::vector<uint8_t> new_data;
        size_t actual_new_size;

        if (do_quantize && !f32_data.empty()) {
            size_t quant_size = nrows * ggml_row_size(quant_type, n_per_row);
            new_data.resize(quant_size);

            fprintf(stderr, "  [%3d/%3d] %50s  %6s -> %6s  quantizing (%lld rows x %lld cols)...\n",
                    (int)0, (int)n_tensors, name,
                    ggml_type_name_ext(orig_type), ggml_type_name_ext(new_type),
                    (long long)nrows, (long long)n_per_row);

            size_t result = ggml_quantize_chunk(quant_type, f32_data.data(), new_data.data(),
                                                 0, nrows, n_per_row, nullptr);

            if (result == 0) {
                fprintf(stderr, "  Quantization FAILED for %s, keeping original\n", name);
                new_type = orig_type;
                new_data = orig_data;
                actual_new_size = orig_size;
            } else {
                actual_new_size = quant_size;
                n_quantized++;
            }
        } else {
            new_type = orig_type;
            new_data = orig_data;
            actual_new_size = orig_size;
            n_skipped++;
        }

        total_new_size += actual_new_size;

        // Create tensor in output context
        ggml_tensor * new_t = ggml_new_tensor(ggml_ctx_out, new_type, 4, ne);
        ggml_set_name(new_t, name);

        // Add tensor to gguf context
        gguf_add_tensor(ctx_out, new_t);

        // Store data for later writing
        tensor_data_list.push_back({std::string(name), std::move(new_data)});

        fprintf(stderr, "  [%3d/%3d] %50s  %6s -> %6s  %8zu -> %8zu bytes%s\n",
                (int)tensor_idx + 1, (int)n_tensors, name,
                ggml_type_name_ext(orig_type), ggml_type_name_ext(new_type),
                orig_size, actual_new_size,
                do_quantize ? "" : " (kept)");
    }

    fclose(fin);

    fprintf(stderr, "\nQuantization complete: %lld quantized, %lld kept\n",
            (long long)n_quantized, (long long)n_skipped);
    fprintf(stderr, "Original: %.2f MB -> New: %.2f MB (%.1f%% reduction)\n",
            total_orig_size / 1024.0 / 1024.0,
            total_new_size / 1024.0 / 1024.0,
            (1.0 - (double)total_new_size / total_orig_size) * 100.0);

    // Write entire file (metadata + data) in single pass
    // Must set tensor data via gguf_set_tensor_data before writing
    for (auto & [name, data] : tensor_data_list) {
        gguf_set_tensor_data(ctx_out, name.c_str(), data.data());
    }

    fprintf(stderr, "Writing %s ...\n", output_path);
    if (!gguf_write_to_file(ctx_out, output_path, false)) {
        fprintf(stderr, "Failed to write output\n");
        gguf_free(ctx_in); gguf_free(ctx_out); ggml_free(ggml_ctx_out); return 1;
    }

    ggml_quantize_free();
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ggml_ctx_out);

    fprintf(stderr, "Done. Output: %s\n", output_path);
    return 0;
}