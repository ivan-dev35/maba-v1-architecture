#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace maba {

struct GQAW {
    Tensor q_proj;  // (640, 640)
    Tensor k_proj;  // (128, 640)
    Tensor v_proj;  // (128, 640)
    Tensor o_proj;  // (640, 640)
    Tensor q_norm;  // (640,)
    Tensor k_norm;  // (640,)
};

using GQAWeights = GQAW;

struct GQACache {
    std::vector<float> k_cache; // (total_seq_len, 10, 64)
    std::vector<float> v_cache; // (total_seq_len, 10, 64)
    size_t seq_len = 0;
};

using GQAKVCache = GQACache;

inline void apply_rope(float* vec, size_t pos, size_t d = 64, float theta = 500000.0f) {
    for (size_t i = 0; i < d / 2; ++i) {
        float freq = 1.0f / std::pow(theta, (2.0f * i) / d);
        float angle = pos * freq;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        float x1 = vec[i];
        float x2 = vec[i + d / 2];

        vec[i] = x1 * cos_a - x2 * sin_a;
        vec[i + d / 2] = x2 * cos_a + x1 * sin_a;
    }
}

inline void gqa_fwd(
    Tensor& out,
    const Tensor& x,
    const GQAW& w,
    GQACache& cache,
    size_t start_pos = 0
) {
    size_t L = x.shape[0];
    size_t D = 640;
    size_t H_q = 10;
    size_t H_kv = 2;
    size_t d = 64;
    size_t group_ratio = H_q / H_kv;

    Tensor q_raw, k_raw, v_raw;
    matmul_transB(x, w.q_proj, q_raw);
    matmul_transB(x, w.k_proj, k_raw);
    matmul_transB(x, w.v_proj, v_raw);

    Tensor q_normed, k_expanded({L, D}), k_normed;
    rmsnorm(q_normed, q_raw, w.q_norm);

    for (size_t t = 0; t < L; ++t) {
        for (size_t kv_h = 0; kv_h < H_kv; ++kv_h) {
            for (size_t g = 0; g < group_ratio; ++g) {
                size_t q_h = kv_h * group_ratio + g;
                std::memcpy(&k_expanded.data[t * D + q_h * d],
                            &k_raw.data[t * 128 + kv_h * d],
                            d * sizeof(float));
            }
        }
    }
    rmsnorm(k_normed, k_expanded, w.k_norm);

    Tensor v_expanded({L, D});
    for (size_t t = 0; t < L; ++t) {
        for (size_t kv_h = 0; kv_h < H_kv; ++kv_h) {
            for (size_t g = 0; g < group_ratio; ++g) {
                size_t q_h = kv_h * group_ratio + g;
                std::memcpy(&v_expanded.data[t * D + q_h * d],
                            &v_raw.data[t * 128 + kv_h * d],
                            d * sizeof(float));
            }
        }
    }

    for (size_t t = 0; t < L; ++t) {
        size_t abs_pos = start_pos + t;
        for (size_t h = 0; h < H_q; ++h) {
            apply_rope(&q_normed.data[t * D + h * d], abs_pos, d);
            apply_rope(&k_normed.data[t * D + h * d], abs_pos, d);
        }
    }

    if (start_pos == 0) {
        cache.seq_len = 0;
        cache.k_cache.clear();
        cache.v_cache.clear();
    } else if (cache.seq_len != start_pos) {
        cache.seq_len = start_pos;
        cache.k_cache.resize(start_pos * D);
        cache.v_cache.resize(start_pos * D);
    }

    size_t total_L = cache.seq_len + L;
    cache.k_cache.resize(total_L * D);
    cache.v_cache.resize(total_L * D);
    std::memcpy(&cache.k_cache[cache.seq_len * D], k_normed.data.data(), L * D * sizeof(float));
    std::memcpy(&cache.v_cache[cache.seq_len * D], v_expanded.data.data(), L * D * sizeof(float));
    cache.seq_len = total_L;

    float scale = 1.0f / std::sqrt((float)d);
    Tensor attn_out({L, D});

    for (size_t t = 0; t < L; ++t) {
        size_t query_pos = start_pos + t;
        for (size_t h = 0; h < H_q; ++h) {
            const float* q_vec = &q_normed.data[t * D + h * d];
            std::vector<float> scores(query_pos + 1);
            float max_score = -1e9f;

            for (size_t key_t = 0; key_t <= query_pos; ++key_t) {
                const float* k_vec = &cache.k_cache[key_t * D + h * d];
                float dot = 0.0f;
                for (size_t i = 0; i < d; ++i) dot += q_vec[i] * k_vec[i];
                scores[key_t] = dot * scale;
                if (scores[key_t] > max_score) max_score = scores[key_t];
            }

            float sum_exp = 0.0f;
            for (size_t key_t = 0; key_t <= query_pos; ++key_t) {
                scores[key_t] = std::exp(scores[key_t] - max_score);
                sum_exp += scores[key_t];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            float* out_vec = &attn_out.data[t * D + h * d];
            for (size_t i = 0; i < d; ++i) out_vec[i] = 0.0f;

            for (size_t key_t = 0; key_t <= query_pos; ++key_t) {
                float weight = scores[key_t] * inv_sum;
                const float* v_vec = &cache.v_cache[key_t * D + h * d];
                for (size_t i = 0; i < d; ++i) {
                    out_vec[i] += weight * v_vec[i];
                }
            }
        }
    }

    matmul_transB(attn_out, w.o_proj, out);
}

inline void gqa_forward(
    Tensor& out, const Tensor& x, const GQAW& w, GQACache& c, size_t sp = 0
) {
    gqa_fwd(out, x, w, c, sp);
}

} // namespace maba

