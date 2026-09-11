#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdint>

namespace maba {

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor() = default;
    Tensor(const std::vector<size_t>& s) : shape(s) {
        size_t total = 1;
        for (auto d : s) total *= d;
        data.resize(total, 0.0f);
    }
    Tensor(const std::vector<size_t>& s, float val) : shape(s) {
        size_t total = 1;
        for (auto d : s) total *= d;
        data.resize(total, val);
    }

    size_t numel() const { return data.size(); }

    void resize(const std::vector<size_t>& s) {
        shape = s;
        size_t total = 1;
        for (auto d : s) total *= d;
        data.assign(total, 0.0f);
    }

    float* raw() { return data.data(); }
    const float* raw() const { return data.data(); }

    inline float& at(size_t i) { return data[i]; }
    inline const float& at(size_t i) const { return data[i]; }

    inline float& at(size_t i, size_t j) { return data[i * shape[1] + j]; }
    inline const float& at(size_t i, size_t j) const { return data[i * shape[1] + j]; }
};

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
inline float dot_product_simd(const float* a, const float* b, size_t K) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        __m256 va0 = _mm256_loadu_ps(a + k);
        __m256 vb0 = _mm256_loadu_ps(b + k);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        __m256 va1 = _mm256_loadu_ps(a + k + 8);
        __m256 vb1 = _mm256_loadu_ps(b + k + 8);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    float buf[8];
    _mm256_storeu_ps(buf, sum0);
    float total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; k < K; ++k) total += a[k] * b[k];
    return total;
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
inline float dot_product_simd(const float* a, const float* b, size_t K) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t va = vld1q_f32(a + k);
        float32x4_t vb = vld1q_f32(b + k);
        sum0 = vmlaq_f32(sum0, va, vb);
    }
    float total = vaddvq_f32(sum0);
    for (; k < K; ++k) total += a[k] * b[k];
    return total;
}
#else
inline float dot_product_simd(const float* a, const float* b, size_t K) {
    float total = 0.0f;
    #pragma omp simd reduction(+:total)
    for (size_t k = 0; k < K; ++k) total += a[k] * b[k];
    return total;
}
#endif

inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.shape.size() >= 2 && B.shape.size() == 2);
    size_t K = A.shape.back();
    assert(B.shape[0] == K);
    size_t N = B.shape[1];
    size_t M = A.numel() / K;

    C.resize({M, N});

    #pragma omp parallel for collapse(2) if (M * N > 64)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float* a_row = &A.data[i * K];
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; ++k) {
                sum += a_row[k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

inline void matmul_transB(const Tensor& A, const Tensor& B, Tensor& C) {
    size_t K = A.shape.back();
    assert(B.shape[1] == K);
    size_t N = B.shape[0];
    size_t M = A.numel() / K;

    C.resize({M, N});

    #pragma omp parallel for collapse(2) if (M * N > 64)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C.data[i * N + j] = dot_product_simd(&A.data[i * K], &B.data[j * K], K);
        }
    }
}

inline void rmsnorm(Tensor& out, const Tensor& in, const Tensor& weight, float eps = 1e-6f) {
    size_t D = weight.numel();
    size_t num_rows = in.numel() / D;
    out.resize(in.shape);

    #pragma omp parallel for if (num_rows > 1)
    for (size_t r = 0; r < num_rows; ++r) {
        const float* in_ptr = &in.data[r * D];
        float* out_ptr = &out.data[r * D];
        
        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (size_t d = 0; d < D; ++d) {
            sum_sq += in_ptr[d] * in_ptr[d];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / D + eps);

        #pragma omp simd
        for (size_t d = 0; d < D; ++d) {
            out_ptr[d] = in_ptr[d] * inv_rms * weight.data[d];
        }
    }
}

inline void gated_residual(Tensor& out, const Tensor& residual, const Tensor& sublayer, const Tensor& gate) {
    size_t D = gate.numel();
    size_t num_rows = residual.numel() / D;
    out.resize(residual.shape);

    #pragma omp parallel for if (num_rows > 16)
    for (size_t r = 0; r < num_rows; ++r) {
        const float* res_ptr = &residual.data[r * D];
        const float* sub_ptr = &sublayer.data[r * D];
        float* out_ptr = &out.data[r * D];

        for (size_t d = 0; d < D; ++d) {
            float g = sigmoid(gate.data[d]);
            out_ptr[d] = g * res_ptr[d] + (1.0f - g) * sub_ptr[d];
        }
    }
}

} // namespace maba

