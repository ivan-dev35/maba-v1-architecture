#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace maba {

struct GDN2W {
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor conv_q;
    Tensor conv_k;
    Tensor conv_v;
    Tensor gate_alpha;
    Tensor gate_erase;
    Tensor gate_write;
    size_t H = 10;
    size_t d = 64;
};

using GDN2Weights = GDN2W;

struct GDN2State {
    std::vector<float> S;
    std::vector<float> conv_buf_q;
    std::vector<float> conv_buf_k;
    std::vector<float> conv_buf_v;

    void reset() {
        std::fill(S.begin(), S.end(), 0.0f);
        std::fill(conv_buf_q.begin(), conv_buf_q.end(), 0.0f);
        std::fill(conv_buf_k.begin(), conv_buf_k.end(), 0.0f);
        std::fill(conv_buf_v.begin(), conv_buf_v.end(), 0.0f);
    }

    void init(size_t H, size_t d, size_t D, size_t k_size = 4) {
        size_t buf_k = k_size > 1 ? (k_size - 1) : 0;
        if (S.size() != H * d * d) S.assign(H * d * d, 0.0f);
        if (conv_buf_q.size() != D * buf_k) conv_buf_q.assign(D * buf_k, 0.0f);
        if (conv_buf_k.size() != D * buf_k) conv_buf_k.assign(D * buf_k, 0.0f);
        if (conv_buf_v.size() != D * buf_k) conv_buf_v.assign(D * buf_k, 0.0f);
    }
};

inline void conv1d_fwd(
    Tensor& out,
    const Tensor& in,
    const Tensor& conv_w,
    std::vector<float>& state_buf,
    bool update_state
) {
    size_t L = in.shape[0];
    size_t D = in.shape[1];
    out.resize({L, D});
    size_t k_size = (conv_w.numel() > 0 && D > 0) ? (conv_w.numel() / D) : 4;
    int buf_k = (int)k_size - 1;

    for (size_t t = 0; t < L; ++t) {
        for (size_t c = 0; c < D; ++c) {
            float sum = 0.0f;
            const float* w = &conv_w.data[c * k_size];
            for (size_t k = 0; k < k_size; ++k) {
                int lag = buf_k - (int)k;
                float val = 0.0f;
                int pos = (int)t - lag;
                if (pos >= 0) {
                    val = in.at(pos, c);
                } else if (!state_buf.empty() && buf_k > 0) {
                    int buf_idx = buf_k + pos;
                    if (buf_idx >= 0 && buf_idx < buf_k) {
                        val = state_buf[c * buf_k + buf_idx];
                    }
                }
                sum += val * w[k];
            }
            out.at(t, c) = silu(sum);
        }
    }

    if (update_state && L > 0 && buf_k > 0) {
        if ((int)L >= buf_k) {
            for (size_t c = 0; c < D; ++c) {
                for (int i = 0; i < buf_k; ++i) {
                    state_buf[c * buf_k + i] = in.at(L - buf_k + i, c);
                }
            }
        } else {
            std::vector<float> old_dyn;
            float old_stk[16];
            float* old = (buf_k <= 16) ? old_stk : (old_dyn.resize(buf_k), old_dyn.data());
            for (size_t c = 0; c < D; ++c) {
                for (int i = 0; i < buf_k; ++i) old[i] = state_buf[c * buf_k + i];
                for (int i = 0; i < buf_k; ++i) {
                    int idx = (int)L + i;
                    if (idx < buf_k) state_buf[c * buf_k + i] = old[idx];
                    else state_buf[c * buf_k + i] = in.at(idx - buf_k, c);
                }
            }
        }
    }
}

inline void causal_conv1d_forward(
    Tensor& out, const Tensor& in, const Tensor& conv_w, std::vector<float>& buf, bool upd
) {
    conv1d_fwd(out, in, conv_w, buf, upd);
}

inline void gdn2_fwd(
    Tensor& out,
    const Tensor& x,
    const GDN2W& w,
    GDN2State& state
) {
    size_t L = x.shape[0];
    size_t D = x.shape[1];
    size_t H = (w.H > 0) ? w.H : (w.gate_alpha.shape.empty() ? 10 : w.gate_alpha.shape[0]);
    size_t d = (w.d > 0) ? w.d : (D / H);

    size_t k_size = (w.conv_q.numel() > 0 && D > 0) ? (w.conv_q.numel() / D) : 4;
    state.init(H, d, D, k_size);

    Tensor q_proj, k_proj, v_proj;
    matmul_transB(x, w.q_proj, q_proj);
    matmul_transB(x, w.k_proj, k_proj);
    matmul_transB(x, w.v_proj, v_proj);

    Tensor q_conv, k_conv, v_conv;
    conv1d_fwd(q_conv, q_proj, w.conv_q, state.conv_buf_q, true);
    conv1d_fwd(k_conv, k_proj, w.conv_k, state.conv_buf_k, true);
    conv1d_fwd(v_conv, v_proj, w.conv_v, state.conv_buf_v, true);

    Tensor alpha_g, erase_g, write_g;
    matmul_transB(x, w.gate_alpha, alpha_g);
    matmul_transB(x, w.gate_erase, erase_g);
    matmul_transB(x, w.gate_write, write_g);

    Tensor o_concat({L, D});

    std::vector<float> k_t(d), e_t(d), z_t(d), e_S(d), delta(d);

    for (size_t t = 0; t < L; ++t) {
        for (size_t h = 0; h < H; ++h) {
            float alpha = sigmoid(alpha_g.at(t, h));
            float b_val = sigmoid(erase_g.at(t, h));
            float w_val = sigmoid(write_g.at(t, h));

            const float* q_t = &q_conv.data[t * D + h * d];
            const float* k_raw = &k_conv.data[t * D + h * d];
            const float* v_raw = &v_conv.data[t * D + h * d];

            float* S_h = &state.S[h * d * d];

            float k_norm_sq = 0.0f;
            #pragma omp simd reduction(+:k_norm_sq)
            for (size_t i = 0; i < d; ++i) k_norm_sq += k_raw[i] * k_raw[i];
            float inv_k_norm = 1.0f / (std::sqrt(k_norm_sq) + 1e-6f);

            for (size_t i = 0; i < d; ++i) {
                k_t[i] = k_raw[i] * inv_k_norm;
                e_t[i] = b_val * k_t[i];
                z_t[i] = w_val * v_raw[i];
            }

            for (size_t col = 0; col < d; ++col) e_S[col] = 0.0f;
            for (size_t row = 0; row < d; ++row) {
                float e_val = e_t[row] * alpha;
                const float* S_row = &S_h[row * d];
                #pragma omp simd
                for (size_t col = 0; col < d; ++col) {
                    e_S[col] += e_val * S_row[col];
                }
            }

            for (size_t i = 0; i < d; ++i) delta[i] = z_t[i] - e_S[i];

            for (size_t row = 0; row < d; ++row) {
                float k_val = k_t[row];
                float* S_row = &S_h[row * d];
                #pragma omp simd
                for (size_t col = 0; col < d; ++col) {
                    S_row[col] = alpha * S_row[col] + k_val * delta[col];
                }
            }

            float* o_ptr = &o_concat.data[t * D + h * d];
            for (size_t col = 0; col < d; ++col) o_ptr[col] = 0.0f;
            for (size_t row = 0; row < d; ++row) {
                float q_val = q_t[row];
                const float* S_row = &S_h[row * d];
                #pragma omp simd
                for (size_t col = 0; col < d; ++col) {
                    o_ptr[col] += q_val * S_row[col];
                }
            }
        }
    }

    matmul_transB(o_concat, w.o_proj, out);
}

inline void gdn2_forward(Tensor& out, const Tensor& x, const GDN2W& w, GDN2State& s) {
    gdn2_fwd(out, x, w, s);
}

} // namespace maba

