#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace maba {

struct GDN2W {
    Tensor q_proj;      // (640, 640)
    Tensor k_proj;      // (640, 640)
    Tensor v_proj;      // (640, 640)
    Tensor o_proj;      // (640, 640)
    Tensor conv_q;      // (640, 1, 4)
    Tensor conv_k;      // (640, 1, 4)
    Tensor conv_v;      // (640, 1, 4)
    Tensor gate_alpha;  // (10, 640)
    Tensor gate_erase;  // (10, 640)
    Tensor gate_write;  // (10, 640)
};

using GDN2Weights = GDN2W;

struct GDN2State {
    std::vector<float> S;
    std::vector<float> conv_buf_q;
    std::vector<float> conv_buf_k;
    std::vector<float> conv_buf_v;

    GDN2State() {
        S.assign(10 * 64 * 64, 0.0f);
        conv_buf_q.assign(640 * 3, 0.0f);
        conv_buf_k.assign(640 * 3, 0.0f);
        conv_buf_v.assign(640 * 3, 0.0f);
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

    for (size_t t = 0; t < L; ++t) {
        for (size_t c = 0; c < D; ++c) {
            float sum = 0.0f;
            const float* w = &conv_w.data[c * 4];
            for (int k = 0; k < 4; ++k) {
                int lag = 3 - k;
                float val = 0.0f;
                int pos = (int)t - lag;
                if (pos >= 0) {
                    val = in.at(pos, c);
                } else if (!state_buf.empty()) {
                    int buf_idx = (int)state_buf.size() / D + pos;
                    if (buf_idx >= 0 && buf_idx < 3) {
                        val = state_buf[c * 3 + buf_idx];
                    }
                }
                sum += val * w[k];
            }
            out.at(t, c) = silu(sum);
        }
    }

    if (update_state && L > 0) {
        if (L >= 3) {
            for (size_t c = 0; c < D; ++c) {
                for (int i = 0; i < 3; ++i) {
                    state_buf[c * 3 + i] = in.at(L - 3 + i, c);
                }
            }
        } else {
            for (size_t c = 0; c < D; ++c) {
                float old0 = state_buf[c * 3 + 0];
                float old1 = state_buf[c * 3 + 1];
                float old2 = state_buf[c * 3 + 2];
                for (int i = 0; i < 3; ++i) {
                    int idx = (int)L + i;
                    if (idx == 0) state_buf[c * 3 + i] = old0;
                    else if (idx == 1) state_buf[c * 3 + i] = old1;
                    else if (idx == 2) state_buf[c * 3 + i] = old2;
                    else state_buf[c * 3 + i] = in.at(idx - 3, c);
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
    size_t D = 640;
    size_t H = 10;
    size_t d = 64;

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
            for (size_t i = 0; i < d; ++i) k_norm_sq += k_raw[i] * k_raw[i];
            float inv_k_norm = 1.0f / (std::sqrt(k_norm_sq) + 1e-6f);

            float k_t[64], e_t[64], z_t[64];
            for (size_t i = 0; i < d; ++i) {
                k_t[i] = k_raw[i] * inv_k_norm;
                e_t[i] = b_val * k_t[i];
                z_t[i] = w_val * v_raw[i];
            }

            float e_S[64] = {0.0f};
            for (size_t col = 0; col < d; ++col) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (size_t row = 0; row < d; ++row) {
                    sum += e_t[row] * (alpha * S_h[row * d + col]);
                }
                e_S[col] = sum;
            }

            float delta[64];
            for (size_t i = 0; i < d; ++i) delta[i] = z_t[i] - e_S[i];

            for (size_t row = 0; row < d; ++row) {
                float k_val = k_t[row];
                float* S_row = &S_h[row * d];
                #pragma omp simd
                for (size_t col = 0; col < d; ++col) {
                    S_row[col] = alpha * S_row[col] + k_val * delta[col];
                }
            }

            for (size_t col = 0; col < d; ++col) {
                float o_val = 0.0f;
                for (size_t row = 0; row < d; ++row) {
                    o_val += q_t[row] * S_h[row * d + col];
                }
                o_concat.data[t * D + h * d + col] = o_val;
            }
        }
    }

    matmul_transB(o_concat, w.o_proj, out);
}

inline void gdn2_forward(Tensor& out, const Tensor& x, const GDN2W& w, GDN2State& s) {
    gdn2_fwd(out, x, w, s);
}

} // namespace maba

