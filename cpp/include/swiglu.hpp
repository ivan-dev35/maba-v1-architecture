#pragma once

#include "tensor.hpp"

namespace maba {

struct FFNW {
    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

using SwiGLUWeights = FFNW;

inline void ffn_fwd(
    Tensor& out,
    const Tensor& x,
    const FFNW& w
) {
    size_t L = x.shape[0];
    size_t d_ffn = !w.w_gate.shape.empty() ? w.w_gate.shape[0] : 1728;

    Tensor gate, up;
    matmul_transB(x, w.w_gate, gate);
    matmul_transB(x, w.w_up, up);

    Tensor intermediate({L, d_ffn});
    #pragma omp parallel for collapse(2) if (L * d_ffn > 1024)
    for (size_t t = 0; t < L; ++t) {
        for (size_t i = 0; i < d_ffn; ++i) {
            intermediate.at(t, i) = silu(gate.at(t, i)) * up.at(t, i);
        }
    }

    matmul_transB(intermediate, w.w_down, out);
}

inline void swiglu_forward(Tensor& out, const Tensor& x, const FFNW& w) {
    ffn_fwd(out, x, w);
}

} // namespace maba

