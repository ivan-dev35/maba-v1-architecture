#pragma once

#include "tensor.hpp"
#include <vector>

namespace maba {

struct EmbW {
    Tensor w_emb;
    Tensor w_proj_in;
    Tensor w_proj_out;
};

using EmbeddingWeights = EmbW;

inline void factorized_embed(
    Tensor& out,
    const std::vector<int>& token_ids,
    const EmbW& w
) {
    size_t L = token_ids.size();
    size_t vocab_size = w.w_emb.shape[0];
    size_t d_emb = w.w_emb.shape[1];

    Tensor factor_emb({L, d_emb});
    for (size_t t = 0; t < L; ++t) {
        int tid = token_ids[t];
        if (tid < 0 || (size_t)tid >= vocab_size) tid = 0;
        const float* src = &w.w_emb.data[tid * d_emb];
        std::memcpy(&factor_emb.data[t * d_emb], src, d_emb * sizeof(float));
    }

    matmul_transB(factor_emb, w.w_proj_in, out);
}

inline void dequantize_logits(
    Tensor& logits,
    const Tensor& hidden,
    const EmbW& w
) {
    Tensor comp;
    matmul_transB(hidden, w.w_proj_out, comp);
    matmul_transB(comp, w.w_emb, logits);
}

} // namespace maba

