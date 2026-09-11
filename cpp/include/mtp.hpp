#pragma once

#include "tensor.hpp"
#include "embeddings.hpp"

namespace maba {

struct MTPW {
    Tensor proj_weight; // (640, 768)
    Tensor proj_bias;   // (640,)
};

using MTPWeights = MTPW;

inline void mtp_predict(
    Tensor& mtp_logits,
    const Tensor& final_normed_h,
    int next_token_id,
    const MTPW& mtp_w,
    const EmbW& emb_w
) {
    size_t d_model = 640;
    size_t d_emb = 128;

    Tensor combined({1, d_model + d_emb});
    std::memcpy(&combined.data[0], &final_normed_h.data[final_normed_h.numel() - d_model], d_model * sizeof(float));

    if (next_token_id < 0 || next_token_id >= 32768) next_token_id = 0;
    std::memcpy(&combined.data[d_model], &emb_w.w_emb.data[next_token_id * d_emb], d_emb * sizeof(float));

    Tensor mtp_hidden;
    matmul_transB(combined, mtp_w.proj_weight, mtp_hidden);

    for (size_t i = 0; i < d_model; ++i) {
        mtp_hidden.data[i] += mtp_w.proj_bias.data[i];
    }

    dequantize_logits(mtp_logits, mtp_hidden, emb_w);
}

} // namespace maba

