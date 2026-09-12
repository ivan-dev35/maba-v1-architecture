#pragma once

#include "tensor.hpp"
#include "embeddings.hpp"

namespace maba {

struct MTPW {
    Tensor proj_weight;
    Tensor proj_bias;
};

using MTPWeights = MTPW;

inline void mtp_predict(
    Tensor& mtp_logits,
    const Tensor& final_normed_h,
    int next_token_id,
    const MTPW& mtp_w,
    const EmbW& emb_w
) {
    size_t d_model = !mtp_w.proj_bias.shape.empty() ? mtp_w.proj_bias.numel() : (!emb_w.w_proj_in.shape.empty() ? emb_w.w_proj_in.shape[0] : 640);
    size_t d_emb = !emb_w.w_emb.shape.empty() ? emb_w.w_emb.shape[1] : 128;
    size_t vocab_size = !emb_w.w_emb.shape.empty() ? emb_w.w_emb.shape[0] : 32768;

    Tensor combined({1, d_model + d_emb});
    std::memcpy(&combined.data[0], &final_normed_h.data[final_normed_h.numel() - d_model], d_model * sizeof(float));

    if (next_token_id < 0 || (size_t)next_token_id >= vocab_size) next_token_id = 0;
    std::memcpy(&combined.data[d_model], &emb_w.w_emb.data[next_token_id * d_emb], d_emb * sizeof(float));

    Tensor mtp_hidden;
    matmul_transB(combined, mtp_w.proj_weight, mtp_hidden);

    for (size_t i = 0; i < d_model; ++i) {
        mtp_hidden.data[i] += mtp_w.proj_bias.data[i];
    }

    dequantize_logits(mtp_logits, mtp_hidden, emb_w);
}

} // namespace maba

