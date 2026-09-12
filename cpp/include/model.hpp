#pragma once

#include "tensor.hpp"
#include "embeddings.hpp"
#include "gdn2.hpp"
#include "gqa.hpp"
#include "swiglu.hpp"
#include "mtp.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <cstdint>

namespace maba {

struct BlockW {
    bool is_gqa = false;
    Tensor input_norm;
    Tensor post_attn_norm;
    Tensor attn_gate;
    Tensor ffn_norm;
    Tensor post_ffn_norm;
    Tensor ffn_gate;
    GDN2W gdn2_w;
    GQAW gqa_w;
    FFNW swiglu_w;
};

using BlockWeights = BlockW;

struct BlockState {
    GDN2State gdn_state[2];
    GQACache gqa_cache[2];

    void reset() {
        for (int p = 0; p < 2; ++p) {
            gdn_state[p].reset();
            gqa_cache[p] = GQACache();
        }
    }
};

using BlockRuntimeState = BlockState;

class Model {
public:
    uint32_t vocab_size = 32768;
    uint32_t dim = 640;
    uint32_t d_emb = 128;
    uint32_t d_head = 64;
    uint32_t num_heads = 10;
    uint32_t num_kv_heads = 2;
    uint32_t num_layers = 20;
    uint32_t d_ffn = 1728;

    EmbW emb_w;
    std::vector<BlockW> blocks;
    Tensor final_norm;
    MTPW mtp_w;
    std::vector<BlockState> block_states;

    Model() {
        blocks.resize(num_layers);
        block_states.resize(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            blocks[i].is_gqa = ((i % 4) == 3);
        }
    }

    void reset_state() {
        for (auto& s : block_states) {
            s.reset();
        }
    }

    bool load_weights(const std::string& bin_path) {
        std::ifstream f(bin_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Failed to open weights file: " << bin_path << std::endl;
            return false;
        }

        uint32_t magic, vocab_sz, model_dim, model_d_emb, head_dim, n_h, n_kv_h, n_l, ffn_dim, num_tensors;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&vocab_sz), 4);
        f.read(reinterpret_cast<char*>(&model_dim), 4);
        f.read(reinterpret_cast<char*>(&model_d_emb), 4);
        f.read(reinterpret_cast<char*>(&head_dim), 4);
        f.read(reinterpret_cast<char*>(&n_h), 4);
        f.read(reinterpret_cast<char*>(&n_kv_h), 4);
        f.read(reinterpret_cast<char*>(&n_l), 4);
        f.read(reinterpret_cast<char*>(&ffn_dim), 4);
        f.read(reinterpret_cast<char*>(&num_tensors), 4);

        if (magic != 0x4D414241 && magic != 0x41504558) {
            std::cerr << "Invalid binary magic number: 0x" << std::hex << magic << std::endl;
            return false;
        }

        vocab_size = vocab_sz;
        dim = model_dim;
        d_emb = model_d_emb;
        d_head = head_dim;
        num_heads = n_h;
        num_kv_heads = n_kv_h;
        num_layers = n_l;
        d_ffn = ffn_dim;

        blocks.resize(num_layers);
        block_states.resize(num_layers);
        reset_state();

        std::map<std::string, Tensor> tensors;
        for (uint32_t i = 0; i < num_tensors; ++i) {
            uint32_t name_len;
            f.read(reinterpret_cast<char*>(&name_len), 4);
            std::string name(name_len, '\0');
            f.read(&name[0], name_len);

            uint32_t ndim;
            f.read(reinterpret_cast<char*>(&ndim), 4);
            std::vector<size_t> shape(ndim);
            size_t total_numel = 1;
            for (uint32_t d = 0; d < ndim; ++d) {
                uint32_t dim_val;
                f.read(reinterpret_cast<char*>(&dim_val), 4);
                shape[d] = dim_val;
                total_numel *= dim_val;
            }

            Tensor t(shape);
            f.read(reinterpret_cast<char*>(t.data.data()), total_numel * sizeof(float));
            tensors[name] = std::move(t);
        }

        emb_w.w_emb = std::move(tensors["embeddings.w_emb.weight"]);
        emb_w.w_proj_in = std::move(tensors["embeddings.w_proj_in.weight"]);
        emb_w.w_proj_out = std::move(tensors["embeddings.w_proj_out.weight"]);
        final_norm = std::move(tensors["final_norm.weight"]);
        mtp_w.proj_weight = std::move(tensors["mtp_head.proj.weight"]);
        mtp_w.proj_bias = std::move(tensors["mtp_head.proj.bias"]);

        for (size_t l = 0; l < num_layers; ++l) {
            std::string pfx = "layers." + std::to_string(l) + ".";
            blocks[l].is_gqa = (tensors.find(pfx + "mixer.q_norm.weight") != tensors.end());

            blocks[l].input_norm = std::move(tensors[pfx + "input_norm.weight"]);
            blocks[l].post_attn_norm = std::move(tensors[pfx + "post_attn_norm.weight"]);
            blocks[l].attn_gate = std::move(tensors[pfx + "attn_gate.g_res"]);
            blocks[l].ffn_norm = std::move(tensors[pfx + "ffn_norm.weight"]);
            blocks[l].post_ffn_norm = std::move(tensors[pfx + "post_ffn_norm.weight"]);
            blocks[l].ffn_gate = std::move(tensors[pfx + "ffn_gate.g_res"]);

            blocks[l].swiglu_w.w_gate = std::move(tensors[pfx + "ffn.w_gate.weight"]);
            blocks[l].swiglu_w.w_up = std::move(tensors[pfx + "ffn.w_up.weight"]);
            blocks[l].swiglu_w.w_down = std::move(tensors[pfx + "ffn.w_down.weight"]);

            if (blocks[l].is_gqa) {
                blocks[l].gqa_w.q_proj = std::move(tensors[pfx + "mixer.q_proj.weight"]);
                blocks[l].gqa_w.k_proj = std::move(tensors[pfx + "mixer.k_proj.weight"]);
                blocks[l].gqa_w.v_proj = std::move(tensors[pfx + "mixer.v_proj.weight"]);
                blocks[l].gqa_w.o_proj = std::move(tensors[pfx + "mixer.o_proj.weight"]);
                blocks[l].gqa_w.q_norm = std::move(tensors[pfx + "mixer.q_norm.weight"]);
                blocks[l].gqa_w.k_norm = std::move(tensors[pfx + "mixer.k_norm.weight"]);
                blocks[l].gqa_w.H_q = num_heads;
                blocks[l].gqa_w.H_kv = num_kv_heads;
                blocks[l].gqa_w.d = d_head;
            } else {
                blocks[l].gdn2_w.q_proj = std::move(tensors[pfx + "mixer.q_proj.weight"]);
                blocks[l].gdn2_w.k_proj = std::move(tensors[pfx + "mixer.k_proj.weight"]);
                blocks[l].gdn2_w.v_proj = std::move(tensors[pfx + "mixer.v_proj.weight"]);
                blocks[l].gdn2_w.o_proj = std::move(tensors[pfx + "mixer.o_proj.weight"]);
                blocks[l].gdn2_w.conv_q = std::move(tensors[pfx + "mixer.conv_q.weight"]);
                blocks[l].gdn2_w.conv_k = std::move(tensors[pfx + "mixer.conv_k.weight"]);
                blocks[l].gdn2_w.conv_v = std::move(tensors[pfx + "mixer.conv_v.weight"]);
                blocks[l].gdn2_w.gate_alpha = std::move(tensors[pfx + "mixer.gate_alpha.weight"]);
                blocks[l].gdn2_w.gate_erase = std::move(tensors[pfx + "mixer.gate_erase.weight"]);
                blocks[l].gdn2_w.gate_write = std::move(tensors[pfx + "mixer.gate_write.weight"]);
                blocks[l].gdn2_w.H = num_heads;
                blocks[l].gdn2_w.d = d_head;
            }
        }

        std::cout << "Loaded weights from " << bin_path << " (" << tensors.size() << " tensors)." << std::endl;
        return true;
    }

    void forward_sequence(
        const std::vector<int>& token_ids,
        Tensor& logits,
        Tensor& normed_h,
        size_t start_pos = 0
    ) {
        if (token_ids.empty()) {
            size_t vocab_size = emb_w.w_emb.shape.empty() ? 0 : emb_w.w_emb.shape[0];
            size_t dim = emb_w.w_proj_in.shape.empty() ? 0 : emb_w.w_proj_in.shape[0];
            logits.resize(std::vector<size_t>{0, vocab_size});
            normed_h.resize(std::vector<size_t>{0, dim});
            return;
        }

        Tensor h;
        factorized_embed(h, token_ids, emb_w);

        for (size_t l = 0; l < blocks.size(); ++l) {
            auto& blk = blocks[l];
            auto& state = block_states[l];

            for (int pass = 0; pass < 2; ++pass) {
                Tensor normed_in;
                rmsnorm(normed_in, h, blk.input_norm);

                Tensor mixer_out;
                if (blk.is_gqa) {
                    gqa_fwd(mixer_out, normed_in, blk.gqa_w, state.gqa_cache[pass], start_pos);
                } else {
                    gdn2_fwd(mixer_out, normed_in, blk.gdn2_w, state.gdn_state[pass]);
                }

                Tensor post_mixer;
                rmsnorm(post_mixer, mixer_out, blk.post_attn_norm);

                Tensor h_after_mixer;
                gated_residual(h_after_mixer, h, post_mixer, blk.attn_gate);

                Tensor ffn_normed;
                rmsnorm(ffn_normed, h_after_mixer, blk.ffn_norm);

                Tensor ffn_out;
                ffn_fwd(ffn_out, ffn_normed, blk.swiglu_w);

                Tensor post_ffn;
                rmsnorm(post_ffn, ffn_out, blk.post_ffn_norm);

                gated_residual(h, h_after_mixer, post_ffn, blk.ffn_gate);
            }
        }

        rmsnorm(normed_h, h, final_norm);
        dequantize_logits(logits, normed_h, emb_w);
    }
};

using Apex100MModel = Model;
using MabaModel = Model;
using MabaLM = Model;

} // namespace maba

