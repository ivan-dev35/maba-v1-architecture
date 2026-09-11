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
            gdn_state[p] = GDN2State();
            gqa_cache[p] = GQACache();
        }
    }
};

using BlockRuntimeState = BlockState;

class Model {
public:
    EmbW emb_w;
    std::vector<BlockW> blocks;
    Tensor final_norm;
    MTPW mtp_w;
    std::vector<BlockState> block_states;

    Model() {
        blocks.resize(20);
        block_states.resize(20);
        for (size_t i = 0; i < 20; ++i) {
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

        uint32_t magic, vocab_size, d_emb, d_model, num_layers, d_ffn, num_heads, num_kv_heads, num_tensors;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&vocab_size), 4);
        f.read(reinterpret_cast<char*>(&d_emb), 4);
        f.read(reinterpret_cast<char*>(&d_model), 4);
        f.read(reinterpret_cast<char*>(&num_layers), 4);
        f.read(reinterpret_cast<char*>(&d_ffn), 4);
        f.read(reinterpret_cast<char*>(&num_heads), 4);
        f.read(reinterpret_cast<char*>(&num_kv_heads), 4);
        f.read(reinterpret_cast<char*>(&num_tensors), 4);

        if (magic != 0x4D414241 && magic != 0x41504558) {
            std::cerr << "Invalid binary magic number: 0x" << std::hex << magic << std::endl;
            return false;
        }

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

        emb_w.w_emb = tensors["embeddings.w_emb.weight"];
        emb_w.w_proj_in = tensors["embeddings.w_proj_in.weight"];
        emb_w.w_proj_out = tensors["embeddings.w_proj_out.weight"];
        final_norm = tensors["final_norm.weight"];
        mtp_w.proj_weight = tensors["mtp_head.proj.weight"];
        mtp_w.proj_bias = tensors["mtp_head.proj.bias"];

        for (size_t l = 0; l < 20; ++l) {
            std::string pfx = "layers." + std::to_string(l) + ".";
            blocks[l].input_norm = tensors[pfx + "input_norm.weight"];
            blocks[l].post_attn_norm = tensors[pfx + "post_attn_norm.weight"];
            blocks[l].attn_gate = tensors[pfx + "attn_gate.g_res"];
            blocks[l].ffn_norm = tensors[pfx + "ffn_norm.weight"];
            blocks[l].post_ffn_norm = tensors[pfx + "post_ffn_norm.weight"];
            blocks[l].ffn_gate = tensors[pfx + "ffn_gate.g_res"];

            blocks[l].swiglu_w.w_gate = tensors[pfx + "ffn.w_gate.weight"];
            blocks[l].swiglu_w.w_up = tensors[pfx + "ffn.w_up.weight"];
            blocks[l].swiglu_w.w_down = tensors[pfx + "ffn.w_down.weight"];

            if (blocks[l].is_gqa) {
                blocks[l].gqa_w.q_proj = tensors[pfx + "mixer.q_proj.weight"];
                blocks[l].gqa_w.k_proj = tensors[pfx + "mixer.k_proj.weight"];
                blocks[l].gqa_w.v_proj = tensors[pfx + "mixer.v_proj.weight"];
                blocks[l].gqa_w.o_proj = tensors[pfx + "mixer.o_proj.weight"];
                blocks[l].gqa_w.q_norm = tensors[pfx + "mixer.q_norm.weight"];
                blocks[l].gqa_w.k_norm = tensors[pfx + "mixer.k_norm.weight"];
            } else {
                blocks[l].gdn2_w.q_proj = tensors[pfx + "mixer.q_proj.weight"];
                blocks[l].gdn2_w.k_proj = tensors[pfx + "mixer.k_proj.weight"];
                blocks[l].gdn2_w.v_proj = tensors[pfx + "mixer.v_proj.weight"];
                blocks[l].gdn2_w.o_proj = tensors[pfx + "mixer.o_proj.weight"];
                blocks[l].gdn2_w.conv_q = tensors[pfx + "mixer.conv_q.weight"];
                blocks[l].gdn2_w.conv_k = tensors[pfx + "mixer.conv_k.weight"];
                blocks[l].gdn2_w.conv_v = tensors[pfx + "mixer.conv_v.weight"];
                blocks[l].gdn2_w.gate_alpha = tensors[pfx + "mixer.gate_alpha.weight"];
                blocks[l].gdn2_w.gate_erase = tensors[pfx + "mixer.gate_erase.weight"];
                blocks[l].gdn2_w.gate_write = tensors[pfx + "mixer.gate_write.weight"];
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
        Tensor h;
        factorized_embed(h, token_ids, emb_w);

        for (size_t l = 0; l < 20; ++l) {
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

} // namespace maba

