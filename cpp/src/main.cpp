#include "model.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "maba_weights.bin";

    maba::Model model;
    if (!model.load_weights(model_path)) {
        std::cerr << "Failed to load: " << model_path << std::endl;
        return 1;
    }

    std::vector<int> prompt = {1, 260, 261, 262, 263, 264, 265, 266};
    maba::Tensor logits, normed_h;

    auto t0 = std::chrono::high_resolution_clock::now();
    model.forward_sequence(prompt, logits, normed_h);
    auto t1 = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Prefill (" << prompt.size() << " toks): " << std::fixed << std::setprecision(2) << prefill_ms << " ms" << std::endl;

    size_t last_pos = prompt.size() - 1;
    size_t vocab_size = model.vocab_size;
    float max_p_logit = -1e9f;
    int cur_token = 0;
    for (size_t v = 0; v < vocab_size; ++v) {
        float val = logits.at(last_pos, v);
        if (val > max_p_logit) {
            max_p_logit = val;
            cur_token = (int)v;
        }
    }

    int gen_steps = 10;
    auto tg0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < gen_steps; ++s) {
        std::vector<int> step_tok = {cur_token};
        model.forward_sequence(step_tok, logits, normed_h, prompt.size() + s);

        float max_logit = -1e9f;
        int best_token = 0;
        for (size_t v = 0; v < vocab_size; ++v) {
            float val = logits.at(0, v);
            if (val > max_logit) {
                max_logit = val;
                best_token = (int)v;
            }
        }
        cur_token = best_token;
    }
    auto tg1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    double tok_per_sec = (gen_steps / (gen_ms / 1000.0));

    std::cout << "Decode (" << gen_steps << " toks): " << gen_ms << " ms (" << tok_per_sec << " tok/s)" << std::endl;
    return 0;
}
