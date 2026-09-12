#include "model.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdint>

inline std::string find_file(const std::string& name) {
    std::ifstream f(name);
    if (f.good()) return name;
    std::string p1 = "../" + name;
    std::ifstream f1(p1);
    if (f1.good()) return p1;
    std::string p2 = "../../" + name;
    std::ifstream f2(p2);
    if (f2.good()) return p2;
    return name;
}

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : find_file("maba_ref.bin");
    std::string ref_logits_path = (argc > 2) ? argv[2] : find_file("ref_logits.bin");

    maba::Model model;
    if (!model.load_weights(model_path)) {
        std::cerr << "Failed to load " << model_path << std::endl;
        return 1;
    }

    std::vector<int> tokens = {1, 100, 200, 300};
    maba::Tensor logits, normed_h;
    model.forward_sequence(tokens, logits, normed_h);

    std::ifstream f_ref(ref_logits_path, std::ios::binary);
    if (!f_ref.is_open()) {
        std::cerr << "Cannot open " << ref_logits_path << std::endl;
        return 1;
    }

    uint32_t ref_L, ref_V;
    f_ref.read(reinterpret_cast<char*>(&ref_L), 4);
    f_ref.read(reinterpret_cast<char*>(&ref_V), 4);

    assert(ref_L == tokens.size());
    assert(ref_V == model.vocab_size);

    std::vector<float> ref_data(ref_L * ref_V);
    f_ref.read(reinterpret_cast<char*>(ref_data.data()), ref_data.size() * sizeof(float));

    float max_diff = 0.0f;
    double sum_diff = 0.0;
    size_t count = ref_data.size();

    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(logits.data[i] - ref_data[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }

    float mean_diff = (float)(sum_diff / count);
    std::cout << "Compared " << count << " logits: max_diff=" << max_diff << ", mean_diff=" << mean_diff << std::endl;

    if (max_diff < 1e-2f) {
        std::cout << "PASS: numerical equivalence verified" << std::endl;
        return 0;
    }
    std::cerr << "FAIL: max_diff exceeds tolerance" << std::endl;
    return 1;
}
