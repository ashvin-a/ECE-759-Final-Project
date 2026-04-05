#include "svm.h"
#include "types.h"

#include <fstream>
#include <stdexcept>
#include <string>

LinearSVM::LinearSVM(const std::string& weights_path, const std::string& bias_path)
{
    // Load weights
    std::ifstream wf(weights_path, std::ios::binary | std::ios::ate);
    if (!wf.is_open())
        throw std::runtime_error("Cannot open weights file: " + weights_path);

    std::streamsize bytes = wf.tellg();
    wf.seekg(0, std::ios::beg);

    int n_weights = static_cast<int>(bytes / sizeof(float));
    if (n_weights != HOG_FEAT_DIM) {
        throw std::runtime_error(
            "Weight dimension mismatch: file has " + std::to_string(n_weights) +
            " floats, expected " + std::to_string(HOG_FEAT_DIM));
    }

    weights_.resize(HOG_FEAT_DIM);
    wf.read(reinterpret_cast<char*>(weights_.data()), bytes);
    if (!wf)
        throw std::runtime_error("Failed to read weights from: " + weights_path);

    // Load bias
    std::ifstream bf(bias_path);
    if (!bf.is_open())
        throw std::runtime_error("Cannot open bias file: " + bias_path);
    if (!(bf >> bias_))
        throw std::runtime_error("Failed to parse bias from: " + bias_path);
}

float LinearSVM::predict(const float* feat) const
{
    float dot = 0.0f;
    const float* w = weights_.data();
    for (int i = 0; i < HOG_FEAT_DIM; ++i)
        dot += w[i] * feat[i];
    return dot + bias_;
}
