#pragma once

#include <string>
#include <vector>
#include "types.h"

/**
 * Linear SVM inference.
 * Loads weights from a binary float32 file and a bias from a text file.
 */
class LinearSVM {
public:
    /**
     * @param weights_path  Path to weights.bin (raw float32, HOG_FEAT_DIM values)
     * @param bias_path     Path to bias.txt (single float)
     * @throws std::runtime_error on load failure or dimension mismatch
     */
    LinearSVM(const std::string& weights_path, const std::string& bias_path);

    /**
     * Compute decision value f(x) = w^T * x + b.
     * @param feat  Pointer to HOG_FEAT_DIM float32 values.
     * @return      Signed decision scalar. Positive = rock.
     */
    float predict(const float* feat) const;

    const float* weights() const { return weights_.data(); }
    float        bias()    const { return bias_; }

private:
    std::vector<float> weights_;
    float              bias_;
};
