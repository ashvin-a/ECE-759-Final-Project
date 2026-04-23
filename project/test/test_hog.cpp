/**
 * test_hog.cpp — Correctness validation: C++ HOG vs. Python reference.
 *
 * Usage:
 *   ./build/test_hog <patch.png> <ref_feat.bin> [weights.bin bias.txt]
 *
 *   patch.png     : 64x64 BGR test image
 *   ref_feat.bin  : 1764 float32 values written by Python hog_utils.extract_hog()
 *   weights.bin   : (optional) run SVM and compare decision values
 *   bias.txt      : (optional) required if weights.bin is provided
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "hog.h"
#include "svm.h"
#include "types.h"

static std::vector<float> load_bin(const std::string& path, int expected_n)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
    std::cerr << "Cannot open: " << path << "\n";
        std::exit(1);
    }
    int n = static_cast<int>(f.tellg() / sizeof(float));
    if (n != expected_n) {
        std::fprintf(stderr,
            "ERROR: %s has %d floats, expected %d\n",
            path.c_str(), n, expected_n);
        std::exit(1);
    }
    f.seekg(0);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <patch.png> <ref_feat.bin> [weights.bin bias.txt]\n",
            argv[0]);
        return 1;
    }

    const std::string patch_path = argv[1];
    const std::string ref_path   = argv[2];

    //  Load test patch 
    cv::Mat img = cv::imread(patch_path);
    if (img.empty()) {
        std::cerr << "Cannot read patch: " << patch_path << "\n";
        return 1;
    }
    if (img.rows != HOG_WIN_H || img.cols != HOG_WIN_W) {
        std::fprintf(stderr,
            "ERROR: patch must be %dx%d, got %dx%d\n",
            HOG_WIN_W, HOG_WIN_H, img.cols, img.rows);
        return 1;
    }

    //  Convert BGR → float gray (matching Python coefficients) 
    float gray[HOG_WIN_H * HOG_WIN_W];
    bgr_to_gray(img.data, gray, HOG_WIN_W, HOG_WIN_H);

    //  Run C++ HOG 
    float cpp_feat[HOG_FEAT_DIM];
    hog_extract(gray, cpp_feat);

    //  Load Python reference 
    std::vector<float> ref = load_bin(ref_path, HOG_FEAT_DIM);

    //  Compare 
    float max_abs_diff = 0.0f;
    float sum_sq_diff  = 0.0f;
    int   worst_idx    = 0;

    for (int i = 0; i < HOG_FEAT_DIM; ++i) {
        float d = std::abs(cpp_feat[i] - ref[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
            worst_idx    = i;
        }
        sum_sq_diff += d * d;
    }
    float rmse = std::sqrt(sum_sq_diff / HOG_FEAT_DIM);

    // Empirical tolerances for float32 after L2-Hys normalization
    constexpr float TIGHT_TOL = 1e-3f;
    constexpr float LOOSE_TOL = 1e-2f;

    std::printf("\n── HOG Correctness Report ─────────────────\n");
    std::printf("  Feature dim     : %d\n", HOG_FEAT_DIM);
    std::printf("  Max |diff|      : %.6e  (at index %d)\n", max_abs_diff, worst_idx);
    std::printf("  RMSE            : %.6e\n", rmse);
    std::printf("  C++ feat[%d]  : %.6f\n", worst_idx, cpp_feat[worst_idx]);
    std::printf("  Py  feat[%d]  : %.6f\n", worst_idx, ref[worst_idx]);

    bool tight_pass = (max_abs_diff < TIGHT_TOL);
    bool loose_pass = (max_abs_diff < LOOSE_TOL);

    std::printf("\n  [%s] max_abs_diff < 1e-3 (tight)\n", tight_pass ? "PASS" : "FAIL");
    std::printf("  [%s] max_abs_diff < 1e-2 (loose)\n", loose_pass ? "PASS" : "FAIL");

    // Optional SVM decision check
    if (argc >= 5) {
        const std::string weights_path = argv[3];
        const std::string bias_path    = argv[4];
        try {
            LinearSVM svm(weights_path, bias_path);
            float f_cpp = svm.predict(cpp_feat);
            float f_ref = svm.predict(ref.data());
            float f_diff = std::abs(f_cpp - f_ref);
            bool same_sign = ((f_cpp >= 0) == (f_ref >= 0));

            std::printf("\n── SVM Decision Check ───────────────────────────────\n");
            std::printf("  f(x) C++    : %+.6f  → %s\n",
                        f_cpp, f_cpp > 0 ? "ROCK" : "background");
            std::printf("  f(x) Python : %+.6f  → %s\n",
                        f_ref, f_ref > 0 ? "ROCK" : "background");
            std::printf("  |f_cpp - f_ref| : %.6e\n", f_diff);
            std::printf("  [%s] Same sign (functionally correct)\n",
                        same_sign ? "PASS" : "FAIL");
        } catch (const std::exception& e) {
            std::cerr << "SVM load error: " << e.what() << "\n";
        }
    }

    std::printf("────────────────────────────────────────────────────\n\n");
    return loose_pass ? 0 : 1;
}
