#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "types.h"
#include "svm.h"

/**
 * Run the sequential sliding window detector on a single frame.
 *
 * @param frame      BGR or grayscale OpenCV Mat (any size >= 64x64).
 * @param svm        Trained LinearSVM.
 * @param threshold  SVM decision threshold (default 0.0 — any positive score).
 * @return           Raw detections before NMS.
 */
std::vector<BoundingBox> sliding_window(const cv::Mat& frame,
                                         const LinearSVM& svm,
                                         float threshold = 0.0f);

/**
 * OpenMP-parallelised variant — same interface, same output.
 * Uses collapse(2) over the (y, x) grid and per-thread detection vectors
 * to avoid any shared-memory synchronisation inside the hot loop.
 */
std::vector<BoundingBox> sliding_window_omp(const cv::Mat& frame,
                                             const LinearSVM& svm,
                                             float threshold = 0.0f);
