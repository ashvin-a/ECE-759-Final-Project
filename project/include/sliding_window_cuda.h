#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "types.h"
#include "svm.h"

/**
 * CUDA-parallelised sliding window detector.
 *
 * Maps the entire sliding window grid onto the GPU in one launch:
 *   HOG kernel  — gridDim (n_win_x, n_win_y, 49), blockDim 256
 *   SVM kernel  — gridDim (n_win_x, n_win_y),     blockDim 256
 *
 * Device memory for weights and frame buffers is allocated lazily on the
 * first call and reused across frames (no per-frame cudaMalloc/Free).
 *
 * @param frame     BGR or grayscale OpenCV Mat (any size >= 64x64).
 * @param svm       Trained LinearSVM (weights uploaded to GPU on first call).
 * @param threshold SVM decision threshold (default 0.0).
 * @return          Raw detections before NMS.
 */
std::vector<BoundingBox> sliding_window_cuda(const cv::Mat& frame,
                                              const LinearSVM& svm,
                                              float threshold = 0.0f);
