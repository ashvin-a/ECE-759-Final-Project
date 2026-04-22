// Host-side wrapper for the CUDA sliding window detector.
//
// All OpenCV includes live here so that the CUDA compilation unit
// (sliding_window_cuda.cu) never sees opencv headers — nvcc 11.5 + GCC 11
// cannot parse the std::function usage inside GCC 11's C++ standard library
// headers that OpenCV transitively includes.

#include "sliding_window_cuda.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "svm.h"
#include "types.h"

// Forward declaration of the CUDA-side implementation (defined in
// sliding_window_cuda.cu).  It receives a pre-converted float32 grayscale
// buffer to avoid any OpenCV dependency in the .cu translation unit.
std::vector<BoundingBox> sliding_window_cuda_impl(const float* gray_f32,
                                                   int img_w, int img_h,
                                                   const LinearSVM& svm,
                                                   float threshold);

std::vector<BoundingBox> sliding_window_cuda(const cv::Mat& frame,
                                              const LinearSVM& svm,
                                              float threshold)
{
    // Convert to single-channel float32 on the CPU.
    // The CUDA timer in main.cpp wraps the call to this function, so this
    // preprocessing is included in the CUDA latency measurement — which is
    // intentional: it is part of the GPU pipeline's CPU-side setup.
    cv::Mat gray_mat;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray_mat, cv::COLOR_BGR2GRAY);
    else
        gray_mat = frame;

    cv::Mat gray_f;
    gray_mat.convertTo(gray_f, CV_32F);
    if (!gray_f.isContinuous()) gray_f = gray_f.clone();

    return sliding_window_cuda_impl(gray_f.ptr<float>(),
                                    frame.cols, frame.rows,
                                    svm, threshold);
}
