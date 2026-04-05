#include "sliding_window.h"
#include "hog.h"
#include "types.h"

#include <omp.h>
#include <opencv2/imgproc.hpp>

std::vector<BoundingBox> sliding_window_omp(const cv::Mat& frame,
                                             const LinearSVM& svm,
                                             float threshold)
{
    const int img_h = frame.rows;
    const int img_w = frame.cols;

    // Grayscale conversion is done once outside the parallel region.
    cv::Mat gray_mat;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray_mat, cv::COLOR_BGR2GRAY);
    else
        gray_mat = frame.clone();

    cv::Mat gray_f;
    gray_mat.convertTo(gray_f, CV_32F);

    // Pre-compute grid dimensions so collapse(2) sees a perfectly rectangular
    // iteration space (both loop bounds are loop-invariant constants).
    const int n_y = (img_h - HOG_WIN_H) / SLIDE_STRIDE + 1;
    const int n_x = (img_w - HOG_WIN_W) / SLIDE_STRIDE + 1;

    // Allocate one detection vector per thread to avoid any shared-memory
    // synchronization inside the hot loop (no omp critical needed).
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<BoundingBox>> thread_dets(n_threads);

    // patch_gray and feat are declared inside the loop body so they are
    // implicitly private (stack-allocated per iteration, per thread).
    // hog_extract uses static thread_local scratch buffers, so it is safe.
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int iy = 0; iy < n_y; ++iy) {
        for (int ix = 0; ix < n_x; ++ix) {
            const int y = iy * SLIDE_STRIDE;
            const int x = ix * SLIDE_STRIDE;

            float patch_gray[HOG_WIN_H * HOG_WIN_W];
            float feat[HOG_FEAT_DIM];

            // Copy this window into a contiguous buffer.
            for (int r = 0; r < HOG_WIN_H; ++r) {
                const float* src = gray_f.ptr<float>(y + r) + x;
                float*       dst = patch_gray + r * HOG_WIN_W;
                for (int c = 0; c < HOG_WIN_W; ++c)
                    dst[c] = src[c];
            }

            hog_extract(patch_gray, feat);
            const float score = svm.predict(feat);

            if (score > threshold) {
                thread_dets[omp_get_thread_num()].push_back(
                    {x, y, HOG_WIN_W, HOG_WIN_H, score});
            }
        }
    }

    // Sequential merge of thread-local vectors — happens once per frame.
    std::vector<BoundingBox> detections;
    for (auto& v : thread_dets)
        detections.insert(detections.end(), v.begin(), v.end());

    return detections;
}
