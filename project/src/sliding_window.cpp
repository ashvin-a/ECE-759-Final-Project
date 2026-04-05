#include "sliding_window.h"
#include "hog.h"
#include "types.h"

#include <opencv2/imgproc.hpp>

std::vector<BoundingBox> sliding_window(const cv::Mat& frame,
                                         const LinearSVM& svm,
                                         float threshold)
{
    const int img_h = frame.rows;
    const int img_w = frame.cols;

    // Convert frame to float32 grayscale once for the entire image
    cv::Mat gray_mat;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray_mat, cv::COLOR_BGR2GRAY);
    else
        gray_mat = frame.clone();

    cv::Mat gray_f;
    gray_mat.convertTo(gray_f, CV_32F);

    std::vector<BoundingBox> detections;

    // Window-local buffers
    float patch_gray[HOG_WIN_H * HOG_WIN_W];
    float feat[HOG_FEAT_DIM];

    for (int y = 0; y <= img_h - HOG_WIN_H; y += SLIDE_STRIDE) {
        for (int x = 0; x <= img_w - HOG_WIN_W; x += SLIDE_STRIDE) {
            // Copy patch into contiguous buffer (OpenCV rows may have padding)
            for (int r = 0; r < HOG_WIN_H; ++r) {
                const float* src = gray_f.ptr<float>(y + r) + x;
                float*       dst = patch_gray + r * HOG_WIN_W;
                for (int c = 0; c < HOG_WIN_W; ++c)
                    dst[c] = src[c];
            }

            hog_extract(patch_gray, feat);
            float score = svm.predict(feat);

            if (score > threshold) {
                detections.push_back({x, y, HOG_WIN_W, HOG_WIN_H, score});
            }
        }
    }

    return detections;
}
