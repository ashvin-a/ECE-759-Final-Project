#pragma once

#include <vector>
#include "types.h"

/**
 * Extract a 1764-dimensional HOG descriptor from a 64x64 grayscale patch.
 *
 * @param gray   Pointer to row-major float32 pixel data, shape [WIN_H x WIN_W].
 *               Values in [0, 255]. Stride is exactly WIN_W (no padding).
 * @param feat   Output buffer of length HOG_FEAT_DIM (1764). Caller allocates.
 */
void hog_extract(const float* gray, float* feat);

/**
 * Convert a BGR OpenCV patch (CV_8UC3) to a float32 grayscale buffer.
 * Uses the same coefficients as the Python pipeline:
 *   gray = 0.114*B + 0.587*G + 0.299*R
 *
 * @param bgr    Input: row-major uint8 BGR, shape [WIN_H x WIN_W x 3]
 * @param gray   Output: float32, shape [WIN_H x WIN_W]
 */
void bgr_to_gray(const unsigned char* bgr, float* gray, int width, int height);
