#include "hog.h"
#include "types.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// Constants derived from types.h
static constexpr int W        = HOG_WIN_W;         // 64
static constexpr int H        = HOG_WIN_H;         // 64
static constexpr int CS       = HOG_CELL_SIZE;     // 8
static constexpr int BS       = HOG_BLOCK_SIZE;    // 2  (cells)
static constexpr int NB       = HOG_NBINS;         // 9
static constexpr int CX       = HOG_CELLS_X;       // 8
static constexpr int CY       = HOG_CELLS_Y;       // 8
static constexpr int BX       = HOG_BLOCKS_X;      // 7
static constexpr int BY       = HOG_BLOCKS_Y;      // 7
static constexpr float BIN_W  = 180.0f / NB;       // 20 deg / bin
static constexpr float L2_EPS = 1e-6f;
static constexpr float HYS_CLIP = 0.2f;

// Grayscale conversion
void bgr_to_gray(const unsigned char* bgr, float* gray, int width, int height)
{
    for (int i = 0; i < height * width; ++i) {
        const unsigned char* p = bgr + i * 3;
        gray[i] = 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
    }
}

// HOG extraction
void hog_extract(const float* gray, float* feat)
{
    // Step 1: Compute per-pixel gradients ([-1,0,1] kernel, zero-pad borders)
    // We store magnitude and orientation as flat arrays indexed [row*W + col].
    static thread_local float mag[H * W];
    static thread_local float ori[H * W];  // unsigned degrees [0, 180)

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            // Horizontal gradient
            float left  = (c > 0)     ? gray[r * W + (c - 1)] : gray[r * W + c];
            float right = (c < W - 1) ? gray[r * W + (c + 1)] : gray[r * W + c];
            float gx    = right - left;

            // Vertical gradient
            float up    = (r > 0)     ? gray[(r - 1) * W + c] : gray[r * W + c];
            float down  = (r < H - 1) ? gray[(r + 1) * W + c] : gray[r * W + c];
            float gy    = down - up;

            mag[r * W + c] = std::sqrt(gx * gx + gy * gy);

            // Unsigned orientation: atan2(gy, gx) mapped to [0, 180)
            float angle = std::atan2(gy, gx) * (180.0f / M_PI); // [-180, 180]
            if (angle < 0.0f) angle += 180.0f;
            if (angle >= 180.0f) angle -= 180.0f;
            ori[r * W + c] = angle;
        }
    }

    // Step 2: Build per-cell 9-bin histograms with bilinear bin interpolation
    // Layout: hists[cy][cx][bin]
    static thread_local float hists[CY][CX][NB];
    std::memset(hists, 0, sizeof(hists));

    for (int r = 0; r < H; ++r) {
        int cy = r / CS;
        for (int c = 0; c < W; ++c) {
            int   cx    = c / CS;
            float m     = mag[r * W + c];
            float angle = ori[r * W + c];

            float bin_f  = angle / BIN_W;           // fractional bin index
            int   bin_lo = static_cast<int>(bin_f) % NB;
            int   bin_hi = (bin_lo + 1) % NB;
            float alpha  = bin_f - std::floor(bin_f); // weight for hi bin

            hists[cy][cx][bin_lo] += m * (1.0f - alpha);
            hists[cy][cx][bin_hi] += m * alpha;
        }
    }

    // Step 3: Assemble blocks with L2-Hys normalization
    // Scan blocks in row-major order: by in [0, BY), bx in [0, BX)
    float* out = feat;
    for (int by = 0; by < BY; ++by) {
        for (int bx = 0; bx < BX; ++bx) {
            // Concatenate 2x2 cells → 36-element raw block vector
            float block[BS * BS * NB];
            int idx = 0;
            for (int dy = 0; dy < BS; ++dy)
                for (int dx = 0; dx < BS; ++dx)
                    for (int k = 0; k < NB; ++k)
                        block[idx++] = hists[by + dy][bx + dx][k];

            // L2-Hys: normalize → clip → renormalize
            float sum_sq = 0.0f;
            for (int i = 0; i < BS * BS * NB; ++i)
                sum_sq += block[i] * block[i];
            float norm = 1.0f / std::sqrt(sum_sq + L2_EPS * L2_EPS);
            for (int i = 0; i < BS * BS * NB; ++i)
                block[i] = std::min(block[i] * norm, HYS_CLIP);

            sum_sq = 0.0f;
            for (int i = 0; i < BS * BS * NB; ++i)
                sum_sq += block[i] * block[i];
            float norm2 = 1.0f / std::sqrt(sum_sq + L2_EPS * L2_EPS);
            for (int i = 0; i < BS * BS * NB; ++i)
                *out++ = block[i] * norm2;
        }
    }
    // out - feat should equal HOG_FEAT_DIM (1764) here
}
