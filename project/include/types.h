#pragma once

#include <vector>

struct BoundingBox {
    int x;       // top-left column
    int y;       // top-left row
    int width;
    int height;
    float score; // SVM decision value f(x) = w^T x + b
};

static constexpr int   HOG_WIN_W       = 64;
static constexpr int   HOG_WIN_H       = 64;
static constexpr int   HOG_CELL_SIZE   = 8;   // 8x8 pixels per cell
static constexpr int   HOG_BLOCK_SIZE  = 2;   // 2x2 cells per block
static constexpr int   HOG_NBINS       = 9;   // unsigned orientations 0-180 deg
static constexpr int   HOG_BLOCK_STRIDE= 1;   // blocks overlap by one cell (8px)
static constexpr int   HOG_CELLS_X     = HOG_WIN_W / HOG_CELL_SIZE;  // 8
static constexpr int   HOG_CELLS_Y     = HOG_WIN_H / HOG_CELL_SIZE;  // 8
static constexpr int   HOG_BLOCKS_X    = HOG_CELLS_X - HOG_BLOCK_SIZE + 1; // 7
static constexpr int   HOG_BLOCKS_Y    = HOG_CELLS_Y - HOG_BLOCK_SIZE + 1; // 7
static constexpr int   HOG_FEAT_DIM    =
    HOG_BLOCKS_X * HOG_BLOCKS_Y * HOG_BLOCK_SIZE * HOG_BLOCK_SIZE * HOG_NBINS; // 1764

static constexpr int   SLIDE_STRIDE    = 8;   // pixels between window positions
static constexpr float NMS_IOU_THRESH  = 0.4f;
