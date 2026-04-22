// NOTE: No OpenCV headers here — nvcc 11.5 + GCC 11 cannot parse opencv's
// std::function usage under C++17.  All cv::Mat preprocessing lives in
// sliding_window_cuda_host.cpp (plain C++), which calls sliding_window_cuda_impl.
#include "types.h"
#include "svm.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

// ── Device-side HOG constants ────────────────────────────────────────────────
// Mirror types.h; plain macros so they are usable in __device__ code without
// linking the host constexpr symbols.
#define D_WIN_W      64
#define D_WIN_H      64
#define D_CELL       8        // pixels per cell side
#define D_NBINS      9        // unsigned orientation bins (0–180°)
#define D_BLOCKS_X   7        // HOG blocks horizontally in one window
#define D_BLOCKS_Y   7        // HOG blocks vertically   in one window
#define D_N_BLOCKS   49       // 7 × 7
#define D_BLOCK_FEAT 36       // 2×2 cells × 9 bins per HOG block
#define D_FEAT_DIM   1764     // 49 × 36
#define D_STRIDE     8        // sliding-window step (pixels)
#define D_BLK_PIX    16       // HOG block side in pixels (2 cells × 8 px)
#define D_TPB        256      // threads per CUDA block  (= 16 × 16)
#define D_NWARPS     8        // D_TPB / 32
#define D_L2_EPS_SQ  1e-12f  // L2-Hys epsilon² (matches CPU L2_EPS=1e-6)
#define D_HYS_CLIP   0.2f

// ── CUDA error-check helper ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── Persistent device buffers (allocated lazily, reused across frames) ───────
static float*       d_weights   = nullptr;
static bool         weights_on_device = false;

static float*       d_gray      = nullptr;
static int          d_gray_cap  = 0;       // capacity in floats

static float*       d_feat      = nullptr;
static int          d_feat_cap  = 0;       // capacity in floats

static BoundingBox* d_dets      = nullptr;
static int          d_dets_cap  = 0;       // capacity in BoundingBoxes

static int*         d_det_cnt   = nullptr; // atomic counter on device

// Grow a device buffer if the current capacity is insufficient.
static void ensure_buf(void** ptr, int* cap, int need, size_t elem_sz)
{
    if (*cap >= need) return;
    if (*ptr) CUDA_CHECK(cudaFree(*ptr));
    CUDA_CHECK(cudaMalloc(ptr, need * elem_sz));
    *cap = need;
}

// ── HOG kernel ───────────────────────────────────────────────────────────────
// Grid  : (n_win_x, n_win_y, 49)   — one CUDA block per (window, HOG-block)
// Block : (256)                     — one thread per pixel in the 16×16 region

__global__ void hog_kernel(
    const float* __restrict__ gray,   // float grayscale image, row-major
    int img_w, int img_h,
    float* __restrict__ feat_out,     // [n_win_y * n_win_x * D_FEAT_DIM]
    int n_win_x)
{
    const int win_ix  = blockIdx.x;
    const int win_iy  = blockIdx.y;
    const int hog_blk = blockIdx.z;               // 0..48

    const int hog_by  = hog_blk / D_BLOCKS_X;    // HOG block row (0..6)
    const int hog_bx  = hog_blk % D_BLOCKS_X;    // HOG block col (0..6)

    const int tid     = threadIdx.x;
    const int local_r = tid / D_BLK_PIX;          // 0..15
    const int local_c = tid % D_BLK_PIX;          // 0..15

    // Global pixel position this thread is responsible for
    const int gr = win_iy * D_STRIDE + hog_by * D_CELL + local_r;
    const int gc = win_ix * D_STRIDE + hog_bx * D_CELL + local_c;

    // ── Gradient + orientation ───────────────────────────────────────────
    float mag = 0.0f, bin_f = 0.0f;
    if (gr < img_h && gc < img_w) {
        float ctr   = gray[gr * img_w + gc];
        float left  = (gc > 0)         ? gray[ gr      * img_w + gc - 1] : ctr;
        float right = (gc < img_w - 1) ? gray[ gr      * img_w + gc + 1] : ctr;
        float up    = (gr > 0)         ? gray[(gr - 1)  * img_w + gc    ] : ctr;
        float down  = (gr < img_h - 1) ? gray[(gr + 1)  * img_w + gc    ] : ctr;

        float gx = right - left;
        float gy = down  - up;
        mag = sqrtf(gx * gx + gy * gy);

        float ang = atan2f(gy, gx) * (180.0f / 3.14159265358979f);
        if (ang <    0.0f) ang += 180.0f;
        if (ang >= 180.0f) ang -= 180.0f;
        bin_f = ang * ((float)D_NBINS / 180.0f);  // fractional bin index
    }

    // Bilinear bin interpolation
    const int   bin_lo  = (int)bin_f % D_NBINS;
    const int   bin_hi  = (bin_lo + 1) % D_NBINS;
    const float alpha   = bin_f - floorf(bin_f);
    const float vote_lo = mag * (1.0f - alpha);
    const float vote_hi = mag * alpha;

    // Which of the 4 cells within this 16×16 HOG block does this pixel belong to?
    const int cell_r  = local_r / D_CELL;         // 0 or 1
    const int cell_c  = local_c / D_CELL;         // 0 or 1
    const int cell_id = cell_r * 2 + cell_c;      // 0..3

    // ── Warp-private shared-memory histograms ───────────────────────────
    // 8 warps × 36 bins = 288 floats.  Each warp has exclusive slots so
    // atomicAdd within a warp can only conflict on the same bin — no
    // cross-warp serialisation.
    __shared__ float smem_hists[D_NWARPS * D_BLOCK_FEAT]; // 288 floats

    // Zero-initialise (loop handles 288 > 256 correctly)
    for (int i = tid; i < D_NWARPS * D_BLOCK_FEAT; i += D_TPB)
        smem_hists[i] = 0.0f;
    __syncthreads();

    const int warp_id  = tid / 32;
    const int base     = warp_id * D_BLOCK_FEAT + cell_id * D_NBINS;
    atomicAdd(&smem_hists[base + bin_lo], vote_lo);
    atomicAdd(&smem_hists[base + bin_hi], vote_hi);
    __syncthreads();

    // ── Reduce 8 warp histograms → 1 block histogram (36 elements) ──────
    __shared__ float block_hist[D_BLOCK_FEAT];
    if (tid < D_BLOCK_FEAT) {
        float s = 0.0f;
        for (int w = 0; w < D_NWARPS; ++w)
            s += smem_hists[w * D_BLOCK_FEAT + tid];
        block_hist[tid] = s;
    }
    __syncthreads();

    // ── L2-Hys normalisation ────────────────────────────────────────────
    // Thread 0 serially sums 36 values — trivially fast, avoids another
    // full parallel reduction for only 36 elements.
    __shared__ float sval;

    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < D_BLOCK_FEAT; ++i) s += block_hist[i] * block_hist[i];
        sval = s;
    }
    __syncthreads();

    // First pass: normalise + clip at 0.2
    if (tid < D_BLOCK_FEAT)
        block_hist[tid] = fminf(block_hist[tid] * rsqrtf(sval + D_L2_EPS_SQ), D_HYS_CLIP);
    __syncthreads();

    // Recompute sum for second normalisation
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < D_BLOCK_FEAT; ++i) s += block_hist[i] * block_hist[i];
        sval = s;
    }
    __syncthreads();

    // Write normalised values to global feature array
    if (tid < D_BLOCK_FEAT) {
        const int win_id  = win_iy * n_win_x + win_ix;
        const int out_idx = win_id * D_FEAT_DIM + hog_blk * D_BLOCK_FEAT + tid;
        feat_out[out_idx] = block_hist[tid] * rsqrtf(sval + D_L2_EPS_SQ);
    }
}

// ── SVM kernel ───────────────────────────────────────────────────────────────
// Grid  : (n_win_x, n_win_y)  — one CUDA block per window position
// Block : (256)
//
// Each block computes one dot product w^T x + b over 1764 features via a
// strided loop (ceil(1764/256)=7 elements/thread) followed by a standard
// parallel tree reduction in shared memory.  Positive detections are written
// to the output array using a single global atomicAdd counter.
__global__ void svm_kernel(
    const float* __restrict__ feat,     // [n_win_y * n_win_x * D_FEAT_DIM]
    const float* __restrict__ weights,  // [D_FEAT_DIM]
    float bias,
    float threshold,
    int   n_win_x,
    BoundingBox* __restrict__ det_out,
    int*         __restrict__ det_count,
    int max_dets)
{
    const int win_ix = blockIdx.x;
    const int win_iy = blockIdx.y;
    const int win_id = win_iy * n_win_x + win_ix;
    const int tid    = threadIdx.x;

    // Strided partial dot product
    const float* fp = feat + win_id * D_FEAT_DIM;
    float partial = 0.0f;
    for (int i = tid; i < D_FEAT_DIM; i += D_TPB)
        partial += fp[i] * weights[i];

    // Parallel tree reduction
    __shared__ float smem[D_TPB];
    smem[tid] = partial;
    __syncthreads();
    for (int s = D_TPB / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float score = smem[0] + bias;
        if (score > threshold) {
            int idx = atomicAdd(det_count, 1);
            if (idx < max_dets) {
                BoundingBox b;
                b.x      = win_ix * D_STRIDE;
                b.y      = win_iy * D_STRIDE;
                b.width  = D_WIN_W;
                b.height = D_WIN_H;
                b.score  = score;
                det_out[idx] = b;
            }
        }
    }
}

// ── Host entry point (internal) ───────────────────────────────────────────────
// Called by sliding_window_cuda_host.cpp after OpenCV preprocessing.
// gray_f32: row-major float32 grayscale image of size img_h * img_w.
std::vector<BoundingBox> sliding_window_cuda_impl(const float* gray_f32,
                                                   int img_w, int img_h,
                                                   const LinearSVM& svm,
                                                   float threshold)
{
    // Sliding window grid dimensions
    const int n_win_x  = (img_w - D_WIN_W) / D_STRIDE + 1;
    const int n_win_y  = (img_h - D_WIN_H) / D_STRIDE + 1;
    const int n_wins   = n_win_x * n_win_y;
    const int max_dets = n_wins;  // worst case: every window fires

    // ── Upload SVM weights once (weights pointer is stable for a given svm) ──
    if (!weights_on_device) {
        CUDA_CHECK(cudaMalloc(&d_weights, D_FEAT_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_weights, svm.weights(),
                              D_FEAT_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        weights_on_device = true;
    }

    // ── Ensure device buffers are large enough ────────────────────────────
    ensure_buf((void**)&d_gray,  &d_gray_cap,  img_h * img_w,          sizeof(float));
    ensure_buf((void**)&d_feat,  &d_feat_cap,  n_wins * D_FEAT_DIM,    sizeof(float));
    ensure_buf((void**)&d_dets,  &d_dets_cap,  max_dets,               sizeof(BoundingBox));
    if (!d_det_cnt) CUDA_CHECK(cudaMalloc(&d_det_cnt, sizeof(int)));

    // ── H2D: copy grayscale frame ─────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_gray, gray_f32,
                          img_h * img_w * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_det_cnt, 0, sizeof(int)));

    // ── HOG kernel ────────────────────────────────────────────────────────
    // Each CUDA block handles one (window, HOG-block) pair → 49 blocks per window
    dim3 hog_grid(n_win_x, n_win_y, D_N_BLOCKS);
    hog_kernel<<<hog_grid, D_TPB>>>(
        d_gray, img_w, img_h, d_feat, n_win_x);
    CUDA_CHECK(cudaGetLastError());

    // ── SVM kernel ────────────────────────────────────────────────────────
    dim3 svm_grid(n_win_x, n_win_y);
    svm_kernel<<<svm_grid, D_TPB>>>(
        d_feat, d_weights, svm.bias(), threshold,
        n_win_x, d_dets, d_det_cnt, max_dets);
    CUDA_CHECK(cudaGetLastError());

    // ── D2H: retrieve detections ─────────────────────────────────────────
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_det_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    h_count = std::min(h_count, max_dets);

    std::vector<BoundingBox> result(h_count);
    if (h_count > 0)
        CUDA_CHECK(cudaMemcpy(result.data(), d_dets,
                              h_count * sizeof(BoundingBox),
                              cudaMemcpyDeviceToHost));

    return result;
}
