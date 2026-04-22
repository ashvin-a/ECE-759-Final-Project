#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "hog.h"
#include "nms.h"
#include "sliding_window.h"
#include "sliding_window_cuda.h"
#include "svm.h"
#include "types.h"

static constexpr int BENCHMARK_FRAMES = 300;

static void draw_boxes(cv::Mat& img, const std::vector<BoundingBox>& boxes)
{
    for (const auto& b : boxes) {
        cv::rectangle(img,
                      cv::Rect(b.x, b.y, b.width, b.height),
                      cv::Scalar(0, 255, 0), 2);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", b.score);
        cv::putText(img, buf,
                    cv::Point(b.x, std::max(b.y - 4, 0)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 255, 0), 1);
    }
}

static void print_stats(const std::vector<double>& latencies_ms)
{
    if (latencies_ms.empty()) return;

    std::vector<double> sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double avg = sum / sorted.size();
    double p99 = sorted[static_cast<std::size_t>(sorted.size() * 0.99)];
    double p50 = sorted[sorted.size() / 2];
    double fps = 1000.0 / avg;

    std::cout << "\n── Benchmark Results ──────────────────────────────\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Frames        : " << sorted.size() << "\n";
    std::cout << "  Avg latency   : " << avg  << " ms\n";
    std::cout << "  p50 latency   : " << p50  << " ms\n";
    std::cout << "  p99 latency   : " << p99  << " ms\n";
    std::cout << "  FPS           : " << fps  << "\n";
    std::cout << "───────────────────────────────────────────────────\n";
}

static void usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s <input> <weights.bin> <bias.txt> [output] [threshold] [--mode seq|omp|cuda]\n"
        "  input        : path to video file or image\n"
        "  weights.bin  : binary float32 SVM weights (%d values)\n"
        "  bias.txt     : SVM bias scalar\n"
        "  output       : (optional) path to write annotated output (.mp4/.avi/.png)\n"
        "  threshold    : (optional) SVM decision threshold (default 0.0)\n"
        "  --mode       : seq (sequential, default), omp (OpenMP), or cuda (GPU)\n",
        prog, HOG_FEAT_DIM);
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const std::string input_path   = argv[1];
    const std::string weights_path = argv[2];
    const std::string bias_path    = argv[3];
    const std::string output_path  = (argc >= 5) ? argv[4] : "";
    const float       threshold    = (argc >= 6) ? std::stof(argv[5]) : 0.7f;

    // Parse --mode flag (can appear anywhere after the first 3 required args)
    enum class Mode { SEQ, OMP, CUDA } mode = Mode::SEQ;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            std::string m = argv[i + 1];
            if (m == "omp")  mode = Mode::OMP;
            else if (m == "cuda") mode = Mode::CUDA;
            else              mode = Mode::SEQ;
        }
    }
    const char* mode_str = (mode == Mode::OMP)  ? "OpenMP"
                         : (mode == Mode::CUDA) ? "CUDA"
                                                : "sequential";
    std::cout << "Mode: " << mode_str << "\n";

    // Load SVM
    LinearSVM svm(weights_path, bias_path);
    std::cout << "Loaded SVM  weights=" << weights_path
              << "  bias=" << svm.bias() << "\n";

    // Open input
    // Try as image first; only open VideoCapture if imread fails.
    cv::Mat static_img = cv::imread(input_path);
    bool is_video = static_img.empty();

    cv::VideoCapture cap;
    if (is_video) {
        cap.open(input_path);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Could not open input as image or video: " << input_path << "\n";
            return 1;
        }
    }

    // Open output writer (optional)
    cv::VideoWriter writer;
    if (!output_path.empty() && is_video) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps_in = cap.get(cv::CAP_PROP_FPS);
        if (fps_in <= 0) fps_in = 30.0;
        int out_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int out_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(output_path, fourcc, fps_in, cv::Size(out_w, out_h));
        if (!writer.isOpened())
            std::cerr << "Warning: could not open output writer for " << output_path << "\n";
    }

    // Process frames
    std::vector<double> latencies;
    latencies.reserve(BENCHMARK_FRAMES);

    int frame_count = 0;

    auto process_frame = [&](cv::Mat& frame) {
        // Start timing AFTER decode
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<BoundingBox> raw  = (mode == Mode::OMP)  ? sliding_window_omp(frame, svm, threshold)
                                         : (mode == Mode::CUDA) ? sliding_window_cuda(frame, svm, threshold)
                                                                : sliding_window(frame, svm, threshold);
        std::vector<BoundingBox> kept = nms(raw);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);

        // Draw and write output
        if (!output_path.empty() || frame_count % 30 == 0) {
            draw_boxes(frame, kept);
        }
        if (writer.isOpened()) writer.write(frame);

        if (frame_count % 30 == 0) {
            std::printf("Frame %4d  |  raw=%3zu  kept=%2zu  latency=%.1f ms\n",
                        frame_count, raw.size(), kept.size(), ms);
        }
        ++frame_count;
    };

    if (is_video) {
        cv::Mat frame;
        while (frame_count < BENCHMARK_FRAMES && cap.read(frame)) {
            process_frame(frame);
        }
        if (frame_count < BENCHMARK_FRAMES)
            std::cout << "Video ended after " << frame_count << " frames.\n";
    } else {
        // Single image — run BENCHMARK_FRAMES times for stable timing
        cv::Mat& img = static_img;
        std::cout << "Image mode: running " << BENCHMARK_FRAMES
                  << " iterations for benchmark.\n";
        cv::Mat annotated;
        for (int i = 0; i < BENCHMARK_FRAMES; ++i) {
            cv::Mat copy = img.clone();
            process_frame(copy);
            if (i == 0) annotated = copy;  // keep first annotated frame for output
        }
        if (!output_path.empty()) {
            // Ensure standard 3-channel BGR before writing
            if (annotated.channels() == 4)
                cv::cvtColor(annotated, annotated, cv::COLOR_BGRA2BGR);
            if (!cv::imwrite(output_path, annotated))
                std::cerr << "WARNING: Failed to write output image: " << output_path << "\n";
            else
                std::cout << "Saved annotated image: " << output_path << "\n";
        }
    }

    print_stats(latencies);
    return 0;
}
