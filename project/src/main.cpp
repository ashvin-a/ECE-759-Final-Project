#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
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

static std::string derive_csv_path(const std::string& output_path, const std::string& mode_str)
{
    if (!output_path.empty()) {
        auto dot = output_path.rfind('.');
        std::string base = (dot != std::string::npos) ? output_path.substr(0, dot) : output_path;
        return base + "_" + mode_str + "_results.csv";
    }
    return mode_str + "_results.csv";
}

struct FrameRecord {
    int frame;
    std::size_t raw;
    std::size_t kept;
    double latency_ms;
};

static void write_results(const std::string& csv_path,
                          const std::string& mode_str,
                          const std::vector<FrameRecord>& records)
{
    std::ofstream f(csv_path);
    if (!f) {
        std::cerr << "WARNING: could not write results to " << csv_path << "\n";
        return;
    }

    f << "mode,frame,raw_detections,kept_detections,latency_ms\n";
    for (const auto& r : records)
        f << mode_str << "," << r.frame << "," << r.raw << "," << r.kept << ","
          << std::fixed << std::setprecision(4) << r.latency_ms << "\n";

    std::cout << "Saved per-frame results: " << csv_path << "\n";
}

static void print_stats(const std::vector<double>& latencies_ms,
                        const std::string& csv_path,
                        const std::string& mode_str)
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

    // Append summary block to the CSV
    std::ofstream f(csv_path, std::ios::app);
    if (!f) return;
    f << "\nsummary_metric,value\n";
    f << std::fixed << std::setprecision(4);
    f << "mode," << mode_str << "\n";
    f << "frames," << sorted.size() << "\n";
    f << "avg_latency_ms," << avg << "\n";
    f << "p50_latency_ms," << p50 << "\n";
    f << "p99_latency_ms," << p99 << "\n";
    f << "fps," << fps << "\n";
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
    // First pass: extract flags, collect positional args separately
    enum class Mode { SEQ, OMP, CUDA } mode = Mode::SEQ;
    std::vector<std::string> pos;  // non-flag arguments
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "omp")  mode = Mode::OMP;
            else if (m == "cuda") mode = Mode::CUDA;
            else                  mode = Mode::SEQ;
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() < 3) { usage(argv[0]); return 1; }

    const std::string input_path   = pos[0];
    const std::string weights_path = pos[1];
    const std::string bias_path    = pos[2];
    const std::string output_path  = (pos.size() >= 4) ? pos[3] : "";
    const float       threshold    = (pos.size() >= 5) ? std::stof(pos[4]) : 0.7f;
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

    const std::string csv_path = derive_csv_path(output_path, mode_str);

    // Process frames
    std::vector<double> latencies;
    std::vector<FrameRecord> records;
    latencies.reserve(BENCHMARK_FRAMES);
    records.reserve(BENCHMARK_FRAMES);

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
        records.push_back({frame_count, raw.size(), kept.size(), ms});

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

    write_results(csv_path, mode_str, records);
    print_stats(latencies, csv_path, mode_str);
    return 0;
}
