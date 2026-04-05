#include "nms.h"

#include <algorithm>
#include <cmath>

static float iou(const BoundingBox& a, const BoundingBox& b)
{
    int ax2 = a.x + a.width,  ay2 = a.y + a.height;
    int bx2 = b.x + b.width,  by2 = b.y + b.height;

    int ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
    int ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;

    float inter = static_cast<float>((ix2 - ix1) * (iy2 - iy1));
    float area_a = static_cast<float>(a.width * a.height);
    float area_b = static_cast<float>(b.width * b.height);
    return inter / (area_a + area_b - inter);
}

std::vector<BoundingBox> nms(std::vector<BoundingBox> boxes, float iou_thresh)
{
    if (boxes.empty()) return {};

    // Sort by score descending
    std::sort(boxes.begin(), boxes.end(),
              [](const BoundingBox& a, const BoundingBox& b) {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BoundingBox> kept;

    for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(boxes[i]);
        for (std::size_t j = i + 1; j < boxes.size(); ++j) {
            if (!suppressed[j] && iou(boxes[i], boxes[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }

    return kept;
}
