#pragma once

#include <vector>
#include "types.h"

/**
 * Greedy IoU-based Non-Maximum Suppression.
 *
 * Sorts detections by SVM score (descending) and suppresses any box
 * whose IoU with a higher-scoring kept box exceeds iou_thresh.
 *
 * @param boxes       Input detections (modified in-place by sorting).
 * @param iou_thresh  Suppression threshold (default NMS_IOU_THRESH = 0.4).
 * @return            Filtered list of kept BoundingBoxes.
 */
std::vector<BoundingBox> nms(std::vector<BoundingBox> boxes,
                              float iou_thresh = NMS_IOU_THRESH);
