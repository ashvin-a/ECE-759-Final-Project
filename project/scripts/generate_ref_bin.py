import cv2, sys, os

from hog_utils import extract_hog

SCRIPT_DIR = os.path.dirname("project/results/")
patch = cv2.imread(os.path.join(SCRIPT_DIR, "patch.png"))
feat  = extract_hog(patch)
out   = os.path.join(SCRIPT_DIR, "ref_feat.bin")
feat.tofile(out)
print("Saved", out, feat.shape)
