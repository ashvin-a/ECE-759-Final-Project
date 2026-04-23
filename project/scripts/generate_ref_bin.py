import cv2, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from hog_utils import extract_hog

patch = cv2.imread(os.path.join(SCRIPT_DIR, "patch.png"))
feat  = extract_hog(patch)
out   = os.path.join(SCRIPT_DIR, "ref_feat.bin")
feat.tofile(out)
print("Saved", out, feat.shape)
