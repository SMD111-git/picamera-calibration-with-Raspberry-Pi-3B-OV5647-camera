# undistort_images.py
import cv2
import numpy as np
import glob
import os

data = np.load("calibration_data/calibration.npz")
mtx, dist = data["mtx"], data["dist"]

IN_DIR = "lane_images"
OUT_DIR = "lane_images_undistorted"
os.makedirs(OUT_DIR, exist_ok=True)

images = glob.glob(os.path.join(IN_DIR, "*.jpg"))
if not images:
    raise SystemExit("No images found in lane_images/ — run capture_lane_images.py first.")

for path in images:
    img = cv2.imread(path)
    h, w = img.shape[:2]

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

    filename = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(out_path, undistorted)
    print(f"Saved {out_path}")

print(f"\nDone. {len(images)} images corrected -> {OUT_DIR}/")