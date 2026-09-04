# calibrate.py
import cv2
import numpy as np
import glob
import os

CHESSBOARD = (7,9)       # inner corners: (7-1, 9-1)
SQUARE_SIZE = 0.020        # meters — change if your printed square isn't 20mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []
images = glob.glob("images/*.jpg")
if not images:
    raise SystemExit("No images found in images/ — copy them from the Pi first.")

used = 0
img_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1
        vis = cv2.drawChessboardCorners(img.copy(), CHESSBOARD, corners_refined, found)
        cv2.imshow("Corners", vis)
        cv2.waitKey(200)
    else:
        print(f"Corners NOT found in {fname} (skipped)")

cv2.destroyAllWindows()
print(f"\nUsed {used}/{len(images)} images.")
if used < 10:
    print("WARNING: fewer than 10 good images — capture more for reliable results.")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

print("\n--- Calibration result ---")
print("RMS re-projection error:", ret, "(good: <0.5px, ok: <1.0px)")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

os.makedirs("calibration_data", exist_ok=True)
np.savez("calibration_data/calibration.npz", mtx=mtx, dist=dist)
print("\nSaved calibration_data/calibration.npz")