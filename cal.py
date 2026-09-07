import cv2
import numpy as np
import glob
import os
import json

CHESSBOARD = (7, 9)       # inner corners
SQUARE_SIZE = 0.020       # meters — 20 mm

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Prepare object points
objp = np.zeros(
    (CHESSBOARD[0] * CHESSBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD[0],
    0:CHESSBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob("images/*.jpg")

if not images:
    raise SystemExit(
        "No images found in images/ — copy them from the Pi first."
    )

used = 0
img_shape = None

# Process calibration images
for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print(f"Could not read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD,
        None
    )

    if found:

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_refined)

        used += 1

        vis = cv2.drawChessboardCorners(
            img.copy(),
            CHESSBOARD,
            corners_refined,
            found
        )

        cv2.imshow("Corners", vis)
        cv2.waitKey(200)

    else:
        print(f"Corners NOT found in {fname} (skipped)")

cv2.destroyAllWindows()

print(f"\nUsed {used}/{len(images)} images.")

if used < 10:
    print(
        "WARNING: fewer than 10 good images — "
        "capture more for reliable results."
    )

if used == 0:
    raise SystemExit("No valid calibration images found.")

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_shape,
    None,
    None
)

print("\n--- Calibration result ---")

print(
    "RMS re-projection error:",
    ret,
    "(good: <0.5px, ok: <1.0px)"
)

print("Camera matrix:\n", mtx)

print("Distortion coefficients:\n", dist.ravel())

# Create output directory
os.makedirs("calibration_data", exist_ok=True)

# Convert NumPy arrays to normal Python lists
calibration_data = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.ravel().tolist(),
    "image_width": img_shape[0],
    "image_height": img_shape[1],
    "chessboard": {
        "columns": CHESSBOARD[0],
        "rows": CHESSBOARD[1]
    },
    "square_size_meters": SQUARE_SIZE,
    "rms_reprojection_error": float(ret),
    "images_used": used,
    "total_images": len(images)
}

# Save as JSON
json_path = "calibration_data/calibration.json"

with open(json_path, "w") as f:
    json.dump(calibration_data, f, indent=4)

print(f"\nSaved calibration data to: {json_path}")
