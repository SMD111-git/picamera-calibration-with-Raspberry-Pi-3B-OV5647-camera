import cv2
import numpy as np
import glob
import os

# ============================================================
# SETTINGS
# ============================================================

# Your physical chessboard:
# 7 x 9 squares
#
# Therefore:
# 6 x 8 INNER corners
CHESSBOARD = (6, 8)

# Each square is 20 mm x 20 mm
# 20 mm = 0.020 meters
SQUARE_SIZE = 0.020

# ============================================================
# CORNER REFINEMENT CRITERIA
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ============================================================
# CREATE 3D OBJECT POINTS
# ============================================================

# 6 x 8 = 48 inner corners
objp = np.zeros(
    (CHESSBOARD[0] * CHESSBOARD[1], 3),
    np.float32
)

# Create X,Y coordinates
objp[:, :2] = np.mgrid[
    0:CHESSBOARD[0],
    0:CHESSBOARD[1]
].T.reshape(-1, 2)

# Convert from square units to meters
objp *= SQUARE_SIZE

# ============================================================
# STORAGE
# ============================================================

objpoints = []       # 3D real-world points
imgpoints = []       # 2D image points
used_images = []     # Images where corners were detected

# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(glob.glob("images/*.jpg"))

if not images:
    raise SystemExit(
        "No images found in images/ — "
        "copy your calibration images into the images/ folder."
    )

print(f"Found {len(images)} images.")

used = 0
img_shape = None

# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for fname in images:

    print(f"\nProcessing: {fname}")

    img = cv2.imread(fname)

    if img is None:
        print("ERROR: Could not read image — skipped")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image resolution
    img_shape = gray.shape[::-1]

    # ========================================================
    # FIND CHESSBOARD CORNERS
    # ========================================================

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHESSBOARD
    )

    # ========================================================
    # IF CORNERS FOUND
    # ========================================================

    if found:

        # SB detector already provides accurate corners.
        # We can optionally refine them further.
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Store points
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used_images.append(fname)

        used += 1

        print("Corners FOUND")

        # Draw detected corners
        vis = cv2.drawChessboardCorners(
            img.copy(),
            CHESSBOARD,
            corners_refined,
            found
        )

        cv2.imshow("Chessboard Corners", vis)

        # Show image for 300 ms
        key = cv2.waitKey(300)

        # Press ESC to stop displaying images
        if key == 27:
            break

    else:

        print("Corners NOT found — skipped")

# ============================================================
# FINISHED PROCESSING
# ============================================================

cv2.destroyAllWindows()

print("\n==========================================")
print("IMAGE PROCESSING COMPLETE")
print("==========================================")

print(f"Used {used}/{len(images)} images.")

# ============================================================
# CHECK NUMBER OF GOOD IMAGES
# ============================================================

if used < 10:

    print(
        "\nWARNING:"
        "\nFewer than 10 good images were detected."
        "\nFor reliable calibration, capture more images."
    )

if used < 3:

    raise SystemExit(
        "\nERROR: Not enough valid calibration images."
        "\nAt least 3 images are required."
    )

# ============================================================
# CAMERA CALIBRATION
# ============================================================

print("\n==========================================")
print("RUNNING CAMERA CALIBRATION")
print("==========================================")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_shape,
    None,
    None
)

# ============================================================
# PER-IMAGE REPROJECTION ERROR
# ============================================================

print("\n==========================================")
print("PER-IMAGE REPROJECTION ERRORS")
print("==========================================")

errors = []

for i in range(len(objpoints)):

    # Project the known 3D chessboard points
    # back onto the image
    projected, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        mtx,
        dist
    )

    # Convert both to Nx2
    observed = imgpoints[i].reshape(-1, 2)
    projected = projected.reshape(-1, 2)

    # Calculate error
    error = cv2.norm(
        observed,
        projected,
        cv2.NORM_L2
    ) / len(projected)

    errors.append(error)

    print(
        f"{used_images[i]}: "
        f"{error:.3f} px"
    )

# ============================================================
# SORT WORST IMAGES
# ============================================================

print("\n==========================================")
print("WORST IMAGES")
print("==========================================")

sorted_errors = sorted(
    zip(errors, used_images),
    reverse=True
)

for error, filename in sorted_errors:

    print(
        f"{error:.3f} px  ->  {filename}"
    )

# ============================================================
# CALIBRATION RESULT
# ============================================================

print("\n==========================================")
print("CALIBRATION RESULT")
print("==========================================")

print(
    f"\nRMS re-projection error: "
    f"{ret:.6f} px"
)

print("\nCamera matrix:")
print(mtx)

print("\nDistortion coefficients:")
print(dist.ravel())

# ============================================================
# INTERPRET RMS ERROR
# ============================================================

print("\n------------------------------------------")

if ret < 0.5:
    print("Excellent calibration (< 0.5 px)")

elif ret < 1.0:
    print("Good calibration (< 1.0 px)")

elif ret < 2.0:
    print("Calibration is usable, but could be improved.")

else:
    print("Calibration error is high. Check images and board.")

print("------------------------------------------")

# ============================================================
# SAVE CALIBRATION
# ============================================================

os.makedirs(
    "calibration_data",
    exist_ok=True
)

output_file = (
    "calibration_data/calibration.npz"
)

np.savez(
    output_file,
    mtx=mtx,
    dist=dist
)

# ============================================================
# DONE
# ============================================================

print("\n==========================================")
print("CALIBRATION COMPLETE")
print("==========================================")

print(
    f"\nSaved calibration file:\n"
    f"{output_file}"
)
